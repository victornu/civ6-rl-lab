"""Bounded JSON-lines client for the persistent remote policy worker.

Pass the SSH command as argv (for example ``["ssh", host, "python3", "-u",
"-m", "civ6rl.learning.worker"]``). No shell or payload-provided path is used.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import queue
import subprocess
import threading
import time
from collections.abc import Sequence
from typing import Any
from .wire import (MAX_LINE_BYTES, EXTENDED_MAX_LINE_BYTES,
                   EXTENDED_MAX_DECODED_BYTES, decode_line, encode_line)
from .diagnostic_budget998 import validate_collection, is_extended


def _campaign_summary(value,contract):
    from .campaign import validate_summary
    validate_summary(value,contract)
    return json.loads(json.dumps(value))


def _campaign_ref(value,suffix):
    if (type(value) is not dict or set(value)!={'path','sha256'}
            or type(value['sha256']) is not str or len(value['sha256'])!=64
            or any(c not in '0123456789abcdef' for c in value['sha256'])
            or value['path']!='learner-artifacts/'+value['sha256']+suffix):
        raise ValueError('invalid campaign artifact reference')
    return json.loads(json.dumps(value))

MAX_REQUESTS = 16
MAX_FROZEN_REQUESTS = 65
MAX_REQUEST_TIMEOUT = 20.0
MAX_SESSION_TIMEOUT = 180.0
MAX_STDERR_BYTES = 64 * 1024
MAX_LOCAL_MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
MIN_LOCAL_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024


class TransportError(RuntimeError):
    """Worker process or wire protocol failure."""


def _json_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _encode_line(value: dict, *, extended=False) -> bytes:
    try:
        if extended:
            return encode_line(value, max_line_bytes=EXTENDED_MAX_LINE_BYTES,
                               max_decoded_bytes=EXTENDED_MAX_DECODED_BYTES)
        return encode_line(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TransportError("request wire encoding failed: " + str(exc)[:160]) from exc


def _identity_and_candidates(response: dict) -> tuple[str, int, int, set[str]]:
    try:
        episode = response["episode_id"]
        player = response["observation"]["player_id"]
        sequence = response["sequence"]
        candidates = response["decisions"]["candidates"]
    except (KeyError, TypeError) as exc:
        raise TransportError("native response is missing policy identity or candidates") from exc
    if not isinstance(episode, str) or not episode or type(player) is not int or type(sequence) is not int or sequence < 0:
        raise TransportError("invalid episode/player/sequence identity")
    if not isinstance(candidates, list) or not candidates:
        raise TransportError("native candidate menu is empty")
    ids = {}
    for row in candidates:
        if (not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str)
                or not row["candidate_id"] or not isinstance(row.get("action"), dict)):
            raise TransportError("native candidate ID is malformed")
        if row["candidate_id"] in ids:
            raise TransportError("native candidate IDs are not unique")
        ids[row["candidate_id"]] = row["action"]
    return episode, player, sequence, ids


class WorkerClient:
    """Persistent bounded stdio client; ``argv`` is the caller's SSH argv.

    Counts include initialization. More than 16 requests requires a bounded
    frozen evaluation or diagnostic collection contract. The episode-budget
    contract also bounds request and session timeouts.
    """

    def __init__(self, argv: Sequence[str], *, request_timeout: float = 20.0,
                 total_timeout: float = 180.0, max_requests: int = 16,
                 local_memory_limit_bytes: int | None = None, episode_budget=None,
                 campaign_contract=None):
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("argv must be a nonempty argument sequence")
        if any(not isinstance(arg, str) or not arg for arg in argv):
            raise ValueError("argv entries must be nonempty strings")
        budget = validate_collection(episode_budget) if episode_budget is not None else None
        extended = budget is not None and is_extended(budget)
        if budget is not None and not extended:
            raise ValueError('episode_budget requires the bounded episode schema')
        if campaign_contract is not None:
            from .campaign import validate_contract
            campaign_contract=validate_contract(campaign_contract)
            if not extended:
                raise ValueError('campaign contract requires an extended episode budget')
        max_request_timeout = budget['request_seconds'] if extended else MAX_REQUEST_TIMEOUT
        if campaign_contract is not None:
            from .campaign import campaign_session_seconds
            max_session_timeout=campaign_session_seconds(budget,campaign_contract)
        else:
            max_session_timeout=budget['wall_seconds'] if extended else MAX_SESSION_TIMEOUT
        max_allowed_requests = budget['max_decisions'] + 2 if extended else MAX_FROZEN_REQUESTS
        if not 0 < request_timeout <= max_request_timeout:
            raise ValueError(f"request_timeout must be in (0, {max_request_timeout}] seconds")
        if not 0 < total_timeout <= max_session_timeout:
            raise ValueError(f"total_timeout must be in (0, {max_session_timeout}] seconds")
        if type(max_requests) is not int or not 1 <= max_requests <= max_allowed_requests:
            raise ValueError(f'max_requests must be an integer in [1, {max_allowed_requests}]')
        if local_memory_limit_bytes is not None:
            if os.name != "nt":
                raise ValueError("local_memory_limit_bytes is supported only on Windows")
            if (type(local_memory_limit_bytes) is not int
                    or not MIN_LOCAL_MEMORY_LIMIT_BYTES <= local_memory_limit_bytes <= MAX_LOCAL_MEMORY_LIMIT_BYTES):
                raise ValueError("local_memory_limit_bytes must be an integer in [64 MiB, 8 GiB]")
        self.max_requests = max_requests
        self._episode_budget = budget
        self._campaign_contract = campaign_contract
        self._extended_wire_declared = extended
        self._extended_wire_active = False
        self._reader_line_limit = EXTENDED_MAX_LINE_BYTES if extended else MAX_LINE_BYTES
        # A requested extension is inactive until the frozen init handshake.
        self._request_limit = min(max_requests, MAX_REQUESTS)
        self.request_timeout = float(request_timeout)
        self.total_timeout = float(total_timeout)
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._requests = 0
        self._next_id = 1
        self._initialized = False
        self.policy_version = None
        self._closed = False
        self.last_exchange_wire = None
        self._stderr = bytearray()
        self._local_memory_limit_bytes = local_memory_limit_bytes
        self._job_handle = None
        self._local_resource_summary = None
        self._stderr_lock = threading.Lock()
        self._lines: queue.Queue[bytes | BaseException] = queue.Queue(maxsize=2)
        creationflags = 0
        if os.name == "nt":
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if local_memory_limit_bytes is not None:
                creationflags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x4000)
        try:
            self._process = subprocess.Popen(
                list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0, creationflags=creationflags,
            )
        except OSError as exc:
            self._closed = True
            raise TransportError("could not start policy worker") from exc
        if local_memory_limit_bytes is not None:
            try:
                # The local worker waits for this byte line before importing Torch.
                from ..native_probe import Windows
                self._windows = Windows()
                self._job_handle = self._windows.job(self._process, local_memory_limit_bytes)
                assert self._process.stdin is not None
                self._process.stdin.write(b"go\n")
                self._process.stdin.flush()
            except BaseException as exc:
                self._closed = True
                try:
                    self._terminate_reap()
                finally:
                    self._close_job()
                raise TransportError("could not contain and start local policy worker") from exc
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    @property
    def stderr(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr).decode("utf-8", errors="replace")

    @property
    def local_resource_summary(self) -> dict | None:
        """Latest Windows process memory counters, when local containment is enabled."""
        return None if self._local_resource_summary is None else dict(self._local_resource_summary)

    def _capture_local_resources(self) -> None:
        if self._job_handle is None:
            return
        try:
            self._local_resource_summary = self._windows.memory(self._process._handle)
        except (OSError, RuntimeError):
            pass

    def _close_job(self) -> None:
        handle, self._job_handle = self._job_handle, None
        if handle is not None:
            try:
                self._windows.kernel.CloseHandle(handle)
            except Exception:
                pass

    def _read_stdout(self) -> None:
        try:
            assert self._process.stdout is not None
            while True:
                line = self._process.stdout.readline(self._reader_line_limit + 1)
                if not line:
                    self._lines.put(EOFError("worker closed stdout"))
                    return
                if len(line) > self._reader_line_limit or not line.endswith(b"\n"):
                    self._lines.put(TransportError("worker response exceeds framing limit"))
                    return
                self._lines.put(line)
        except BaseException as exc:
            try:
                self._lines.put_nowait(exc)
            except queue.Full:
                pass

    def _drain_stderr(self) -> None:
        try:
            assert self._process.stderr is not None
            while True:
                data = self._process.stderr.read(4096)
                if not data:
                    return
                with self._stderr_lock:
                    remaining = MAX_STDERR_BYTES - len(self._stderr)
                    if remaining > 0:
                        self._stderr.extend(data[:remaining])
        except (OSError, ValueError):
            return

    def _terminate_reap(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        # Capture Windows peak counters while the process is still queryable.
        self._capture_local_resources()
        try:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired as exc:
                        raise TransportError("worker survived bounded termination waits") from exc
            else:
                process.wait()
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            self._close_job()

    def _fail_closed(self) -> None:
        self._closed = True
        self._terminate_reap()

    def _exchange(self, payload: dict, *, request_timeout: float | None = None) -> dict:
        if self._closed:
            raise TransportError("worker client is closed")
        if self._requests >= self._request_limit:
            self._fail_closed()
            raise TransportError("worker request limit exceeded")
        remaining_total = self.total_timeout - (time.monotonic() - self._started)
        if request_timeout is None:
            request_timeout=self.request_timeout
        elif (self._campaign_contract is None or type(request_timeout) not in (int,float)
                or not 0<request_timeout<=600):
            raise TransportError('campaign update timeout is outside its explicit bound')
        timeout = min(float(request_timeout), remaining_total)
        if timeout <= 0:
            self._fail_closed()
            raise TransportError("worker session timed out")
        request_id = self._next_id
        self._next_id += 1
        payload = {"id": request_id, **payload}
        data = _encode_line(payload, extended=self._extended_wire_active)
        self._requests += 1
        write_done: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

        def write_request():
            try:
                assert self._process.stdin is not None
                pending = memoryview(data)
                while pending:
                    written = self._process.stdin.write(pending)
                    if not isinstance(written, int) or written <= 0:
                        raise OSError("worker pipe made no write progress")
                    pending = pending[written:]
                self._process.stdin.flush()
                write_done.put(None)
            except BaseException as exc:
                try:
                    write_done.put_nowait(exc)
                except queue.Full:
                    pass

        writer = threading.Thread(target=write_request, daemon=True)
        writer.start()
        deadline = time.monotonic() + timeout
        try:
            result = write_done.get(timeout=max(0.0, deadline - time.monotonic()))
            if result is not None:
                raise TransportError("failed writing worker request") from result
            line = self._lines.get(timeout=max(0.0, deadline - time.monotonic()))
            if isinstance(line, BaseException):
                if isinstance(line, TransportError):
                    raise line
                raise TransportError("worker exited before replying") from line
            try:
                decoded = decode_line(line, **(dict(max_line_bytes=EXTENDED_MAX_LINE_BYTES,
                    max_decoded_bytes=EXTENDED_MAX_DECODED_BYTES) if self._extended_wire_active else {}))
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise TransportError("worker returned malformed JSON") from exc
            if not isinstance(decoded, dict) or decoded.get("id") != request_id:
                raise TransportError("worker response ID mismatch")
            if decoded.get("ok") is not True:
                raise TransportError("worker rejected request: " + str(decoded.get("error", "unknown"))[:200])
            self.last_exchange_wire = dict(request_id=request_id,
                request_line_bytes=len(data), response_line_bytes=len(line))
            return decoded
        except queue.Empty as exc:
            self._fail_closed()
            raise TransportError("worker request timed out; process was killed and reaped") from exc
        except TransportError:
            self._fail_closed()
            raise

    def initialize(self, vocabulary_serialized: str, *, device: str = "cpu", seed: int = 0, checkpoint=None, resume=None, critic_mode=None, evaluation_only=False, diagnostic_collection=None, episode_budget=None, campaign_contract=None, resume_commit=None) -> dict:
        with self._lock:
            if self._initialized or self._requests:
                raise TransportError("worker is already initialized or has received requests")
            if not isinstance(vocabulary_serialized, str) or not vocabulary_serialized:
                raise ValueError("a serialized frozen vocabulary is required")
            if device not in ("cpu", "mps") or type(seed) is not int or seed < 0:
                raise ValueError("invalid device or seed")
            if checkpoint is not None and resume is not None:
                raise ValueError('choose policy warm-start or learner resume, not both')
            if type(evaluation_only) is not bool:
                raise ValueError('evaluation_only must be boolean')
            requested_budget = validate_collection(episode_budget) if episode_budget is not None else None
            if requested_budget is not None and not is_extended(requested_budget):
                raise ValueError('episode_budget requires the bounded episode schema')
            if requested_budget != self._episode_budget:
                raise ValueError('initialize episode_budget must exactly match the constructor contract')
            extended_budget = requested_budget is not None and is_extended(requested_budget)
            if evaluation_only and (checkpoint is None or resume is not None or critic_mode not in (None,'player_only')):
                raise ValueError('frozen evaluation requires actor-only checkpoint')
            collection=None
            if diagnostic_collection is not None:
                collection=validate_collection(diagnostic_collection)
                if evaluation_only:
                    raise ValueError('diagnostic collection requires non-frozen policy')
                if extended_budget:
                    if collection != requested_budget:
                        raise ValueError('diagnostic_collection must exactly equal episode_budget')
                elif is_extended(collection):
                    raise ValueError('extended collection requires an episode budget handshake')
                elif self.max_requests!=collection['max_decisions']+2:
                    raise ValueError('diagnostic max_requests must equal max_decisions plus two')
            elif extended_budget and not evaluation_only:
                raise ValueError('training episode_budget requires matching diagnostic_collection')
            if campaign_contract is not None:
                from .campaign import validate_contract
                campaign_contract=validate_contract(campaign_contract)
                if campaign_contract!=self._campaign_contract:
                    raise ValueError('campaign contract must match the constructor timeout declaration')
                if (resume is None or requested_budget is None or not extended_budget or collection is None
                        or evaluation_only):
                    raise ValueError('campaign requires complete resume, explicit episode budget, and non-frozen collection')
                if resume_commit is not None:
                    resume_commit=_campaign_ref(resume_commit,'.commit.json')
            elif resume_commit is not None:
                raise ValueError('resume_commit requires campaign_contract')
            elif self._campaign_contract is not None:
                raise ValueError('constructor campaign timeout requires the exact init campaign contract')
            if extended_budget and self.max_requests > requested_budget['max_decisions'] + 2:
                raise ValueError('max_requests exceeds episode decision budget')
            elif not extended_budget and collection is None and self.max_requests > MAX_REQUESTS:
                if (not evaluation_only or type(checkpoint) is not dict
                        or set(checkpoint) != {'path', 'sha256'}
                        or type(checkpoint['path']) is not str or not checkpoint['path']
                        or type(checkpoint['sha256']) is not str or len(checkpoint['sha256']) != 64
                        or any(c not in '0123456789abcdef' for c in checkpoint['sha256'])
                        or resume is not None or critic_mode not in (None, 'player_only')):
                    raise ValueError('extended request budget requires frozen actor-only checkpoint evaluation')
            payload={"op": "init", "vocabulary": vocabulary_serialized,"device": device,"seed":seed}
            if extended_budget:
                payload['episode_budget'] = dict(requested_budget)
                payload['request_budget'] = requested_budget['max_decisions'] + 2
            elif self.max_requests > MAX_REQUESTS or collection is not None:
                payload['request_budget'] = self.max_requests
            if collection is not None:payload['diagnostic_collection']=dict(collection)
            if evaluation_only:payload['evaluation_only']=True
            if critic_mode is not None:
                if critic_mode not in ('player_only','privileged_diagnostic','history_only_diagnostic'):raise ValueError('invalid critic mode')
                payload['critic_mode']=critic_mode
            if checkpoint is not None:payload['checkpoint']=checkpoint
            if resume is not None:payload['resume']=resume
            if campaign_contract is not None:payload['campaign_contract']=json.loads(json.dumps(campaign_contract))
            if resume_commit is not None:payload['resume_commit']=resume_commit
            result = self._exchange(payload)
            if result.get("model_status") != ('untrained' if checkpoint is None and resume is None else 'diagnostic_updated') or result.get("training_eligible") is not False:
                self._fail_closed()
                raise TransportError("worker did not confirm an untrained diagnostic policy")
            max_resumed_updates=(4096 if campaign_contract is not None
                else (3 if collection is not None else 4))
            if resume is not None and (result.get('learner_resumed') is not True
                    or type(result.get('completed_updates')) is not int
                    or not 1<=result['completed_updates']<=max_resumed_updates):
                self._fail_closed();raise TransportError('worker did not restore learner counters')
            if campaign_contract is not None:
                try:
                    acknowledged=_campaign_summary(result.get('campaign'),campaign_contract)
                    if result.get('completed_updates')!=acknowledged['origin_updates']+acknowledged['completed_updates']:
                        raise ValueError('campaign cumulative updates differ')
                    if (acknowledged['completed_updates']>0)!=(resume_commit is not None):
                        raise ValueError('campaign pair commit presence differs')
                    if (acknowledged['remaining_updates']<1
                            or requested_budget['max_decisions']>acknowledged['remaining_decisions']):
                        raise ValueError('campaign cannot admit the requested episode budget')
                except (TypeError,ValueError) as exc:
                    self._fail_closed();raise TransportError('worker did not confirm the exact learner campaign') from exc
                self._campaign_contract=json.loads(json.dumps(campaign_contract))
                self._campaign_summary=acknowledged
            elif 'campaign' in result:
                self._fail_closed();raise TransportError('worker returned an unrequested campaign')
            if extended_budget:
                if result.get('request_budget') != requested_budget['max_decisions'] + 2:
                    self._fail_closed();raise TransportError('worker did not confirm the exact episode request budget')
                try:
                    acknowledgement=validate_collection(result.get('episode_budget'))
                    if acknowledgement != requested_budget:
                        raise ValueError('episode budget acknowledgement differs')
                except ValueError as exc:
                    self._fail_closed();raise TransportError('worker did not confirm the exact episode budget') from exc
            elif 'episode_budget' in result:
                self._fail_closed();raise TransportError('worker returned an unrequested episode budget')
            if not extended_budget and (self.max_requests > MAX_REQUESTS or collection is not None):
                if type(result.get('request_budget')) is not int or result['request_budget'] != self.max_requests:
                    self._fail_closed();raise TransportError('worker did not confirm the exact request budget')
            elif not extended_budget and 'request_budget' in result:
                self._fail_closed();raise TransportError('worker returned an unrequested request budget')
            if evaluation_only and result.get('evaluation_only') is not True:
                self._fail_closed();raise TransportError('worker did not freeze evaluation')
            if collection is not None:
                try:
                    acknowledgement=validate_collection(result.get('diagnostic_collection'))
                    if acknowledgement!=collection or result.get('evaluation_only') is not False:
                        raise ValueError('diagnostic collection acknowledgement differs')
                except ValueError as exc:
                    self._fail_closed();raise TransportError('worker did not confirm the exact diagnostic collection') from exc
            elif 'diagnostic_collection' in result:
                self._fail_closed();raise TransportError('worker returned an unrequested diagnostic collection')
            self.policy_version=result.get('policy_version')
            self.critic_mode=result.get('critic_mode','player_only')
            self.critic_version=result.get('critic_version')
            if (self.critic_mode not in ('player_only','privileged_diagnostic','history_only_diagnostic')
                    or evaluation_only and self.critic_mode != 'player_only'
                    or critic_mode is not None and self.critic_mode!=critic_mode
                    or self.critic_mode!='player_only' and (type(self.critic_version) is not str or len(self.critic_version)!=64)):
                self._fail_closed();raise TransportError('Worker critic mode or version differs')
            self._request_limit = self.max_requests
            self._initialized = True
            self._extended_wire_active = extended_budget
            self._evaluation_only=evaluation_only
            if extended_budget:
                self._episode_budget_inferences=0
                self._episode_budget_episode=None
            if collection is not None:
                self._diagnostic_collection=collection
                self._diagnostic_collection_inferences=0
                self._diagnostic_collection_updates=0
                self._diagnostic_collection_episode=None
            return result

    def update_diagnostic(self,records,scenario,*,burn_in=0,epochs=2,privileged_snapshots=None,privileged_context=None):
        return self._update_diagnostic_exchange(
            dict(op='update_diagnostic',records=records,scenario=scenario,burn_in=burn_in,epochs=epochs),
            expected_mode='diagnostic_native',privileged_snapshots=privileged_snapshots,
            privileged_context=privileged_context)

    def update_multiplayer_diagnostic(self,records,scenarios,collection,*,epochs=2,
                                     privileged_snapshots=None,privileged_context=None):
        """Update independent player trajectories from one global collection."""
        return self._update_diagnostic_exchange(
            dict(op='update_multiplayer_diagnostic',records=records,scenarios=scenarios,
                 collection=collection,epochs=epochs),
            expected_mode='diagnostic_native_multiplayer',privileged_snapshots=privileged_snapshots,
            privileged_context=privileged_context)

    def _update_diagnostic_exchange(self,payload,*,expected_mode,privileged_snapshots,privileged_context):
        with self._lock:
            if getattr(self,'_evaluation_only',False):raise TransportError('frozen evaluation refuses updates')
            if not self._initialized or self.policy_version is None:raise TransportError('Versioned policy initialization required')
            collection=getattr(self,'_diagnostic_collection',None)
            if collection is not None:
                if payload['op']!='update_multiplayer_diagnostic':
                    raise TransportError('diagnostic collection permits only a multiplayer update')
                if self._diagnostic_collection_updates>=collection['max_updates']:
                    raise TransportError('diagnostic collection update budget exhausted')
            payload['policy_version']=self.policy_version
            if self.critic_mode=='privileged_diagnostic':
                if privileged_snapshots is None or privileged_context is None:
                    raise ValueError('privileged learner needs its separate complete sidecar')
                payload.update(privileged_snapshots=privileged_snapshots,privileged_context=privileged_context,
                               critic_version=self.critic_version)
            elif privileged_snapshots is not None or privileged_context is not None:
                raise ValueError('player-only learner cannot receive private sidecar')
            if self.critic_mode=='history_only_diagnostic':
                payload['critic_version']=self.critic_version
            expected_context=None
            allow_single=False
            if payload['op']=='update_multiplayer_diagnostic':
                requested=payload['collection']
                if type(requested) is dict and 'allow_single_public_skill_group' in requested:
                    from .skill_context993 import validate_skill_scenario
                    from .skill_task991 import validate_public_sources
                    if requested['allow_single_public_skill_group'] is not True or collection is None:
                        raise ValueError('single public skill group requires an explicit diagnostic budget')
                    validate_collection(collection)
                    validate_public_sources(requested.get('public_continuity_sources'))
                    if type(payload['scenarios']) is not list or not 1<=len(payload['scenarios'])<=4:
                        raise ValueError('single public skill group requires skill scenarios')
                    for row in payload['scenarios']:
                        if (type(row) is not dict or set(row)!={'player_id','scenario','burn_in'}
                                or type(row['player_id']) is not int or not 0<=row['player_id']<=63):
                            raise ValueError('single public skill group requires skill scenarios')
                        validate_skill_scenario(row['scenario'])
                    allow_single=True
                expected_context=hashlib.sha256(json.dumps(
                    dict(scenarios=payload['scenarios'],collection=payload['collection']),
                    sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
            campaign_contract=getattr(self,'_campaign_contract',None)
            if campaign_contract is not None:
                from .campaign import update_timeout_seconds
                update_timeout=update_timeout_seconds(len(payload['records']))
            else:
                update_timeout=None
            result=self._exchange(payload,request_timeout=update_timeout)
            accepted_mode=result.get('mode')==expected_mode
            if allow_single:
                groups=result.get('player_groups')
                group_ids=[g.get('player_id') for g in groups] if type(groups) is list and all(type(g) is dict for g in groups) else []
                declared={row['player_id'] for row in payload['scenarios']}
                valid_groups=(bool(group_ids) and all(type(pid) is int and pid in declared for pid in group_ids)
                              and len(group_ids)==len(set(group_ids)))
                accepted_mode=valid_groups and ((len(group_ids)==1 and result.get('mode')=='diagnostic_native_singleplayer_skill')
                    or (len(group_ids)>=2 and result.get('mode')==expected_mode))
            if (result.get('training_eligible') is not False or not accepted_mode
                    or result.get('model_status')!='diagnostic_updated' or result.get('previous_policy_version')!=self.policy_version
                    or type(result.get('policy_version')) is not str or len(result['policy_version'])!=64):
                self._fail_closed();raise TransportError('Invalid diagnostic update receipt')
            if expected_context is not None and (result.get('scenario_sha256')!=expected_context
                    or result.get('collection')!=payload['collection']):
                self._fail_closed();raise TransportError('Multiplayer collection binding differs')
            campaign_contract=getattr(self,'_campaign_contract',None)
            if campaign_contract is not None:
                try:
                    previous=self._campaign_summary
                    acknowledged=_campaign_summary(result.get('campaign'),campaign_contract)
                    if (acknowledged['campaign_id']!=previous['campaign_id']
                            or acknowledged['contract_sha256']!=previous['contract_sha256']
                            or acknowledged['origin_updates']!=previous['origin_updates']
                            or acknowledged['completed_updates']!=previous['completed_updates']+1
                            or acknowledged['decisions']!=previous['decisions']+len(payload['records'])
                            or acknowledged['optimized_decisions']<=previous['optimized_decisions']
                            or acknowledged['optimized_decisions']>acknowledged['decisions']
                            or type(result.get('completed_updates')) is not int
                            or result['completed_updates']!=acknowledged['origin_updates']+acknowledged['completed_updates']):
                        raise ValueError('campaign update counters differ')
                    commit_ref=_campaign_ref(result.get('campaign_commit'),'.commit.json')
                except (TypeError,ValueError) as exc:
                    self._fail_closed();raise TransportError('Invalid learner campaign update receipt') from exc
                self._campaign_summary=acknowledged
                result['campaign_commit']=commit_ref
                result['campaign']=acknowledged
            elif 'campaign' in result or 'campaign_commit' in result:
                self._fail_closed();raise TransportError('worker returned an unrequested campaign update')
            if self.critic_mode!='player_only':
                if (result.get('critic_mode')!=self.critic_mode or result.get('previous_critic_version')!=self.critic_version
                        or type(result.get('critic_version')) is not str or len(result['critic_version'])!=64):
                    self._fail_closed();raise TransportError('Invalid privileged update receipt')
                self.critic_version=result['critic_version']
            self.policy_version=result['policy_version']
            if collection is not None:self._diagnostic_collection_updates+=1
            return result

    def infer(self, native_response: dict, *, scenario=None, completed_decisions=None) -> dict:
        with self._lock:
            if not self._initialized:
                raise TransportError("initialize worker before inference")
            episode, player, sequence, candidates = _identity_and_candidates(native_response)
            collection=getattr(self,'_diagnostic_collection',None)
            if collection is not None:
                if self._diagnostic_collection_updates:
                    raise TransportError('diagnostic collection is closed after its update')
                if self._diagnostic_collection_inferences>=collection['max_decisions']:
                    raise TransportError('diagnostic collection inference budget exhausted')
                if (type(native_response.get('game_generation')) is not str or not native_response['game_generation']
                        or type(episode) is not str or not episode):
                    raise ValueError('diagnostic collection requires an episode and generation')
                identity=(native_response['game_generation'],episode)
                if self._diagnostic_collection_episode not in (None,identity):
                    raise TransportError('diagnostic collection cannot change episode or generation')
            elif self._episode_budget is not None and is_extended(self._episode_budget):
                if self._episode_budget_inferences>=self._episode_budget['max_decisions']:
                    raise TransportError('episode inference budget exhausted')
                if (type(native_response.get('game_generation')) is not str or not native_response['game_generation']
                        or type(episode) is not str or not episode):
                    raise ValueError('episode budget requires an episode and generation')
                identity=(native_response['game_generation'],episode)
                if self._episode_budget_episode not in (None,identity):
                    raise TransportError('episode budget cannot change episode or generation')
            payload={"op": "infer", "response": native_response}
            if scenario is not None:
                from .task_context import scenario_digest
                digest=scenario_digest(scenario)
                if type(completed_decisions) is not int or completed_decisions<0:
                    raise ValueError('task inference requires a completed decision count')
                payload.update(scenario=scenario,completed_decisions=completed_decisions)
            elif completed_decisions is not None:
                raise ValueError('decision count requires an explicit scenario')
            result = self._exchange(payload)
            if scenario is not None and (result.get('scenario_sha256')!=digest
                    or result.get('task_completed_decisions')!=completed_decisions):
                self._fail_closed();raise TransportError('Worker task binding differs')
            if self.policy_version is not None and result.get('policy_version')!=self.policy_version:
                self._fail_closed();raise TransportError('Worker inference policy version changed')
            if getattr(self,'critic_version',None)!=result.get('critic_version'):
                self._fail_closed();raise TransportError('Worker behavior critic version changed')
            if self.critic_mode!='player_only' and result.get('critic_mode')!=self.critic_mode:
                self._fail_closed();raise TransportError('Worker behavior critic information mode changed')
            if (result.get("episode_id") != episode or result.get("player_id") != player
                    or result.get("sequence") != sequence):
                self._fail_closed()
                raise TransportError("worker response identity mismatch")
            candidate_id = result.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id not in candidates:
                self._fail_closed()
                raise TransportError("worker selected a candidate outside the published menu")
            action = candidates[candidate_id]
            has_assignments = "assignments" in result
            assignments = result.get("assignments")
            if has_assignments:
                if action.get("kind") != "policies" or type(assignments) is not list:
                    self._fail_closed()
                    raise TransportError("worker returned a joint assignment for a non-policy action")
                try:
                    policy_catalog = native_response.get("decisions", {}).get("policies")
                    observation = native_response.get("observation")
                    if (type(policy_catalog) is not dict or action.get("slots") != policy_catalog.get("slots")
                            or type(observation) is not dict
                            or policy_catalog.get("player_id") != player
                            or policy_catalog.get("turn") != observation.get("turn")):
                        raise ValueError("candidate policy slots differ from the response catalogue")
                    from .policy_assignments import validate_joint_assignment
                    validate_joint_assignment(action.get("slots"), assignments)
                except (TypeError, ValueError, RuntimeError) as exc:
                    self._fail_closed()
                    raise TransportError("worker returned an invalid or nonmember policy assignment") from exc
            elif action.get("kind") == "policies":
                self._fail_closed()
                raise TransportError("worker omitted the policy action's joint assignment")
            if result.get("training_eligible") is not False:
                self._fail_closed()
                raise TransportError("worker response changed diagnostic training eligibility")
            for key in ("log_prob", "value", "entropy", "inference_seconds"):
                value = result.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    self._fail_closed()
                    raise TransportError("worker returned invalid inference metrics")
            if result["inference_seconds"] < 0 or result["entropy"] < 0:
                self._fail_closed()
                raise TransportError("worker returned invalid timing or entropy")
            if collection is not None:
                self._diagnostic_collection_episode=identity
                self._diagnostic_collection_inferences+=1
            elif self._episode_budget is not None and is_extended(self._episode_budget):
                self._episode_budget_episode=identity
                self._episode_budget_inferences+=1
            return result

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                if self._process.stdin:
                    self._process.stdin.close()
            except OSError:
                pass
        self._terminate_reap()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


__all__ = ["WorkerClient", "TransportError", "MAX_LINE_BYTES",
           "EXTENDED_MAX_LINE_BYTES", "EXTENDED_MAX_DECODED_BYTES", "MAX_REQUESTS"]
