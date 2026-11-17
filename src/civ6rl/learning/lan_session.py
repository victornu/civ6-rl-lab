"""One public LAN collection lifecycle, using the existing client and collector.

No native backend, model, reward, worker, or optimizer is created here. Callers
provide a bounded host client and a policy worker context factory. The same entry
point works with a local or remote WorkerClient. Artifacts are private diagnostic
records, not evidence of native execution or strategic training admission.
"""
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

from .lan_collection import MAX_EVALUATION_DECISIONS, collect_episode
from .skill_context993 import skill_digest, validate_skill_scenario


def collect_session(client, worker_factory, scenario, output, *, max_decisions=12,
                    max_collection_seconds=None, diagnostic_collection=None):
    """Reset once, collect explicit choices, close, and retain the public I/O.

    ``worker_factory()`` supplies a context manager yielding the initialized
    worker. It is entered AFTER native reset, so expensive GameCore loading does
    not exhaust the policy worker's wall budget. It may wrap a persistent worker
    in nullcontext, leaving that worker's ownership with the caller.
    The caller owns policy/checkpoint identity, process limits, and any subsequent
    update. This function never starts training or certifies
    a host scope. On failure it records the error and aborts the client, without
    turning infrastructure failure into a task loss. An existing output directory
    is refused before reset. Logs stream to disk rather than duplicating the
    collector's rollout in memory. The supplied context manager owns cleanup of
    a newly created worker on success or failure.
    """
    validate_skill_scenario(scenario)
    if (type(max_decisions) is not int or max_decisions < 1
            or max_decisions > scenario['horizon']['max_decisions']):
        raise ValueError('Session decision budget exceeds the declared scenario horizon')
    if not callable(worker_factory):
        raise ValueError('Session needs an explicit policy worker context factory')
    cap = MAX_EVALUATION_DECISIONS
    time_cap = 900
    if diagnostic_collection is not None:
        from .diagnostic_budget998 import validate_collection
        diagnostic_collection = validate_collection(diagnostic_collection)
        cap = diagnostic_collection['max_decisions']
        time_cap = min(time_cap, diagnostic_collection.get('wall_seconds', 120)
                       - diagnostic_collection.get('request_seconds', 20)-30)
    if max_decisions > cap:
        raise ValueError('Session decision budget exceeds the negotiated collection limit')
    if max_collection_seconds is not None and (type(max_collection_seconds) not in (int, float)
            or not isfinite(max_collection_seconds) or not 0 < max_collection_seconds <= time_cap):
        raise ValueError('Session collection time must be finite and within the collector bound')
    scenario = deepcopy(scenario)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    client_closed = False
    counts = dict(actor_decisions=0, completed_commands=0, polls=0)
    summary = dict(schema='civ6rl.lan_collection_session.v1', status='starting',
                   scenario_sha256=skill_digest(scenario), training_ready=False,
                   optimizer_updates=0)

    def write_status(phase, view=None, **extra):
        state = dict(summary, status=phase, **counts,
                     elapsed_seconds=perf_counter()-started, **extra)
        if view is not None:
            state.update(sequence=view['sequence'],
                         turn=(view['observation'] or {}).get('turn'))
        temporary = output / 'progress.json.tmp'
        temporary.write_text(json.dumps(state, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(output / 'progress.json')

    write_status('starting')
    with (output / 'exchange.jsonl').open('x', encoding='utf-8', newline='\n') as exchange:
        def log(stage, **data):
            exchange.write(json.dumps(dict(stage=stage, **data), allow_nan=False)+'\n')
            exchange.flush()

        class PublicClient:
            @property
            def session_state(self):
                return client.session_state

            def view(self):
                return client.view()

            def act(self, player, decision, candidate, parameters=None):
                begin = perf_counter()
                view = client.act(player, decision, candidate, parameters=parameters)
                receipt = client.last_command_receipt(player)
                log('act', view=view, receipt=receipt, seconds=perf_counter()-begin)
                counts['completed_commands'] += receipt['command_result']['status'] == 'completed'
                write_status('collecting', view)
                return view

            def poll(self, player):
                begin = perf_counter()
                view = client.poll(player)
                counts['polls'] += 1
                log('poll', view=view, seconds=perf_counter()-begin)
                write_status('collecting', view)
                return view

            def last_command_receipt(self, player):
                return client.last_command_receipt(player)

        class PublicWorker:
            @property
            def diagnostic_collection(self):
                return getattr(worker, 'diagnostic_collection', None)

            @property
            def collection_seconds_remaining(self):
                return getattr(worker, 'collection_seconds_remaining', None)

            def infer(self, view, **kwargs):
                choice = worker.infer(view, **kwargs)
                counts['actor_decisions'] += 1
                log('actor', choice=choice)
                return choice

        try:
            begin = perf_counter()
            view = client.reset()
            reset_seconds = perf_counter()-begin
            log('reset', view=view, seconds=reset_seconds)
            write_status('collecting', view)
            begin = perf_counter()
            with worker_factory() as worker:
                worker_startup_seconds = perf_counter()-begin
                begin = perf_counter()
                collected = collect_episode(PublicClient(), PublicWorker(), scenario,
                    max_decisions=max_decisions, max_collection_seconds=max_collection_seconds,
                    diagnostic_collection=diagnostic_collection)
                collection_seconds = perf_counter()-begin
                worker_closing = perf_counter()
            worker_close_seconds = perf_counter()-worker_closing
            player = scenario['skill_task']['player_id']
            begin = perf_counter()
            closed = client.close(player)
            client_closed = True
            close_seconds = perf_counter()-begin
            log('close', view=closed, seconds=close_seconds)
            summary.update(status='collected', reset_seconds=reset_seconds,
                collection_seconds=collection_seconds, close_seconds=close_seconds,
                worker_startup_seconds=worker_startup_seconds,
                worker_close_seconds=worker_close_seconds,
                client_closed=True, native_lifecycle_verified=False,
                summary={k:v for k,v in collected.items() if k != 'records'}, **counts)
            # The rollout stays in the existing collector/learner format.
            with (output / 'records.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
                for record in collected['records']:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
            (output / 'result.json').write_text(json.dumps(summary, allow_nan=False)+'\n',
                                               encoding='utf-8')
            write_status('collected', closed)
            return collected
        except BaseException as error:
            # Preserve the original failure even if cleanup or disk writes fail.
            cleanup_error = None
            if not client_closed:
                try:
                    client.abort()
                except BaseException as abort_error:
                    cleanup_error = repr(abort_error)
            try:
                write_status('failed', error=repr(error), cleanup_error=cleanup_error,
                             client_closed=client_closed)
            except (OSError, ValueError):
                pass
            raise
