"""Read-only local monitor for published Civ6 RL run artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

SCHEMA = "civ6rl.learning_run.v1"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ROWS = 10_000
MAX_REPLAYS = 2_000
_append_lock = Lock()


def append_metric(run_dir: str | os.PathLike[str], step: int,
                  wall_seconds: float, stage: str,
                  metrics: dict[str, Any]) -> None:
    """Append one sample; reject invalid and non-finite metric values."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)) or not math.isfinite(wall_seconds):
        raise ValueError("wall_seconds must be finite")
    if not isinstance(stage, str) or not isinstance(metrics, dict):
        raise ValueError("stage and metrics have invalid types")
    safe = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("metrics must contain only finite numeric values")
        safe[key] = value
    line = json.dumps({"step": step, "wall_seconds": wall_seconds, "stage": stage, "metrics": safe},
                      ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with _append_lock, (Path(run_dir) / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None, "file_too_large"
        value = json.loads(path.read_text(encoding="utf-8"))
        return (value, None) if isinstance(value, dict) else (None, "invalid_format")
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "malformed"


def _read_jsonl(path: Path, *, limit: int = MAX_ROWS) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Ignore an incomplete final writer line; report malformed complete lines."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return [], False, "file_too_large"
        raw = path.read_bytes()
    except FileNotFoundError:
        return [], False, None
    except OSError:
        return [], False, "unreadable"
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [], False, "malformed"
    lines = content.splitlines()
    partial = bool(content) and not content.endswith(("\n", "\r"))
    if partial:
        lines = lines[:-1]
    rows: list[dict[str, Any]] = []
    malformed = False
    for line in lines[:limit]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            else:
                malformed = True
        except json.JSONDecodeError:
            malformed = True
    if len(lines) > limit:
        return rows, partial, "row_limit"
    return rows, partial, ("malformed_middle" if malformed else None)


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _scenario_sha256(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return value.lower()
    return None


def _clean_metric(row: dict[str, Any]) -> dict[str, Any] | None:
    metrics = row.get("metrics")
    step = row.get("step")
    wall = row.get("wall_seconds")
    if not isinstance(metrics, dict) or isinstance(step, bool) or not isinstance(step, int) or step < 0 or not _finite(wall):
        return None
    return {"step": step, "wall_seconds": wall,
            "stage": row.get("stage") if isinstance(row.get("stage"), str) else "",
            "metrics": {k: v for k, v in metrics.items() if isinstance(k, str) and _finite(v)}}


def _critic_safe_metrics(rows: list[dict[str, Any]], manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Expose privileged-only aliases only when the manifest declares that mode."""
    if isinstance(manifest, dict) and manifest.get("critic_mode") == "privileged_diagnostic":
        return rows
    aliases = {"privileged_value_loss", "privileged_grad_norm"}
    result = []
    for row in rows:
        metrics = {key: value for key, value in row["metrics"].items() if key not in aliases}
        result.append({**row, "metrics": metrics})
    return result


def _safe_skill_instruction(context: dict[str, Any]) -> dict[str, Any] | None:
    """Accept only the public objective union, never task IDs or source data."""
    instruction = context.get('instruction')
    if (type(instruction) is not dict or set(instruction) != {'schema','objective','deadline_turn'}
            or instruction['schema'] != 'civ6rl.public_skill_instruction991.v1'
            or type(instruction['deadline_turn']) is not int
            or not 1 <= instruction['deadline_turn'] <= 1000000):
        return None
    objective = instruction['objective']
    if type(objective) is not dict or objective.get('kind') != context.get('objective'):
        return None
    kind = objective['kind']
    def integer(value, low=1, high=4096):
        return type(value) is int and low <= value <= high
    def location(value):
        return (type(value) is dict and set(value) == {'x','y'}
                and all(integer(value[k],0,16383) for k in ('x','y')))
    if kind == 'development_capacity_and_reserve':
        if (set(objective) != {'kind','unit_type','target_count','gold_reserve'}
                or objective['unit_type'] not in ('UNIT_BUILDER','UNIT_SETTLER')
                or not integer(objective['target_count'])
                or not integer(objective['gold_reserve'],0,2147483647)):
            return None
        safe = {key:objective[key] for key in ('kind','unit_type','target_count','gold_reserve')}
    elif kind == 'settlement_and_survival':
        if (set(objective) != {'kind','target_cities','origin_city'}
                or not integer(objective['target_cities']) or not location(objective['origin_city'])):
            return None
        safe = dict(kind=kind,target_cities=objective['target_cities'],origin_city=dict(objective['origin_city']))
    elif kind == 'short_tactical_defense':
        force = objective.get('force_types')
        if (set(objective) != {'kind','defended_city','force_types','minimum_force'}
                or not location(objective['defended_city']) or not integer(objective['minimum_force'])
                or type(force) is not list or not 0 < len(force) <= 100000
                or any(type(name) is not str or re.fullmatch(r'UNIT_[A-Z0-9_]{1,123}',name) is None for name in force)
                or force != sorted(set(force))):
            return None
        safe = dict(kind=kind,defended_city=dict(objective['defended_city']),
                    force_types=list(force),minimum_force=objective['minimum_force'])
    else:
        return None
    return dict(schema=instruction['schema'],objective=safe,deadline_turn=instruction['deadline_turn'])


def _safe_manifest(raw: dict[str, Any] | None, *, metric_scope: str | None = None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if raw.get("schema") != SCHEMA:
        return {"schema": raw.get("schema") if isinstance(raw.get("schema"), str) else "", "error": "unsupported_schema"}
    mode = raw.get("mode") if raw.get("mode") in ("diagnostic", "training") else "unknown"
    limitations = raw.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations]
    if not isinstance(limitations, list):
        limitations = []
    scope = metric_scope if metric_scope == "recorded_inference" else raw.get("metric_scope")
    if scope not in ("recorded_inference", "training", "diagnostic"):
        scope = ""
    error = raw.get("error") if isinstance(raw.get("error"), str) else ""
    close_error = raw.get("close_error") if isinstance(raw.get("close_error"), str) else ""
    out = {"schema": SCHEMA, "run_id": str(raw.get("run_id", ""))[:200], "mode": mode,
            "status": str(raw.get("status", "unknown"))[:80], "started_at": str(raw.get("started_at", ""))[:100],
            "training_eligible": raw.get("training_eligible") is True,
            "metric_scope": scope,
            "limitations": [str(x)[:500] for x in limitations[:40]]}
    # Task context is optional so older runs remain valid. Publish only the
    # diagnostic exercise fields agreed with the collector; never pass through
    # arbitrary scenario IDs, feature names, seeds, or other manifest keys.
    context = raw.get("task_context")
    if isinstance(context, dict) and context.get("scope") == "diagnostic":
        objective = context.get("objective")
        target = context.get("target")
        max_decisions = context.get("max_decisions")
        max_turns = context.get("max_turns")
        scenario_hash = _scenario_sha256(context.get("scenario_sha256"))
        common_valid = (type(max_decisions) is int and max_decisions > 0
                and type(max_turns) is int and max_turns > 0
                and scenario_hash is not None)
        if (common_valid and isinstance(objective, str) and objective and len(objective) <= 100
                and _finite(target) and 'instruction' not in context):
            out["task_context"] = {
                "objective": objective,
                "target": target,
                "max_decisions": max_decisions,
                "max_turns": max_turns,
                "scenario_sha256": scenario_hash,
                "scope": "diagnostic",
            }
        elif common_valid and context.get('training_eligible') is False:
            instruction = _safe_skill_instruction(context)
            if instruction is not None:
                out['task_context'] = dict(objective=objective,instruction=instruction,
                    max_decisions=max_decisions,max_turns=max_turns,scenario_sha256=scenario_hash,
                    scope='diagnostic',training_eligible=False,
                    provenance_status='declared_hash_bindings_only_not_verified_admission')
    if type(raw.get("limits_are_caps")) is bool:
        out["limits_are_caps"] = raw["limits_are_caps"]
    if error:
        out["error"] = error[:2000]
    if close_error:
        out["close_error"] = close_error[:2000]
    for key in ("requested_steps", "steps_requested", "executed_steps", "steps_executed", "native_choices",
                "requested_updates", "executed_updates", "requested_episodes", "completed_episodes",
                "step_limit", "update_limit", "episode_limit"):
        value = raw.get(key)
        if type(value) is int and value >= 0:
            out[key] = value
    for key in ("max_seconds", "elapsed_seconds", "time_limit_seconds"):
        value = raw.get(key)
        if _finite(value) and value >= 0:
            out[key] = value
    stop_reason = raw.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason:
        out["stop_reason"] = stop_reason[:240]
    heartbeat = raw.get("heartbeat_at")
    if isinstance(heartbeat, str) and len(heartbeat) <= 100:
        out["heartbeat_at"] = heartbeat
    timeout = raw.get("progress_timeout_seconds")
    if _finite(timeout) and timeout > 0:
        out["progress_timeout_seconds"] = timeout
    model_status = raw.get("model_status")
    if model_status in ("untrained", "trained", "diagnostic_updated"):
        out["model_status"] = model_status
    elif any("non entraîné" in value.lower() or "non entraine" in value.lower()
             for value in out["limitations"]):
        out["model_status"] = "untrained"
    critic_mode = raw.get("critic_mode")
    if critic_mode in ("player_only", "privileged_diagnostic", "history_only_diagnostic"):
        out["critic_mode"] = critic_mode
    supervisor = raw.get("native_supervisor")
    if isinstance(supervisor, dict):
        safe_supervisor: dict[str, Any] = {}
        for key in ("exit_code", "last_sequence"):
            value = supervisor.get(key)
            if type(value) is int:
                safe_supervisor[key] = value
        for key in ("elapsed_seconds", "timeout_seconds"):
            value = supervisor.get(key)
            if _finite(value):
                safe_supervisor[key] = value
        for key in ("reaped", "timed_out", "game_executable_launched", "normal_destroy_called"):
            value = supervisor.get(key)
            if type(value) is bool:
                safe_supervisor[key] = value
        for key in ("stop_reason", "error"):
            value = supervisor.get(key)
            if isinstance(value, str) and value:
                safe_supervisor[key] = value[:2000]
        out["native_supervisor"] = safe_supervisor
    return out


def _safe_action(action: Any) -> Any:
    allowed = {"type", "name", "action", "kind", "unit_id", "city_id", "target_x", "target_y",
               "player_id", "choice", "route", "index", "hash", "item_type", "category", "item_index",
               "item_hash", "placement_mode", "plot_id", "x", "y", "source_x", "source_y",
               "destination_x", "destination_y", "operation", "mode", "directive", "cost",
               "yield_type", "yield_index", "favored", "disfavored", "occupied", "locked", "workers", "capacity"}
    if isinstance(action, str):
        return action[:200]
    if isinstance(action, (int, float)) and not isinstance(action, bool):
        return action if not isinstance(action, float) or math.isfinite(action) else None
    if isinstance(action, dict):
        return {k: v[:200] if isinstance(v, str) else v for k, v in action.items()
                if k in allowed and isinstance(v, (str, int, float, bool, type(None)))
                and (not isinstance(v, float) or math.isfinite(v))}
    return None


def _safe_observation(obs: Any) -> dict[str, Any] | None:
    """Return only published map and entity fields; discard unknown payload keys."""
    if not isinstance(obs, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("width", "height", "turn", "player_id"):
        value = obs.get(key)
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            result[key] = value
    allow = {"tiles": {"index", "x", "y", "terrain", "feature", "terrain_type", "feature_type",
                       "owner", "visible", "revealed", "water", "impassable", "resource_type", "resource_count"},
             "plots": {"index", "x", "y", "terrain_type", "feature_type", "owner", "visible", "revealed",
                       "water", "impassable", "resource_type", "resource_count"},
             "units": {"x", "y", "id", "type", "name", "owner", "damage", "moves", "max_moves", "health", "movement"},
             "visible_enemy_units": {"x", "y", "id", "type", "name", "owner", "damage", "visible"},
             "cities": {"x", "y", "id", "name", "owner", "population"},
             "entities": {"x", "y", "id", "type", "name", "owner", "health", "population"}}
    def safe_tile(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict) or not _finite(item.get("x")) or not _finite(item.get("y")):
            return None
        # Never materialize unrevealed cells, even if a malformed publisher
        # includes coordinates or fields for them.
        if item.get("revealed") is False:
            return None
        fields = allow["tiles"]
        if item.get("visible") is False and item.get("revealed") is not True:
            fields = {"index", "x", "y", "visible", "revealed"}
        return {k: (v[:200] if isinstance(v, str) else v) for k, v in item.items()
                if k in fields and isinstance(v, (str, int, float, bool))
                and (not isinstance(v, float) or math.isfinite(v))}
    # Some publishers group the same public fields under observation.map.
    map_value = obs.get("map")
    if isinstance(map_value, dict):
        safe_map: dict[str, Any] = {}
        for key in ("width", "height"):
            value = map_value.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                safe_map[key] = value
        map_tiles = map_value.get("tiles")
        if isinstance(map_tiles, list):
            safe_map["tiles"] = [clean for item in map_tiles[:5000]
                                  if (clean := safe_tile(item)) is not None]
        result["map"] = safe_map
    legacy_tiles = obs.get("tiles")
    if isinstance(legacy_tiles, list):
        result["tiles"] = [clean for item in legacy_tiles[:5000]
                            if (clean := safe_tile(item)) is not None]
    for collection, fields in allow.items():
        if collection == "tiles":
            continue
        values = obs.get(collection)
        if not isinstance(values, list):
            continue
        rows = []
        for item in values[:5000]:
            if not isinstance(item, dict):
                continue
            if collection == "plots":
                clean_tile = safe_tile(item)
                if clean_tile is not None:
                    rows.append(clean_tile)
                continue
            if collection == "visible_enemy_units" and item.get("visible", True) is not True:
                continue
            clean = {k: (v[:200] if isinstance(v, str) else v) for k, v in item.items()
                     if k in fields and isinstance(v, (str, int, float, bool))
                     and (not isinstance(v, float) or math.isfinite(v))}
            if "x" in clean and "y" in clean:
                rows.append(clean)
        result[collection] = rows
    return result


def _diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize published decision metrics, not game results or PPO updates."""
    counts: dict[int, int] = {}
    handoffs = boundaries = 0
    for row in rows:
        if row["stage"] != "transition":
            continue
        metrics = row["metrics"]
        pid = metrics.get("player_id")
        if type(pid) is not int or not 0 <= pid <= 63:
            continue
        counts[pid] = counts.get(pid, 0) + 1
        successor = metrics.get("successor_player_id")
        handoff = metrics.get("handoff")
        if (type(successor) is int and 0 <= successor <= 63
                and type(handoff) is int and handoff == int(pid != successor)):
            boundaries += 1
            handoffs += handoff
    return {"scope": "published_transition_metrics", "decisions": sum(counts.values()),
            "players": sorted(counts), "counts": {str(pid): counts[pid] for pid in sorted(counts)},
            "handoffs": handoffs, "handoff_boundaries": boundaries}


class MonitorService:
    def __init__(self, runs_dir: str | os.PathLike[str]):
        self.root = Path(runs_dir).resolve()

    def _folder(self, run_id: str) -> Path | None:
        if not isinstance(run_id, str) or not run_id or run_id in (".", "..") or any(c in run_id for c in ("/", "\\", "\x00")):
            return None
        try:
            candidate = self.root / run_id
            folder = candidate.resolve()
            return folder if (not candidate.is_symlink() and folder == candidate
                              and folder.parent == self.root and folder.is_dir()) else None
        except (OSError, RuntimeError):
            return None

    @staticmethod
    def _replay_file(folder: Path, player: int | None) -> tuple[Path | None, str | None]:
        name = "replays.jsonl" if player is None else f"player-{player}-replays.jsonl"
        candidate = folder / name
        try:
            if candidate.is_symlink() or candidate.resolve() != candidate:
                return None, "unsafe_replay_path"
            if not candidate.exists():
                return None, None
            if not candidate.is_file():
                return None, "unsafe_replay_path"
            return candidate, None
        except (OSError, RuntimeError):
            return None, "unreadable_replay_path"

    def _replay_sources(self, folder: Path) -> dict[str, Any]:
        # Exactly65 canonical names, no unbounded directory traversal/glob and
        # no reading another player's replay merely to discover its existence.
        players, errors = [], []
        for pid in range(64):
            path, error = self._replay_file(folder, pid)
            if path is not None:
                players.append(pid)
            if error:
                errors.append(f"player_{pid}:{error}")
        legacy, error = self._replay_file(folder, None)
        if error:
            errors.append("legacy:" + error)
        return {"players": players, "legacy": legacy is not None, "errors": errors}

    def runs(self) -> dict[str, Any]:
        result = []
        try:
            children = sorted(self.root.iterdir(), key=lambda p: p.name)[:500]
        except OSError:
            children = []
        for folder in children:
            if self._folder(folder.name) is None:
                continue
            manifest, m_err = _read_json(folder / "manifest.json")
            rows, _, d_err = _read_jsonl(folder / "metrics.jsonl")
            clean = [item for item in (_clean_metric(row) for row in rows) if item is not None]
            safe_manifest = _safe_manifest(manifest)
            clean = _critic_safe_metrics(clean, safe_manifest)
            scope = "recorded_inference" if any(row["stage"] == "recorded_native_inference" for row in clean) else None
            result.append({"id": folder.name, "manifest": _safe_manifest(manifest, metric_scope=scope),
                           "latest_metrics": clean[-1] if clean else None, "sample_count": len(clean),
                           "errors": [x for x in ((m_err if m_err != "missing" else None), d_err) if x]})
        return {"runs": result, "no_data": not bool(result)}

    def run(self, run_id: str) -> tuple[int, dict[str, Any]]:
        folder = self._folder(run_id)
        if folder is None:
            return 404, {"error": "run_not_found"}
        manifest, m_err = _read_json(folder / "manifest.json")
        rows, partial, d_err = _read_jsonl(folder / "metrics.jsonl")
        clean = [item for item in (_clean_metric(row) for row in rows) if item is not None]
        safe_manifest = _safe_manifest(manifest)
        clean = _critic_safe_metrics(clean, safe_manifest)
        scope = "recorded_inference" if any(row["stage"] == "recorded_native_inference" for row in clean) else None
        errors = [x for x in ((m_err if m_err != "missing" else None), d_err) if x]
        try:
            mtime = datetime.fromtimestamp((folder / "metrics.jsonl").stat().st_mtime, timezone.utc).isoformat()
        except OSError:
            mtime = None
        return 200, {"id": folder.name, "manifest": _safe_manifest(manifest, metric_scope=scope), "metrics": clean,
                     "tail_incomplete": partial, "last_metrics_write_utc": mtime,
                     "errors": errors, "no_data": not bool(clean),
                     "replay_sources": self._replay_sources(folder),
                     "diagnostic_summary": _diagnostic_summary(clean)}

    def replay(self, run_id: str, index: int, player: int | None = None) -> tuple[int, dict[str, Any]]:
        """Select one perspective; omitted player explicitly means legacy file.

        A requested player never falls back to legacy or another player. Every
        complete row in that file must confirm the selected observation owner,
        even if the requested index would otherwise point to a valid row.
        """
        if type(index) is not int or index < 0:
            return 400, {"error": "invalid_index"}
        if player is not None and (type(player) is not int or not 0 <= player <= 63):
            return 400, {"error": "invalid_player"}
        folder = self._folder(run_id)
        if folder is None:
            return 404, {"error": "run_not_found"}
        path, path_error = self._replay_file(folder, player)
        if path_error:
            return 403, {"error": path_error}
        if path is None:
            return 404, {"error": "replay_not_found", "count": 0, "tail_incomplete": False, "data_error": None}
        rows, partial, error = _read_jsonl(path, limit=MAX_REPLAYS)
        if error in ("file_too_large", "row_limit"):
            return 413, {"error": error}
        if player is not None:
            if error:
                return 422, {"error": "invalid_player_replay", "data_error": error}
            for row in rows:
                obs = row.get("observation")
                if (type(obs) is not dict or obs.get("schema") != "civ6rl-player-observation-v1"
                        or type(obs.get("player_id")) is not int or obs["player_id"] != player
                        or type(obs.get("turn")) is not int or obs["turn"] < 0):
                    return 422, {"error": "replay_perspective_mismatch"}
                for key in ("player_id", "turn"):
                    if key in row and (type(row[key]) is not int or row[key] != obs[key]):
                        return 422, {"error": "replay_metadata_mismatch"}
        if index < 0 or index >= len(rows):
            return 404, {"error": "replay_not_found", "count": len(rows), "tail_incomplete": partial, "data_error": error}
        row = rows[index]
        replay = {k: row[k] for k in ("step", "player_id", "turn")
                  if isinstance(row.get(k), (int, float, str)) and not isinstance(row.get(k), bool)}
        if player is not None:
            replay.update(player_id=player, turn=row["observation"]["turn"])
        scenario_hash = _scenario_sha256(row.get("scenario_sha256"))
        if scenario_hash is not None:
            replay["scenario_sha256"] = scenario_hash
        replay["observation"] = _safe_observation(row.get("observation"))
        replay["action"] = _safe_action(row.get("action"))
        return 200, {"replay": replay, "index": index, "count": len(rows), "tail_incomplete": partial,
                     "data_error": error, "source": "legacy" if player is None else "player", "player_id": player}


def make_handler(service: MonitorService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Civ6RLMonitor/1"
        def log_message(self, _format: str, *args: Any) -> None:
            return
        def _json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload))); self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(payload)
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            if parsed.path == "/api/project":
                # Fixed public delivery plan, separate from learner/private artifacts.
                path = Path(__file__).resolve().parents[2] / "config" / "phase1-delivery.json"
                data, error = _read_json(path)
                if error:
                    self._json(404 if error == "missing" else 503, {"error": error}); return
                if data.get("schema") != "civ6rl.phase1_delivery.v1":
                    self._json(503, {"error": "unknown_project_plan"}); return
                self._json(200, data); return
            if parsed.path == "/dashboard.html":
                try:
                    data = Path(__file__).with_name("dashboard.html").read_bytes()
                except OSError:
                    self._json(404, {"error": "dashboard_unavailable"}); return
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
                self.end_headers(); self.wfile.write(data); return
            if parsed.path in ("/", "/api/runs"):
                if parsed.path == "/":
                    self.send_response(302); self.send_header("Location", "/dashboard.html"); self.end_headers(); return
                self._json(200, service.runs()); return
            if parsed.path == "/api/run":
                status, data = service.run(query.get("id", [""])[0]); self._json(status, data); return
            if parsed.path == "/api/replay":
                if len(query.get("id", [])) != 1 or len(query.get("index", ["0"])) != 1:
                    self._json(400, {"error": "ambiguous_replay_query"}); return
                player = None
                if "player" in query:
                    values = query["player"]
                    if (len(values) != 1 or re.fullmatch(r"(?:0|[1-9][0-9]?)", values[0]) is None
                            or int(values[0]) > 63):
                        self._json(400, {"error": "invalid_player"}); return
                    player = int(values[0])
                try:
                    index = int(query.get("index", ["0"])[0])
                except ValueError:
                    self._json(400, {"error": "invalid_index"}); return
                status, data = service.replay(query.get("id", [""])[0], index, player); self._json(status, data); return
            self._json(404, {"error": "not_found"})
    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the local read-only RL run monitor")
    parser.add_argument("--runs", required=True, help="Folder containing run folders")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(MonitorService(args.runs)))
    print(f"Monitor local : http://127.0.0.1:{args.port}/dashboard.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
