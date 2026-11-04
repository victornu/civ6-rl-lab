"""Bounded, evidence-driven run control for diagnostic and learning runs.

This module records only progress and metrics explicitly reported by its
caller. It never derives rewards, episode outcomes, or training eligibility.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Callable
import uuid

from .monitor import append_metric
from .task_context import monitor_task_context


SCHEMA = "civ6rl.learning_run.v1"
TERMINAL_STATES = {"completed", "stopped", "failed"}


class RunControlError(RuntimeError):
    """A run was stopped or rejected by an explicit control rule."""


def _utc_iso(now: datetime) -> str:
    if not isinstance(now, datetime):
        raise TypeError("utc clock must return datetime")
    if now.tzinfo is None:
        raise ValueError("utc clock must return timezone-aware datetime")
    return now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_limit(value, name: str, *, positive: bool = True):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _count_limit(value, name: str):
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _numeric_metrics(metrics: dict) -> dict[str, float | int]:
    if type(metrics) is not dict or not metrics:
        raise ValueError("caller metrics must be a nonempty mapping")
    result = {}
    for key, value in metrics.items():
        finite = False
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
        if not isinstance(key, str) or not key or len(key) > 100 or not finite:
            raise ValueError("caller metrics must contain finite numeric values")
        result[key] = value
    return result


class RunController:
    """Write a compatible manifest and bound caller-reported run progress.

    ``record_transition`` counts one real transition; ``record_update`` counts
    one completed optimization update with metrics from the optimizer; and
    ``record_episode_completed`` counts only a caller-confirmed terminal
    episode. Evaluation rows are tagged by split and never increment training
    update or episode counters. A stable or zero reward does not trigger a
    stop; only absence of transitions/updates does.
    """

    def __init__(self, run_dir: str | os.PathLike[str], *, run_id: str,
                 mode: str = "diagnostic", max_seconds: float = 3600.0,
                 max_steps: int | None = None, max_updates: int | None = None,
                 max_episodes: int | None = None,
                 progress_timeout_seconds: float | None = 300.0,
                 model_status: str = "untrained", metric_scope: str | None = None,
                 scenario: dict | None = None, critic_mode: str = "player_only",
                 limitations: tuple[str, ...] | list[str] = (),
                 monotonic: Callable[[], float] = time.monotonic,
                 utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        if not isinstance(run_id, str) or not run_id or len(run_id) > 200:
            raise ValueError("run_id must be a nonempty bounded string")
        if mode not in ("diagnostic", "training"):
            raise ValueError("mode must be diagnostic or training")
        if model_status not in ("untrained", "diagnostic_updated", "trained", "unknown"):
            raise ValueError("invalid model_status")
        if critic_mode not in ("player_only", "privileged_diagnostic", "history_only_diagnostic"):
            raise ValueError("critic_mode must be player_only, privileged_diagnostic, or history_only_diagnostic")
        if scenario is not None and mode != "diagnostic":
            raise ValueError("scenario task context is diagnostic only")
        # monitor_task_context validates the whole scenario contract and
        # returns a new flat, public-only value object. It does not retain the
        # caller's mutable scenario mappings or expose their private IDs.
        task_context = monitor_task_context(scenario) if scenario is not None else None
        if metric_scope is None:
            metric_scope = "diagnostic" if mode == "diagnostic" else "training"
        if metric_scope not in ("training", "diagnostic"):
            raise ValueError("metric_scope must be training or diagnostic")
        if type(limitations) not in (list, tuple) or any(not isinstance(x, str) for x in limitations):
            raise ValueError("limitations must be a sequence of strings")
        self.max_seconds = _finite_limit(max_seconds, "max_seconds")
        self.max_steps = _count_limit(max_steps, "max_steps")
        self.max_updates = _count_limit(max_updates, "max_updates")
        self.max_episodes = _count_limit(max_episodes, "max_episodes")
        self.progress_timeout_seconds = _finite_limit(progress_timeout_seconds,
                                                      "progress_timeout_seconds")
        if self.progress_timeout_seconds is not None and self.progress_timeout_seconds > self.max_seconds:
            raise ValueError("progress timeout cannot exceed the run time bound")
        if all(value is None for value in (self.max_seconds, self.max_steps, self.max_updates, self.max_episodes)):
            raise ValueError("at least one finite run bound is required")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        if any((self.run_dir / name).exists() for name in ("manifest.json", "metrics.jsonl", "replays.jsonl")):
            raise FileExistsError("run directory already contains run artifacts")
        self.run_id = run_id
        self.mode = mode
        self.model_status = model_status
        self.metric_scope = metric_scope
        self._task_context = dict(task_context) if task_context is not None else None
        self.critic_mode = critic_mode
        self._clock = monotonic
        self._utc_now = utc_now
        self._started = self._clock()
        self._last_progress = self._started
        if not math.isfinite(self._started):
            raise ValueError("monotonic clock returned a non-finite value")
        self.steps = 0
        self.updates = 0
        self.episodes = 0
        self.status = "running"
        self.stop_reason = ""
        self.error = ""
        self.limitations = list(limitations)[:40]
        self.limitations.append("RunController ne qualifie jamais l’éligibilité à l’entraînement.")
        self.started_at = _utc_iso(self._utc_now())
        self.heartbeat_at = self.started_at
        self._write_manifest()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    def _manifest(self) -> dict:
        return {
            "schema": SCHEMA, "run_id": self.run_id, "mode": self.mode,
            "status": self.status, "started_at": self.started_at,
            "heartbeat_at": self.heartbeat_at,
            "progress_timeout_seconds": self.progress_timeout_seconds,
            "metric_scope": self.metric_scope,
            "model_status": self.model_status,
            "critic_mode": self.critic_mode,
            "training_eligible": False,
            "limits_are_caps": True,
            "step_limit": self.max_steps,
            "update_limit": self.max_updates,
            "episode_limit": self.max_episodes,
            "time_limit_seconds": self.max_seconds,
            "requested_steps": self.max_steps,
            "executed_steps": self.steps,
            "requested_updates": self.max_updates,
            "executed_updates": self.updates,
            "requested_episodes": self.max_episodes,
            "completed_episodes": self.episodes,
            "max_seconds": self.max_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "limitations": self.limitations,
            "stop_reason": self.stop_reason,
            "error": self.error,
            **({"task_context": dict(self._task_context)} if self._task_context is not None else {}),
        }

    def _write_manifest(self) -> None:
        payload = json.dumps(self._manifest(), ensure_ascii=False, allow_nan=False,
                             sort_keys=True, indent=2) + "\n"
        temporary = self.run_dir / f".manifest-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _ensure_running(self) -> None:
        if self.status != "running":
            raise RunControlError(f"run is {self.status}: {self.stop_reason or self.error}")
        self.check()
        if self.status != "running":
            raise RunControlError(f"run stopped: {self.stop_reason or self.error}")

    def _refresh_heartbeat(self) -> None:
        self.heartbeat_at = _utc_iso(self._utc_now())

    def _transition(self, status: str, *, reason: str = "", error: str = "") -> None:
        if self.status in TERMINAL_STATES:
            return
        if status not in TERMINAL_STATES:
            raise ValueError("invalid terminal run state")
        self.status = status
        self.stop_reason = reason[:240]
        self.error = error[:2000]
        self._refresh_heartbeat()
        self._write_manifest()

    def _reject(self, reason: str, message: str) -> None:
        self._transition("failed", reason=reason, error=message)
        raise RunControlError(message)

    def _check_caps(self) -> None:
        if ((self.max_steps is not None and self.steps >= self.max_steps)
                or (self.max_updates is not None and self.updates >= self.max_updates)
                or (self.max_episodes is not None and self.episodes >= self.max_episodes)):
            reason = ("max_steps" if self.max_steps is not None and self.steps >= self.max_steps else
                      "max_updates" if self.max_updates is not None and self.updates >= self.max_updates else
                      "max_episodes")
            self._transition("completed", reason=f"limit_reached:{reason}")

    def check(self) -> str:
        """Apply elapsed-time and no-transition/update limits; return current status."""
        if self.status != "running":
            return self.status
        elapsed = self.elapsed_seconds
        if self.max_seconds is not None and elapsed >= self.max_seconds:
            self._transition("stopped", reason="time_limit")
        elif (self.progress_timeout_seconds is not None
              and self._clock() - self._last_progress >= self.progress_timeout_seconds):
            self._transition("stopped", reason="no_progress_timeout")
        return self.status

    def heartbeat(self) -> str:
        """Record caller liveness without pretending it is learning progress."""
        if self.status == "running":
            self.check()
        if self.status == "running":
            self._refresh_heartbeat()
            self._write_manifest()
        return self.status

    def record_transition(self, step: int, *, metrics: dict | None = None) -> int:
        self._ensure_running()
        if type(step) is not int or step != self.steps + 1:
            self._reject("stale_progress", "transition step is stale or skipped")
        safe_metrics = None
        if metrics is not None:
            try:
                safe_metrics = _numeric_metrics(metrics)
            except ValueError as exc:
                self._reject("nonfinite_metric", str(exc))
        self.steps = step
        self._last_progress = self._clock()
        self._refresh_heartbeat()
        if safe_metrics:
            try:
                append_metric(self.run_dir, self.steps, self.elapsed_seconds, "transition", safe_metrics)
            except (OSError, ValueError) as exc:
                self._reject("metrics_write_failed", str(exc))
        self._check_caps()
        if self.status == "running":
            self._write_manifest()
        return self.steps

    def record_update(self, update_index: int, metrics: dict) -> int:
        """Record one completed PPO/optimizer update, with caller-supplied metrics."""
        self._ensure_running()
        if type(update_index) is not int or update_index != self.updates + 1:
            self._reject("stale_progress", "update index is stale or skipped")
        try:
            safe_metrics = _numeric_metrics(metrics)
        except ValueError as exc:
            self._reject("nonfinite_metric", str(exc))
        try:
            append_metric(self.run_dir, update_index, self.elapsed_seconds, "ppo_update", safe_metrics)
        except (OSError, ValueError) as exc:
            self._reject("metrics_write_failed", str(exc))
        self.updates = update_index
        self._last_progress = self._clock()
        self._refresh_heartbeat()
        self._check_caps()
        if self.status == "running":
            self._write_manifest()
        return self.updates

    def record_episode_completed(self, episode_index: int) -> int:
        """Count only an episode completion explicitly reported by the caller."""
        self._ensure_running()
        if type(episode_index) is not int or episode_index != self.episodes + 1:
            self._reject("stale_progress", "episode index is stale or skipped")
        self.episodes = episode_index
        self._last_progress = self._clock()
        self._refresh_heartbeat()
        self._check_caps()
        if self.status == "running":
            self._write_manifest()
        return self.episodes

    def record_evaluation(self, metrics: dict, *, split: str, step: int | None = None) -> None:
        """Append an evaluation row without counting it as training progress."""
        self._ensure_running()
        if split not in ("heldout", "training", "diagnostic"):
            raise ValueError("evaluation split must be heldout, training, or diagnostic")
        try:
            safe_metrics = _numeric_metrics(metrics)
        except ValueError as exc:
            self._reject("nonfinite_metric", str(exc))
        if step is None:
            step = self.updates
        if type(step) is not int or step < 0:
            self._reject("stale_progress", "evaluation step is invalid")
        append_metric(self.run_dir, step, self.elapsed_seconds, f"evaluation_{split}", safe_metrics)

    def stop(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("explicit stop reason is required")
        self._transition("stopped", reason=reason)

    def fail(self, error: str, *, reason: str = "caller_error") -> None:
        if not isinstance(error, str) or not error:
            raise ValueError("explicit failure reason is required")
        self._transition("failed", reason=reason, error=error)

    def complete(self, reason: str = "caller_completed") -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("explicit completion reason is required")
        # A caller's cleanup/completion callback must not bypass elapsed-time
        # or no-progress bounds just because it did not call check() first.
        # Existing stopped/failed states remain terminal and are preserved.
        if self.status == "running":
            self.check()
        self._transition("completed", reason=reason)


__all__ = ["RunController", "RunControlError", "SCHEMA"]
