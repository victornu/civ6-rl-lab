"""Publish a retained public LAN diagnostic in the standard monitor format."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .lan_rollout import transition
from .monitor import MAX_REPLAYS, SCHEMA, _safe_action, _safe_observation, append_metric

_PPO_METRICS = (
    "loss", "policy_loss", "value_loss", "privileged_value_loss",
    "privileged_grad_norm", "entropy", "kl", "clip_fraction", "grad_norm",
    "epochs_completed", "early_stop_kl",
)


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected finite nonnegative diagnostic duration")
    return value


def _decisions(exchange):
    rows = iter(exchange)
    reset = next(rows, {})
    if reset.get("stage") != "reset" or reset["view"]["sequence"] != 0:
        raise ValueError("The public exchange must begin at reset")
    before = reset["view"]
    elapsed = 0.0
    row = next(rows, {})
    while row.get("stage") == "actor":
        choice = row["choice"]
        obs = before["observation"]
        if (choice["episode_id"], choice["player_id"], choice["sequence"]) != (
            before["episode_id"], obs["player_id"], before["sequence"]
        ) or choice["training_eligible"] is not False:
            raise ValueError("Actor choice is not from this diagnostic decision")
        act = next(rows, {})
        if act.get("stage") != "act":
            raise ValueError("Actor choice has no native response")
        successors = [act["view"]]
        elapsed += _seconds(choice["inference_seconds"]) + _seconds(act["seconds"])
        row = next(rows, {})
        while row.get("stage") == "poll":
            successors.append(row["view"])
            elapsed += _seconds(row["seconds"])
            row = next(rows, {})
        successor = transition(dict(before=before, choice=choice,
                                    command_receipt=act["receipt"], successors=successors))
        yield obs, act["view"]["applied_action"], choice, elapsed
        before = successor
    if row.get("stage") != "close" or next(rows, None) is not None:
        raise ValueError("Public diagnostic exchange must end at close")
    close = row["view"]
    if (close["episode_id"], close["game_generation"], close["sequence"], close["outcome"]["reason"]) != (
        before["episode_id"], before["game_generation"], before["sequence"] + 1, "requested_close"
    ):
        raise ValueError("Diagnostic close is stale or foreign")


def publish(runs, run_id, exchange, result, *, update=None):
    """Write only public replay projections and allowlisted numeric diagnostics."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", run_id):
        raise ValueError("Invalid diagnostic run ID")
    if result["training_ready"] is not False or result["clean_native_close"] is not True:
        raise ValueError("Expected a completed native diagnostic capture")
    decisions = list(_decisions(exchange))
    summary = result["summary"]
    count = len(decisions)
    task = summary["task_result"]
    if (not 1 <= count <= MAX_REPLAYS or count != summary["decisions"]
            or task["training_eligible"] is not False):
        raise ValueError("Diagnostic result differs from the retained decisions")
    player = decisions[0][0]["player_id"]
    if task["player_id"] != player:
        raise ValueError("Diagnostic task result has a different player")
    collection_seconds = _seconds(result["collection_seconds"])
    if collection_seconds == 0:
        raise ValueError("Diagnostic collection has no measured duration")
    manifest = dict(schema=SCHEMA, run_id=run_id, mode="diagnostic", metric_scope="diagnostic",
                    status="completed", training_eligible=False, executed_steps=count,
                    model_status=decisions[0][2]["model_status"], stop_reason=task["reason"],
                    limitations=["Une trajectoire BBG instrumentée, sans évaluation sur un monde indépendant."])
    if result.get("seed_selected_for_two_initial_passes") is True:
        manifest["limitations"].append("Graine choisie pour observer deux passages forcés.")
    if update is not None:
        if (update["training_eligible"] is not False or update["critic_mode"] != "privileged_diagnostic"
                or update["optimized_decisions"] != count or update["completed_updates"] != 1
                or update["scenario_sha256"] != task["scenario_sha256"]
                or update["task_steps"][-1] != task
                or any(choice["policy_version"] != update["previous_policy_version"]
                       for _, _, choice, _ in decisions)):
            raise ValueError("Expected one matching asymmetric diagnostic update")
        manifest.update(executed_updates=1, critic_mode=update["critic_mode"],
                        model_status=update["model_status"])
        manifest["limitations"].append("Une mise à jour PPO hors ligne sur les décisions natives rejouées.")
    runs = Path(runs).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / run_id
    with TemporaryDirectory(dir=runs.parent) as tmp:
        staging = Path(tmp) / run_id
        staging.mkdir()
        with (staging / f"player-{player}-replays.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for step, (obs, action, choice, elapsed) in enumerate(decisions, 1):
                public = dict(_safe_observation(obs), schema=obs["schema"])
                frame = dict(step=step, observation=public, action=_safe_action(action),
                             scenario_sha256=task["scenario_sha256"])
                stream.write(json.dumps(frame, allow_nan=False, separators=(",", ":")) + "\n")
                append_metric(staging, step, elapsed, "transition",
                              dict(player_id=player, entropy=choice["entropy"]))
        timing = summary["timings"]
        append_metric(staging, count, collection_seconds, "native_diagnostic_summary", dict(
            steps_per_second=count / collection_seconds, reward=task["reward"],
            task_success=int(task["task_success"]), polls=summary["polls"],
            reset_seconds=_seconds(result["reset_seconds"]),
            inference_seconds=_seconds(timing["inference_seconds"]),
            engine_seconds=_seconds(timing["action_seconds"]) + _seconds(timing["poll_seconds"]),
        ))
        if update is not None:
            metrics = {key: update["metrics"][key] for key in _PPO_METRICS}
            metrics.update(update_seconds=_seconds(update["update_seconds"]),
                           steps_per_second=count / collection_seconds)
            append_metric(staging, 1, collection_seconds + metrics["update_seconds"], "ppo_update", metrics)
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    return destination
