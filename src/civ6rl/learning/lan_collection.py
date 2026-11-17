"""Collect public LAN diagnostic intervals through the shared host client."""
from copy import deepcopy
from math import isfinite
from time import perf_counter

from . import lan_inputs, skill_task991
from .action_receipt import selected_action_and_assignments
from .lan_rollout import _ready, _require, early_task_terminal, official_result, task_step, transition
from .skill_context993 import evaluate_skill_transition, validate_skill_scenario

MAX_POLLS = 12
MAX_DIAGNOSTIC_DECISIONS = 12
MAX_EVALUATION_DECISIONS = 64


def collect_episode(client, worker, scenario, *, max_decisions=MAX_DIAGNOSTIC_DECISIONS,
                    max_collection_seconds=None, diagnostic_collection=None):
    """Collect a bounded public episode from an already reset LAN client.

    The caller owns reset, process supervision and close. The worker receives
    only the public decision view and announced task. A refused command or
    incomplete poll interval raises before any episode is returned.
    The optional wall-time limit is checked after each complete interval.
    The caller still enforces a hard limit on the native session. Collections
    above the legacy learner limit require an explicitly negotiated collection
    budget for learning. Without that opt-in the historical evaluation cap
    remains unchanged. Collection boundaries are truncations, not task failure.
    An official result without a scoreable skill transition returns no PPO
    records. Close the session and restart the worker from its clean learner
    boundary before collecting again.
    """
    budget = None
    if diagnostic_collection is not None:
        from .diagnostic_budget998 import validate_collection
        budget = validate_collection(diagnostic_collection)
        _require(getattr(worker, "diagnostic_collection", None) == budget,
                 "LAN collection budget must match the worker handshake")
    decision_cap = MAX_EVALUATION_DECISIONS if budget is None else budget["max_decisions"]
    _require(type(max_decisions) is int and 1 <= max_decisions <= decision_cap,
             "LAN collection decision bound exceeded")
    seconds_cap = 900
    if budget is not None:
        remaining = getattr(worker, "collection_seconds_remaining", None)
        _require(type(remaining) in (int, float) and isfinite(remaining) and remaining > 0,
                 "LAN worker has no remaining collection time after update and cleanup reserve")
        seconds_cap = min(seconds_cap, remaining,
                          budget.get("wall_seconds", 120) - budget.get("request_seconds", 20) - 30)
    _require(max_collection_seconds is None or (type(max_collection_seconds) in (int, float)
             and 0 < max_collection_seconds <= seconds_cap and isfinite(max_collection_seconds)),
             "LAN collection time bound exceeds its declared limit")
    if budget is not None and max_collection_seconds is None:
        max_collection_seconds = seconds_cap
    validate_skill_scenario(scenario)
    view = client.view()
    _require(not lan_inputs.validate(view) and _ready(view), "LAN collection needs a ready initial view")
    player = view["observation"]["player_id"]
    _require(client.session_state["human_player_ids"] == [player],
             "LAN diagnostic collector requires one captured human seat")
    skill_task991.validate_start(scenario["skill_task"], view["observation"])
    _require(view["sequence"] == 0, "LAN collection requires a fresh native reset")
    records = []
    elapsed = dict(inference_seconds=0.0, action_seconds=0.0, poll_seconds=0.0)
    poll_count = 0
    native_result = None
    collection_started = perf_counter()
    for index in range(max_decisions):
        before = view
        started = perf_counter()
        choice = worker.infer(deepcopy(before), scenario=deepcopy(scenario), completed_decisions=index)
        elapsed["inference_seconds"] += perf_counter() - started
        _require(type(choice) is dict and all(type(choice.get(key)) is type(value) and choice[key] == value
                 for key, value in (("episode_id", before["episode_id"]), ("player_id", player),
                                    ("sequence", before["sequence"])))
                 and choice.get("game_generation", before["game_generation"]) == before["game_generation"],
                 "LAN worker choice has a stale or foreign decision identity")
        action, assignments = selected_action_and_assignments(before, choice)
        started = perf_counter()
        parameters = None if assignments is None else dict(assignments=assignments)
        view = client.act(player, before["decisions"]["decision_id"], choice["candidate_id"],
                          parameters=parameters)
        elapsed["action_seconds"] += perf_counter() - started
        receipt = client.last_command_receipt(player)
        record = dict(before=before, choice=choice, command_receipt=receipt, successors=[view])
        command_status = receipt["command_result"]["status"]
        completed = command_status == "completed" and view["applied_action"] == action
        if not view["outcome"]["terminated"]:
            _require(completed, "LAN command did not complete the sampled action")
            for poll in range(MAX_POLLS + 1):
                if view["outcome"]["terminated"] or view["outcome"]["truncated"] or _ready(view):
                    break
                early = skill_task991.evaluate_step(scenario["skill_task"], before["observation"],
                                                    view["observation"])
                if early_task_terminal(scenario, early):
                    break
                _require(poll < MAX_POLLS, "LAN poll budget exhausted before a usable successor")
                started = perf_counter()
                view = client.poll(player)
                elapsed["poll_seconds"] += perf_counter() - started
                poll_count += 1
                record["successors"].append(view)
        if view["outcome"]["terminated"]:
            successors = record["successors"]
            last_board = (successors[-2] if len(successors) > 1 else before)["observation"]
            native_result = official_result(before, view, sequence=before["sequence"] + len(successors),
                                            minimum_turn=last_board["turn"])
            _require(completed or (command_status in ("refused", "submitted", "pending")
                                  and view["applied_action"] is None),
                     "Official terminal contradicts the sampled action receipt")
            if not completed or view["observation"] is None:
                unknown = evaluate_skill_transition(
                    scenario, before["observation"], None, decision_count=index + 1,
                    host_failure="official terminal without a scoreable skill transition",
                )
                result = dict(unknown, reason="official_terminal_without_scoreable_skill_transition",
                              native_game_ended=True,
                              official_game_outcome="victory" if native_result["status"] == "won" else "defeat")
                return dict(records=[], task_result=result, decisions=index + 1, polls=poll_count,
                            timings=elapsed, official_result=native_result, training_ready=False)
        transition(record)
        result = task_step(scenario, before, record["successors"], decision_count=index + 1)
        records.append(record)
        if result["terminated"] or result["truncated"]:
            break
        time_exhausted = (max_collection_seconds is not None
                          and perf_counter() - collection_started >= max_collection_seconds)
        rounds_exhausted = (budget is not None and "max_rounds" in budget
                            and view["observation"]["turn"] - scenario["skill_task"]["start_turn"]
                            >= budget["max_rounds"])
        if view["outcome"]["truncated"] or index + 1 == max_decisions or time_exhausted or rounds_exhausted:
            result = dict(result, status="truncated", truncated=True,
                          reason=("explicit_collection_time_boundary" if time_exhausted
                                  else "explicit_collection_round_boundary" if rounds_exhausted
                                  else "explicit_global_collection_boundary"))
            break
    summary = dict(records=records, task_result=result, decisions=len(records), polls=poll_count,
                   timings=elapsed, training_ready=False)
    if budget is not None:
        summary["collection"] = budget
    if native_result is not None:
        summary["official_result"] = native_result
    return summary
