"""Join a public LAN decision, its native command receipt and ordered views."""
from . import lan_inputs, skill_task991
from .action_receipt import selected_action_and_assignments
from .skill_context993 import evaluate_skill_transition

RECORD_FIELDS = {"before", "choice", "command_receipt", "successors"}
_RECEIPT_FIELDS = {
    "schema", "episode_id", "game_generation", "player_id", "sequence",
    "decision_turn", "response_turn", "selection", "command_result",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def budgeted_records(records, budget):
    """Check the declared LAN shape/bound before replay or optimizer admission."""
    from .diagnostic_budget998 import validate_collection
    budget = validate_collection(budget)
    _require(type(records) is list and 1 <= len(records) <= budget["max_decisions"],
             "LAN records exceed the declared collection budget")
    _require(all(type(row) is dict and set(row) == RECORD_FIELDS
                 and type(row["before"]) is dict and "turn_mode" in row["before"]
                 for row in records), "Budgeted single-player update requires LAN intervals")
    return budget


def _ready(view):
    controls = view["decisions"]["turn_controls"]
    return controls["active"] and controls["active_complete"]


def official_result(before, final, *, sequence, minimum_turn):
    """Bind the public SessionClient result even when no board is published."""
    from civ6_host.contracts.session_result import SCHEMA

    outcome = final["outcome"]
    result = outcome.get("result")
    fields = {"schema", "player_id", "turn", "team_id", "status", "winning_team", "victory_type", "is_alive"}
    _require(outcome["terminated"] is True and outcome["truncated"] is False
             and outcome["reason"] == "official_native_victory"
             and all(final[k] == before[k] for k in ("episode_id", "game_generation", "turn_mode"))
             and type(final["sequence"]) is int and final["sequence"] == sequence
             and type(result) is dict and set(result) == fields and result["schema"] == SCHEMA
             and all(type(result[k]) is int for k in ("player_id", "turn", "team_id", "winning_team", "victory_type"))
             and result["player_id"] == before["observation"]["player_id"] and result["turn"] >= minimum_turn
             and 0 <= result["team_id"] < 64 and result["winning_team"] != -1 and type(result["is_alive"]) is bool
             and result["status"] == ("won" if result["team_id"] == result["winning_team"] else "lost"),
             "Official LAN result is missing, stale or foreign")
    return result


def transition(record):
    """Validate a same-seat interval without replacing any native boundaries."""
    _require(type(record) is dict and set(record) == RECORD_FIELDS,
             "Invalid public LAN rollout record")
    before, choice = record["before"], record["choice"]
    _require(not lan_inputs.validate(before) and _ready(before),
             "LAN diagnostic requires a ready player's decision")
    successors = record["successors"]
    _require(type(successors) is list and 1 <= len(successors) <= 13,
             "LAN action requires an immediate response and bounded polls")
    receipt = record["command_receipt"]
    _require(type(receipt) is dict and set(receipt) == _RECEIPT_FIELDS
             and receipt["schema"] == "civ6-host.command-receipt.v1",
             "Missing exact SessionClient command receipt")
    action, assignments = selected_action_and_assignments(before, choice)
    selection = dict(decision_id=before["decisions"]["decision_id"], candidate_id=choice["candidate_id"])
    if assignments is not None:
        selection['parameters'] = dict(assignments=assignments)
    pid = before["observation"]["player_id"]
    expected = dict(episode_id=before["episode_id"], game_generation=before["game_generation"],
                    player_id=pid, sequence=before["sequence"] + 1,
                    decision_turn=before["observation"]["turn"], selection=selection)
    _require(all(type(receipt[key]) is type(value) and receipt[key] == value
                 for key, value in expected.items()), "LAN command receipt is stale or foreign")
    command = receipt["command_result"]
    _require(type(command) is dict and set(command) == {"status", "reason", "requested_action"}
             and command["status"] == "completed" and command["requested_action"] == action
             and (command["reason"] is None or type(command["reason"]) is str),
             "LAN command did not complete the sampled action")

    turn = before["observation"]["turn"]
    for index, view in enumerate(successors):
        ended = lan_inputs.validate(view)
        observation = view["observation"]
        _require(view["episode_id"] == before["episode_id"]
                 and view["game_generation"] == before["game_generation"]
                 and view["sequence"] == receipt["sequence"] + index
                 and observation["player_id"] == pid and observation["turn"] >= turn
                 and (observation["width"], observation["height"]) ==
                     (before["observation"]["width"], before["observation"]["height"]),
                 "LAN successor is stale, reordered, missing or foreign")
        _require(view["applied_action"] == (action if index == 0 else None),
                 "LAN successor applies a different or extra action")
        if index == 0:
            _require(type(receipt["response_turn"]) is int and receipt["response_turn"] == observation["turn"],
                     "LAN receipt and immediate response turn differ")
        _require(view["outcome"]["reason"] != "requested_close",
                 "LAN cleanup is not an action successor")
        if index < len(successors) - 1:
            _require(not ended and not _ready(view),
                     "LAN polls must stop at the next ready or terminal view")
        turn = observation["turn"]
    return successors[-1]


def early_task_terminal(scenario, result):
    """A deadline can be scored only after the seat's turn is ready."""
    return (result["terminated"] and result["reason"] != "public_deadline_unmet"
            and not (result["task_success"]
                     and scenario["skill_task"]["objective"]["kind"] == skill_task991.DEFENSE))


def task_step(scenario, before, successors, *, decision_count):
    """Score every observed interval boundary before applying the decision cap."""
    start = before["observation"]
    for view in successors[:-1]:
        result = skill_task991.evaluate_step(scenario["skill_task"], start, view["observation"])
        # The engine can announce the deadline turn before giving the seat
        # its completed turn state. Check the deadline at the acting boundary.
        _require(not early_task_terminal(scenario, result),
                 "LAN interval continued after a task result")
    final = successors[-1]
    ended = final["outcome"]["terminated"]
    result_status = None
    if ended:
        native = official_result(before, final, sequence=before["sequence"] + len(successors),
                                 minimum_turn=final["observation"]["turn"])
        _require(native["turn"] == final["observation"]["turn"], "Official result and final board turns differ")
        result_status = "victory" if native["status"] == "won" else "defeat"
    result = evaluate_skill_transition(scenario, start, final["observation"],
                                       decision_count=decision_count,
                                       official_game_ended=ended, official_game_outcome=result_status)
    early_terminal = early_task_terminal(scenario, result)
    ready = _ready(final)
    if result["terminated"] and not early_terminal:
        _require(ended or ready, "LAN deadline requires a ready or official terminal state")
    _require(ended or early_terminal or ready or final["outcome"]["truncated"],
             "LAN nonterminal interval requires a ready or natively truncated successor")
    return result
