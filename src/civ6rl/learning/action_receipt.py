"""Pure-Python verification of learner choices against native action receipts."""
from __future__ import annotations


def selected_action_and_assignments(before: dict, choice: dict) -> tuple[dict, list[int] | None]:
    """Return the exact expected applied action and joint policy hashes.

    Policy masks come from the selected published candidate. The applied
    receipt has the native ``native_policy_catalog.choose`` shape, including
    signed native hashes. Other action kinds reject an assignments field even
    when it is null, so a joint policy output cannot be silently ignored.
    """
    if type(before) is not dict or type(choice) is not dict:
        raise ValueError("native response and worker choice must be objects")
    candidates = before.get("decisions", {}).get("candidates")
    candidate_id = choice.get("candidate_id")
    if type(candidates) is not list or not isinstance(candidate_id, str):
        raise ValueError("candidate choice is missing its published menu")
    matches = [row for row in candidates
               if type(row) is dict and row.get("candidate_id") == candidate_id]
    if len(matches) != 1 or type(matches[0].get("action")) is not dict:
        raise ValueError("candidate choice is absent or ambiguous")
    action = matches[0]["action"]
    if action.get("kind") != "policies":
        if "assignments" in choice:
            raise ValueError("non-policy action must not carry joint assignments")
        return action, None

    assignments = choice.get("assignments")
    if (type(assignments) is not list or not assignments
            or any(type(value) is not int or not 0 <= value <= 0xffffffff for value in assignments)):
        raise ValueError("policy action requires unsigned joint assignment hashes")
    from .policy_assignments import validate_joint_assignment
    validated = validate_joint_assignment(action.get("slots"), assignments)
    obs = before.get("observation")
    if type(obs) is not dict or type(obs.get("player_id")) is not int or type(obs.get("turn")) is not int:
        raise ValueError("policy receipt lacks native player and turn identity")
    slots = action["slots"]
    candidate = {
        "status": "candidate",
        "player_id": obs["player_id"],
        "turn": obs["turn"],
        "slot_count": len(slots),
        "assignments": [[i, value] for i, value in enumerate(validated["native_hashes"])],
    }
    prior_indices = [row["prior_index"] for row in slots]
    if any(value != -1 for value in prior_indices):
        candidate["prior_indices"] = prior_indices
    return {"kind": "policies", "candidate": candidate}, list(assignments)


def verify_applied_action(before: dict, choice: dict, after: dict) -> list[int] | None:
    """Require the immediate native receipt to match the chosen action exactly."""
    expected, assignments = selected_action_and_assignments(before, choice)
    if type(after) is not dict or after.get("applied_action") != expected:
        raise ValueError("native applied action differs from the selected candidate or assignment")
    if after.get("automatic_choices_applied") is not False:
        raise ValueError("native action receipt includes automatic choices")
    return assignments


__all__ = ["selected_action_and_assignments", "verify_applied_action"]
