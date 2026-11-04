"""Exact, bounded conditional masks for native policy-slot assignments.

This module has no native or ML dependencies. It consumes the public
``native_policy_catalog`` slot shape, uses unsigned public hashes as matching
vertices, and keeps the signed ``native_hash`` solely for submission mapping.
It never builds or enumerates the Cartesian product of complete assignments.
"""
from __future__ import annotations

import re


MAX_POLICY_SLOTS = 64
MAX_OPTIONS_PER_SLOT = 4096
MAX_POLICY_RELATIONS = 8192
MAX_EDGE_SCANS = 2_000_000
# A complete policy assignment decoder does not make the native action space
# complete; all other native gaps remain outside this isolated helper.
COMPLETE_ACTION_SPACE = False
_POLICY_TYPE = re.compile(r"POLICY_[A-Z0-9_]+\Z")


class PolicyAssignmentLimitError(RuntimeError):
    """The exact catalogue or matching work exceeded a declared hard bound."""


def _int(value, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _validate_slots(slots: list[dict]) -> list[dict]:
    if type(slots) is not list or len(slots) > MAX_POLICY_SLOTS:
        raise ValueError("policy slots must be a bounded native list")
    prior_indices = set()
    by_index = {}
    by_hash = {}
    by_type = {}
    relations = 0
    for slot_number, row in enumerate(slots):
        if (type(row) is not dict or set(row) != {"slot", "prior_index", "options"}
                or not _int(row["slot"], 0, MAX_POLICY_SLOTS - 1)
                or row["slot"] != slot_number
                or not _int(row["prior_index"], -1, 0x7fffffff)
                or type(row["options"]) is not list
                or len(row["options"]) > MAX_OPTIONS_PER_SLOT):
            raise ValueError("malformed native policy slot")
        prior = row["prior_index"]
        if prior != -1:
            if prior in prior_indices:
                raise ValueError("duplicate prior policy occupant")
            prior_indices.add(prior)
        seen_hashes = set()
        previous_index = -1
        for option in row["options"]:
            if (type(option) is not dict
                    or set(option) != {"index", "hash", "native_hash", "policy_type"}
                    or not _int(option["index"], 0, 0x7fffffff)
                    or not _int(option["hash"], 0, 0xffffffff)
                    or not _int(option["native_hash"], -0x80000000, 0xffffffff)
                    or (option["native_hash"] & 0xffffffff) != option["hash"]
                    or type(option["policy_type"]) is not str
                    or len(option["policy_type"]) > 128
                    or _POLICY_TYPE.fullmatch(option["policy_type"]) is None):
                raise ValueError("malformed native policy option")
            index, public_hash, native_hash, policy_type = (
                option["index"], option["hash"], option["native_hash"], option["policy_type"])
            if (public_hash in seen_hashes or index <= previous_index
                    or (index in by_index and by_index[index] != (public_hash, native_hash, policy_type))
                    or (public_hash in by_hash and by_hash[public_hash] != (index, native_hash, policy_type))
                    or (policy_type in by_type and by_type[policy_type] != index)):
                raise ValueError("duplicate, unordered, or inconsistent policy identity")
            seen_hashes.add(public_hash)
            previous_index = index
            by_index[index] = (public_hash, native_hash, policy_type)
            by_hash[public_hash] = (index, native_hash, policy_type)
            by_type[policy_type] = index
            relations += 1
            if relations > MAX_POLICY_RELATIONS:
                raise PolicyAssignmentLimitError("native policy relation bound exceeded")
    for row in slots:
        if any(option["index"] in prior_indices and option["index"] != row["prior_index"]
               for option in row["options"]):
            raise ValueError("active policy relocation is outside this adapter")
    return slots


class _WorkBudget:
    def __init__(self):
        self.scans = 0

    def scan(self):
        self.scans += 1
        if self.scans > MAX_EDGE_SCANS:
            raise PolicyAssignmentLimitError("exact policy matching work bound exceeded")


def _perfect_matching(slots: list[dict], remaining: list[int], unavailable: set[int],
                      budget: _WorkBudget) -> dict[int, dict] | None:
    """Return one exact left-perfect matching or None using augmenting paths."""
    if not remaining:
        return {}
    # Fewest remaining edges first reduces work but does not discard choices.
    order = sorted(remaining, key=lambda i: (
        sum(option["hash"] not in unavailable for option in slots[i]["options"]), i))
    hash_owner: dict[int, int] = {}
    slot_option: dict[int, dict] = {}

    def augment(slot_index: int, visited_hashes: set[int]) -> bool:
        for option in slots[slot_index]["options"]:
            budget.scan()
            public_hash = option["hash"]
            if public_hash in unavailable or public_hash in visited_hashes:
                continue
            visited_hashes.add(public_hash)
            previous_slot = hash_owner.get(public_hash)
            if previous_slot is None or augment(previous_slot, visited_hashes):
                hash_owner[public_hash] = slot_index
                slot_option[slot_index] = option
                return True
        return False

    for slot_index in order:
        if not augment(slot_index, set()):
            return None
    return slot_option


def _has_changed_completion(slots: list[dict], remaining: list[int], unavailable: set[int],
                            changed: bool, budget: _WorkBudget) -> bool:
    matching = _perfect_matching(slots, remaining, unavailable, budget)
    if matching is None:
        return False
    if changed or any(matching[i]["index"] != slots[i]["prior_index"] for i in remaining):
        return True

    # The matching found is the unchanged prior assignment. A different valid
    # assignment exists iff some non-prior edge can be forced while the
    # remaining slots still have a perfect matching. This tests edges, never
    # complete policy bundles.
    for slot_index in remaining:
        for option in slots[slot_index]["options"]:
            budget.scan()
            if (option["index"] == slots[slot_index]["prior_index"]
                    or option["hash"] in unavailable):
                continue
            suffix = [i for i in remaining if i != slot_index]
            if _perfect_matching(slots, suffix, unavailable | {option["hash"]}, budget) is not None:
                return True
    return False


def valid_next_hashes(slots: list[dict], prefix: list[int]) -> list[int]:
    """Return every next-slot public hash with a unique changed completion.

    ``prefix`` is a dense sequence of unsigned public hashes, one for each
    preceding slot. Malformed, duplicate, or out-of-menu prefixes raise
    ``ValueError``. A syntactically valid dead-end prefix returns an empty
    mask. A valid complete assignment has no next choice and returns ``[]``.
    """
    _validate_slots(slots)
    if type(prefix) is not list or len(prefix) > len(slots):
        raise ValueError("policy prefix must be a dense bounded list")
    used: set[int] = set()
    changed = False
    for slot_index, public_hash in enumerate(prefix):
        if not _int(public_hash, 0, 0xffffffff):
            raise ValueError("policy prefix hashes must be unsigned public hashes")
        if public_hash in used:
            raise ValueError("policy prefix repeats a policy hash")
        matches = [option for option in slots[slot_index]["options"]
                   if option["hash"] == public_hash]
        if len(matches) != 1:
            raise ValueError("policy prefix choice is absent from its slot menu")
        used.add(public_hash)
        changed = changed or matches[0]["index"] != slots[slot_index]["prior_index"]

    if not slots:
        return []
    if len(prefix) == len(slots):
        if not changed:
            raise ValueError("complete policy prefix is unchanged")
        return []

    budget = _WorkBudget()
    next_slot = len(prefix)
    suffix = list(range(next_slot + 1, len(slots)))
    legal = []
    for option in slots[next_slot]["options"]:
        public_hash = option["hash"]
        if public_hash in used:
            continue
        next_used = used | {public_hash}
        next_changed = changed or option["index"] != slots[next_slot]["prior_index"]
        if _has_changed_completion(slots, suffix, next_used, next_changed, budget):
            legal.append(public_hash)
    return legal


def validate_joint_assignment(slots: list[dict], assignments: list[int]) -> dict:
    """Validate a complete assignment and separate content hashes from submit hashes.

    Public ``hash`` values determine menu membership and uniqueness. Returned
    ``native_hashes`` preserve the catalogue's signed/unsigned engine numeric
    representation exclusively for the native submission boundary.
    """
    _validate_slots(slots)
    if (not slots or type(assignments) is not list or len(assignments) != len(slots)
            or any(not _int(value, 0, 0xffffffff) for value in assignments)):
        raise ValueError("policy assignment must fill every native slot")
    if len(set(assignments)) != len(assignments):
        raise ValueError("policy assignment hashes must be unique")
    selected = []
    for slot_index, public_hash in enumerate(assignments):
        matches = [option for option in slots[slot_index]["options"]
                   if option["hash"] == public_hash]
        if len(matches) != 1:
            raise ValueError("policy assignment is absent from its slot menu")
        selected.append(matches[0])
    if all(option["index"] == slots[i]["prior_index"] for i, option in enumerate(selected)):
        raise ValueError("policy assignment is unchanged")
    return {
        "policy_indices": [option["index"] for option in selected],
        "policy_hashes": [option["hash"] for option in selected],
        "native_hashes": [option["native_hash"] for option in selected],
        "policy_types": [option["policy_type"] for option in selected],
    }
