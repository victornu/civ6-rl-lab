"""Transport admission tests for a bounded single-player LAN diagnostic."""
import threading
import unittest
from unittest.mock import Mock, patch

from civ6rl.learning.diagnostic_budget998 import EXTENDED_SCHEMA
from civ6rl.learning.lan_rollout import RECORD_FIELDS
from civ6rl.learning.transport import TransportError, WorkerClient


POLICY_VERSION = "a" * 64
NEXT_POLICY_VERSION = "b" * 64


def budget(max_decisions=16):
    return {
        "schema": EXTENDED_SCHEMA,
        "max_decisions": max_decisions,
        "max_updates": 1,
        "max_rounds": 20,
        "wall_seconds": 180,
        "request_seconds": 60,
    }


def lan_record(sequence=0):
    return {
        "before": {
            "turn_mode": "simultaneous",
            "episode_id": "episode-1",
            "game_generation": 1,
            "sequence": sequence,
        },
        "choice": {"candidate_id": "candidate-1"},
        "command_receipt": {"status": "completed"},
        "successors": [{"sequence": sequence + 1}],
    }


def update_ack(collection):
    return {
        "training_eligible": False,
        "mode": "diagnostic_native",
        "model_status": "diagnostic_updated",
        "previous_policy_version": POLICY_VERSION,
        "policy_version": NEXT_POLICY_VERSION,
        "collection": collection,
    }


def mocked_client(collection, ack=None):
    """Build an initialized client without starting a worker process."""
    worker = WorkerClient.__new__(WorkerClient)
    worker._lock = threading.RLock()
    worker._initialized = True
    worker._evaluation_only = False
    worker.policy_version = POLICY_VERSION
    worker.critic_mode = "player_only"
    worker.critic_version = None
    worker._diagnostic_collection = dict(collection)
    worker._diagnostic_collection_updates = 0
    worker._campaign_contract = None
    worker._exchange = Mock(return_value=ack if ack is not None else update_ack(collection))
    worker._fail_closed = Mock()
    return worker


class LanBudgetTransportTests(unittest.TestCase):
    def test_remaining_collection_time_reserves_update_and_cleanup_after_elapsed_work(self):
        worker = mocked_client(budget())
        worker._started = 20.
        worker.total_timeout = 180.
        worker.request_timeout = 60.
        with patch("civ6rl.learning.transport.time.monotonic", return_value=100.):
            self.assertEqual(worker.collection_seconds_remaining, 10.)
        with patch("civ6rl.learning.transport.time.monotonic", return_value=111.):
            self.assertEqual(worker.collection_seconds_remaining, 0.)
        copy = worker.diagnostic_collection
        copy["max_decisions"] = 256
        self.assertEqual(worker.diagnostic_collection["max_decisions"], 16)

    def test_legacy_policy_semantics_cannot_be_promoted_by_a_forged_lan_schema(self):
        from tests.test_learning_transport import policy_response
        from civ6rl.learning.lan_inputs import policy_reconfirmation_allowed
        view = policy_response()
        self.assertFalse(policy_reconfirmation_allowed(view))
        view["decisions"]["schema"] = "civ6-host.network-capability-decisions.v29"
        with self.assertRaisesRegex(ValueError, "public SessionClient LAN view"):
            policy_reconfirmation_allowed(view)

    def test_single_lan_update_accepts_under_budget_acknowledges_exact_collection(self):
        contract = budget()
        worker = mocked_client(contract)

        result = worker.update_diagnostic([lan_record()], {"schema": "scenario.v1"})

        self.assertEqual("diagnostic_native", result["mode"])
        self.assertEqual(contract, result["collection"])
        payload = worker._exchange.call_args.args[0]
        self.assertEqual("update_diagnostic", payload["op"])
        self.assertEqual([lan_record()], payload["records"])
        self.assertEqual(1, worker._diagnostic_collection_updates)

    def test_over_budget_records_are_rejected_before_exchange(self):
        contract = budget(max_decisions=2)
        worker = mocked_client(contract)

        with self.assertRaises((ValueError, TransportError)):
            worker.update_diagnostic([lan_record(0), lan_record(1), lan_record(2)],
                                     {"schema": "scenario.v1"})

        worker._exchange.assert_not_called()

    def test_non_lan_records_are_rejected_before_exchange(self):
        worker = mocked_client(budget())
        record = lan_record()
        del record["before"]["turn_mode"]

        with self.assertRaises((ValueError, TransportError)):
            worker.update_diagnostic([record], {"schema": "scenario.v1"})

        worker._exchange.assert_not_called()

    def test_altered_collection_ack_is_rejected_and_client_fails_closed(self):
        contract = budget()
        altered = dict(contract, max_decisions=contract["max_decisions"] - 1)
        worker = mocked_client(contract, update_ack(altered))

        with self.assertRaises(TransportError):
            worker.update_diagnostic([lan_record()], {"schema": "scenario.v1"})

        worker._fail_closed.assert_called_once()

    def test_negotiated_single_update_budget_is_consumed_after_one_update(self):
        worker = mocked_client(budget())
        worker.update_diagnostic([lan_record()], {"schema": "scenario.v1"})

        with self.assertRaises(TransportError):
            worker.update_diagnostic([lan_record(1)], {"schema": "scenario.v1"})

        self.assertEqual(1, worker._exchange.call_count)


if __name__ == "__main__":
    unittest.main()
