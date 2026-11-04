"""Stdlib protocol tests; deliberately do not import the Torch worker/model."""
import sys
import unittest
import subprocess

from civ6rl.learning.transport import MAX_LINE_BYTES, TransportError, WorkerClient
from civ6rl.learning.wire import MAX_DECODED_BYTES, decode_line, encode_line


FAKE_WORKER = r'''
import json, sys, time
mode = sys.argv[1]
for line in sys.stdin:
    req = json.loads(line)
    if req["op"] == "init":
        result = {"id": req["id"], "ok": True, "model_status": "untrained", "device": req["device"], "training_eligible": False}
    else:
        if mode == "timeout":
            time.sleep(1.0)
        if mode == "framing":
            sys.stdout.buffer.write(b"x" * (int(sys.argv[2]) + 1) + b"\n")
            sys.stdout.flush()
            break
        result = {"id": req["id"], "ok": True,
                  "episode_id": "ep", "player_id": 1,
                  "sequence": 2 if mode == "stale" else 1,
                  "candidate_id": "outside" if mode == "nonmember" else "c1",
                  "log_prob": -0.4, "value": 0.2, "entropy": 0.5,
                  "inference_seconds": 0.01, "training_eligible": False}
        if mode in ("assignments", "bad_assignment"):
            result["assignments"] = [101, 202] if mode == "assignments" else [101, 999]
        if mode == "nonfinite":
            result["value"] = float("nan")
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
    sys.stdout.flush()
'''


def response(candidate="c1"):
    return {"episode_id": "ep", "sequence": 1,
            "observation": {"player_id": 1},
            "decisions": {"candidates": [{"candidate_id": candidate, "action": {}}]}}


def policy_response():
    slots = [
        {"slot": 0, "prior_index": 8, "options": [
            {"index": 0, "hash": 100, "native_hash": 100, "policy_type": "POLICY_A"},
            {"index": 1, "hash": 101, "native_hash": 101, "policy_type": "POLICY_B"}]},
        {"slot": 1, "prior_index": 9, "options": [
            {"index": 2, "hash": 200, "native_hash": 200, "policy_type": "POLICY_C"},
            {"index": 3, "hash": 202, "native_hash": 202, "policy_type": "POLICY_D"}]},
    ]
    value = response()
    value["observation"]["turn"] = 7
    value["decisions"]["policies"] = {"player_id": 1, "turn": 7, "slots": slots}
    value["decisions"]["candidates"][0]["action"] = {"kind": "policies", "slots": slots}
    return value


def client(mode, timeout=2.0):
    result = WorkerClient([sys.executable, "-u", "-c", FAKE_WORKER, mode, str(MAX_LINE_BYTES)],
                          request_timeout=timeout, total_timeout=10.0)
    result.initialize('{"version":"v","tokens":[]}', device="cpu", seed=1)
    return result


class LearningTransportTests(unittest.TestCase):
    def test_valid_round_trip_and_candidate_identity(self):
        worker = client("ok")
        try:
            result = worker.infer(response())
            self.assertEqual("c1", result["candidate_id"])
            self.assertEqual("ep", result["episode_id"])
            self.assertFalse(result["training_eligible"])
        finally:
            worker.close()
        self.assertIsNotNone(worker._process.poll())

    def test_stale_worker_sequence_is_rejected_and_reaped(self):
        worker = client("stale")
        with self.assertRaisesRegex(TransportError, "identity mismatch"):
            worker.infer(response())
        self.assertIsNotNone(worker._process.poll())

    def test_candidate_outside_native_menu_is_rejected_and_reaped(self):
        worker = client("nonmember")
        with self.assertRaisesRegex(TransportError, "outside the published menu"):
            worker.infer(response())
        self.assertIsNotNone(worker._process.poll())

    def test_joint_policy_assignment_round_trips_only_for_published_policy_candidate(self):
        worker = client("assignments")
        try:
            result = worker.infer(policy_response())
            self.assertEqual([101, 202], result["assignments"])
        finally:
            worker.close()

    def test_nonmember_joint_assignment_is_rejected_and_reaped(self):
        worker = client("bad_assignment")
        with self.assertRaisesRegex(TransportError, "invalid or nonmember policy assignment"):
            worker.infer(policy_response())
        self.assertIsNotNone(worker._process.poll())

    def test_joint_assignment_is_rejected_for_non_policy_action(self):
        worker = client("assignments")
        with self.assertRaisesRegex(TransportError, "non-policy action"):
            worker.infer(response())
        self.assertIsNotNone(worker._process.poll())

    def test_request_timeout_kills_and_reaps_worker(self):
        worker = client("timeout", timeout=0.1)
        with self.assertRaisesRegex(TransportError, "timed out"):
            worker.infer(response())
        self.assertIsNotNone(worker._process.poll())

    def test_oversized_worker_frame_is_rejected_and_reaped(self):
        worker = client("framing")
        with self.assertRaisesRegex(TransportError, "framing limit"):
            worker.infer(response())
        self.assertIsNotNone(worker._process.poll())

    def test_request_over_decoded_limit_is_rejected_before_write(self):
        worker = client("ok")
        payload = response()
        payload["padding"] = "x" * MAX_DECODED_BYTES
        try:
            with self.assertRaisesRegex(TransportError, "8 MiB"):
                worker.infer(payload)
            self.assertEqual(worker._requests, 1)
        finally:
            worker.close()

    def test_non_finite_result_metrics_are_rejected(self):
        worker = client("nonfinite")
        with self.assertRaisesRegex(TransportError, "malformed JSON"):
            worker.infer(response())
        self.assertIsNotNone(worker._process.poll())

    def test_large_request_and_worker_send_reply_round_trip_losslessly(self):
        code = '''
import sys
from civ6rl.learning.wire import decode_line, MAX_LINE_BYTES
from civ6rl.learning.worker import _send
for line in iter(lambda: sys.stdin.buffer.readline(MAX_LINE_BYTES + 1), b''):
    request = decode_line(line)
    _send({'id': request['id'], 'ok': True, 'data': request['data']})
'''
        worker = WorkerClient([sys.executable, '-B', '-u', '-c', code], total_timeout=10)
        payload = {'op':'echo', 'data': [{'actor': [1, 2, 3.25], 'text':'économie'}] * 35000}
        try:
            result = worker._exchange(payload)
            self.assertEqual(result['data'], payload['data'])
        finally:
            worker.close()
        self.assertIsNotNone(worker._process.poll())

    def test_real_worker_decodes_large_frame_before_rejecting_unknown_operation(self):
        raw = encode_line({'id': 17, 'op': 'wire_only_test', 'padding': 'x' * MAX_LINE_BYTES})
        process = subprocess.run([sys.executable, '-B', '-m', 'civ6rl.learning.worker'],
            input=raw, capture_output=True, timeout=10,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(process.returncode, 0, process.stderr)
        response = decode_line(process.stdout)
        self.assertEqual(response, {'id': 17, 'ok': False, 'error': 'unsupported_operation'})


if __name__ == "__main__":
    unittest.main()
