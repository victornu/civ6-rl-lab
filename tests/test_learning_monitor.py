import json
import io
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from civ6rl.learning.monitor import MonitorService, append_metric, make_handler
from civ6rl.learning.run_control import RunController


class LearningMonitorTests(unittest.TestCase):
    def setUp(self):
        # Keep temporary artifacts in the test workspace (some sandboxed
        # Windows environments deny writes to the system Temp directory).
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name) / "runs"
        self.root.mkdir()
        self.service = MonitorService(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_dir(self, run_id="r1"):
        folder = self.root / run_id
        folder.mkdir(exist_ok=True)
        (folder / "manifest.json").write_text(json.dumps({
            "schema": "civ6rl.learning_run.v1", "run_id": run_id,
            "mode": "diagnostic", "status": "running", "started_at": "now",
            "training_eligible": False, "limitations": ["diagnostic only"]
        }), encoding="utf-8")
        return folder

    def test_empty_runs_reports_no_data_without_inventing_metrics(self):
        self.assertEqual({"runs": [], "no_data": True}, self.service.runs())
        self.assertEqual(404, self.service.run("missing")[0])

    def test_run_tolerates_incomplete_tail_but_reports_bad_middle_line(self):
        folder = self.run_dir()
        with (folder / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            stream.write('{"step":0,"wall_seconds":1,"stage":"train","metrics":{"reward":2}}\n')
            stream.write('{not json}\n')
            stream.write('{"step":1')
        status, data = self.service.run("r1")
        self.assertEqual(200, status)
        self.assertTrue(data["tail_incomplete"])
        self.assertIn("malformed_middle", data["errors"])
        self.assertEqual(1, len(data["metrics"]))
        self.assertEqual(2, data["metrics"][0]["metrics"]["reward"])

    def test_path_traversal_is_rejected(self):
        folder = self.run_dir()
        outside = Path(self.temp.name) / "secret"
        outside.mkdir()
        (outside / "manifest.json").write_text("{}", encoding="utf-8")
        for malicious in ("../secret", str(outside), "..\\secret", "%2e%2e"):
            self.assertEqual(404, self.service.run(malicious)[0])
            self.assertEqual(404, self.service.replay(malicious, 0)[0])

    def test_diagnostic_manifest_never_claims_training_eligibility(self):
        self.run_dir()
        result = self.service.runs()["runs"][0]["manifest"]
        self.assertEqual("diagnostic", result["mode"])
        self.assertFalse(result["training_eligible"])
        self.assertNotIn("critic_mode", result)

    def test_task_context_is_optional_diagnostic_and_strictly_whitelisted(self):
        folder = self.run_dir("task-context")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        digest = "a" * 64
        manifest["task_context"] = {
            "objective": "economy_gold_threshold", "target": 1200,
            "max_decisions": 12, "max_turns": 30,
            "scenario_sha256": digest.upper(), "scope": "diagnostic",
            "scenario_id": "private-id", "seed": 876543, "features": ["hidden"],
        }
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        _, data = self.service.run("task-context")
        self.assertEqual({"objective": "economy_gold_threshold", "target": 1200,
                          "max_decisions": 12, "max_turns": 30,
                          "scenario_sha256": digest, "scope": "diagnostic"},
                         data["manifest"]["task_context"])

        manifest["task_context"]["target"] = True
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        _, data = self.service.run("task-context")
        self.assertNotIn("task_context", data["manifest"])

    def test_replay_publishes_only_valid_scenario_hash(self):
        folder = self.run_dir("scenario-replay")
        digest = "b" * 64
        (folder / "replays.jsonl").write_text(json.dumps({
            "step": 1, "turn": 2, "player_id": 0, "scenario_sha256": digest.upper(),
            "scenario_id": "private-id", "seed": 4, "observation": {"width": 1},
        }) + "\n", encoding="utf-8")
        status, data = self.service.replay("scenario-replay", 0)
        self.assertEqual(200, status)
        self.assertEqual(digest, data["replay"]["scenario_sha256"])
        self.assertNotIn("scenario_id", data["replay"])
        self.assertNotIn("seed", data["replay"])

        row = json.loads((folder / "replays.jsonl").read_text(encoding="utf-8"))
        row["scenario_sha256"] = "bad-hash"
        (folder / "replays.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        status, data = self.service.replay("scenario-replay", 0)
        self.assertEqual(200, status)
        self.assertNotIn("scenario_sha256", data["replay"])

    def test_missing_measurements_do_not_create_fake_metrics_or_rate(self):
        self.run_dir()
        _, data = self.service.run("r1")
        self.assertEqual([], data["metrics"])
        self.assertTrue(data["no_data"])

    def test_append_metric_rejects_nonfinite_values(self):
        folder = self.run_dir()
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                append_metric(folder, 1, 1.0, "train", {"loss": value})
        with self.assertRaises(ValueError):
            append_metric(folder, 1, math.nan, "train", {})
        self.assertFalse((folder / "metrics.jsonl").exists())

    def test_append_metric_writes_finite_jsonl_sample(self):
        folder = self.run_dir()
        append_metric(folder, 3, 1.5, "update", {"reward": 2.25, "loss": 0})
        row = json.loads((folder / "metrics.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(3, row["step"])
        self.assertEqual({"reward": 2.25, "loss": 0}, row["metrics"])

    def test_replay_whitelists_published_observation_fields(self):
        folder = self.run_dir()
        replay = {"step": 8, "turn": 4, "player_id": 1,
                  "observation": {"width": 3, "map": {"width": 3, "tiles": [{"x": 0, "y": 0, "terrain": "grass", "secret": "hidden"}], "secret": "hidden"},
                                  "tiles": [{"x": 0, "y": 0, "terrain": "grass", "secret": "hidden"},
                                            {"x": 1, "y": 0, "terrain": "unseen", "visible": False}],
                                  "units": [{"x": 1, "y": 1, "type": "scout", "secret": "hidden"}],
                                  "raw_state": {"fogged": "hidden"}},
                  "action": {"type": "move", "target_x": 2, "secret": "hidden"},
                  "save_file": "secret.sav"}
        (folder / "replays.jsonl").write_text(json.dumps(replay) + "\n", encoding="utf-8")
        status, data = self.service.replay("r1", 0)
        self.assertEqual(200, status)
        obs = data["replay"]["observation"]
        self.assertNotIn("raw_state", obs)
        self.assertNotIn("secret", obs["map"])
        self.assertNotIn("secret", obs["tiles"][0])
        self.assertNotIn("terrain", obs["tiles"][1])
        self.assertNotIn("save_file", data["replay"])
        self.assertNotIn("secret", data["replay"]["action"])

    def test_native_replay_preserves_revealed_plots_and_visible_enemy_units(self):
        folder = self.run_dir("native-replay-mps")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(error="collector stopped: " + "x" * 3000, close_error="close failed")
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        replay = {"step": 0, "player_id": 0, "turn": 135,
                  "observation": {"schema": "civ6rl-player-observation-v1", "player_id": 0,
                                  "turn": 135, "width": 44, "height": 26,
                                  "plots": [
                                      {"index": 10, "x": 10, "y": 0, "visible": True, "revealed": True,
                                       "terrain_type": 15, "feature_type": -1, "water": True},
                                      {"index": 11, "x": 11, "y": 0, "visible": False, "revealed": True,
                                       "terrain_type": 3, "feature_type": 2},
                                      {"index": 12, "x": 12, "y": 0, "visible": False, "revealed": False,
                                       "terrain_type": 77, "secret": "fog"}],
                                  "units": [{"id": 7, "type": 21, "x": 10, "y": 1, "owner": 0,
                                             "damage": 0, "moves": 2, "max_moves": 2}],
                                  "visible_enemy_units": [
                                      {"id": 9, "type": 38, "x": 11, "y": 1, "owner": 2, "visible": True},
                                      {"id": 10, "type": 38, "x": 12, "y": 1, "owner": 2, "visible": False}],
                                  "cities": [{"id": 20, "x": 10, "y": 2, "owner": 0, "population": 5}],
                                  "secret_state": {"hidden": True}},
                  "action": {"kind": "unit_order", "unit_id": 7, "route": "command",
                             "item_type": "UNITCOMMAND_WAKE", "hash": 200562917, "secret": "drop"}}
        (folder / "replays.jsonl").write_text(json.dumps(replay) + "\n", encoding="utf-8")
        (folder / "metrics.jsonl").write_text(json.dumps({
            "step": 1, "wall_seconds": 0.004, "stage": "recorded_native_inference",
            "metrics": {"inference_seconds": 0.0038, "steps_per_second": 264.0}
        }) + "\n", encoding="utf-8")

        status, data = self.service.replay("native-replay-mps", 0)
        self.assertEqual(200, status)
        obs = data["replay"]["observation"]
        self.assertEqual(2, len(obs["plots"]))
        self.assertEqual(15, obs["plots"][0]["terrain_type"])
        self.assertFalse(obs["plots"][1]["visible"])
        self.assertEqual(3, obs["plots"][1]["terrain_type"])
        self.assertEqual([9], [u["id"] for u in obs["visible_enemy_units"]])
        self.assertEqual("unit_order", data["replay"]["action"]["kind"])
        self.assertEqual("UNITCOMMAND_WAKE", data["replay"]["action"]["item_type"])
        self.assertNotIn("secret", data["replay"]["action"])
        self.assertNotIn("secret_state", obs)
        self.assertNotIn("secret", obs["plots"][1])

        _, run = self.service.run("native-replay-mps")
        self.assertEqual("recorded_inference", run["manifest"]["metric_scope"])
        self.assertEqual(2000, len(run["manifest"]["error"]))
        self.assertEqual("close failed", run["manifest"]["close_error"])

    def test_live_run_progress_and_training_metadata_are_whitelisted(self):
        folder = self.run_dir("live")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(requested_steps=20, executed_steps=4, native_choices=4,
                        heartbeat_at="2026-10-16T11:01:29Z", progress_timeout_seconds=45,
                        model_status="untrained", native_supervisor={"last_sequence": 4,
                        "reaped": True, "secret_path": "C:/private/save.sav"})
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (folder / "metrics.jsonl").write_text(json.dumps({
            "step": 4, "wall_seconds": 1.0, "stage": "update",
            "metrics": {"transport_seconds": 0.02, "worker_start_seconds": 0.3,
                        "gae_loss": 1.25, "policy_loss": -0.2, "grad_norm": 0.8}
        }) + "\n", encoding="utf-8")
        _, data = self.service.run("live")
        safe = data["manifest"]
        self.assertEqual(20, safe["requested_steps"])
        self.assertEqual(4, safe["executed_steps"])
        self.assertEqual("untrained", safe["model_status"])
        self.assertEqual("2026-10-16T11:01:29Z", safe["heartbeat_at"])
        self.assertEqual(45, safe["progress_timeout_seconds"])
        self.assertEqual({"last_sequence": 4, "reaped": True}, safe["native_supervisor"])
        self.assertNotIn("secret_path", safe["native_supervisor"])
        self.assertEqual(0.02, data["metrics"][0]["metrics"]["transport_seconds"])
        self.assertEqual(1.25, data["metrics"][0]["metrics"]["gae_loss"])

    def test_run_controller_counters_errors_and_diagnostic_updated_status_are_whitelisted(self):
        folder = self.run_dir("updated")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(status="stopped", requested_steps=50, executed_steps=12,
                        requested_updates=8, executed_updates=3,
                        requested_episodes=4, completed_episodes=1,
                        max_seconds=900.0, elapsed_seconds=121.5,
                        limits_are_caps=True, step_limit=50, update_limit=8,
                        episode_limit=4, time_limit_seconds=900.0,
                        stop_reason="no_progress_timeout", error="bounded caller error",
                        model_status="diagnostic_updated", metric_scope="diagnostic",
                        critic_mode="privileged_diagnostic",
                        private_path="C:/private/save.sav")
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        _, data = self.service.run("updated")
        safe = data["manifest"]
        self.assertEqual(8, safe["requested_updates"])
        self.assertEqual(3, safe["executed_updates"])
        self.assertEqual(4, safe["requested_episodes"])
        self.assertEqual(1, safe["completed_episodes"])
        self.assertEqual(900.0, safe["max_seconds"])
        self.assertEqual(121.5, safe["elapsed_seconds"])
        self.assertEqual("no_progress_timeout", safe["stop_reason"])
        self.assertEqual("diagnostic_updated", safe["model_status"])
        self.assertEqual("diagnostic", safe["metric_scope"])
        self.assertTrue(safe["limits_are_caps"])
        self.assertEqual(50, safe["step_limit"])
        self.assertEqual(8, safe["update_limit"])
        self.assertEqual(4, safe["episode_limit"])
        self.assertEqual(900.0, safe["time_limit_seconds"])
        self.assertEqual("privileged_diagnostic", safe["critic_mode"])
        self.assertNotIn("private_path", safe)

    def test_http_run_and_dashboard_publish_diagnostic_update_summary(self):
        folder = self.run_dir("http-updated")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(status="completed", model_status="diagnostic_updated",
                        requested_updates=6, executed_updates=2,
                        requested_episodes=3, completed_episodes=1,
                        elapsed_seconds=18.5, max_seconds=90,
                        limits_are_caps=True, step_limit=8, update_limit=6,
                        episode_limit=3, time_limit_seconds=90,
                        requested_steps=8, executed_steps=6,
                        stop_reason="limit_reached:max_updates", metric_scope="diagnostic",
                        critic_mode="history_only_diagnostic")
        (folder / "metrics.jsonl").write_text(json.dumps({
            "step": 1, "wall_seconds": 2.0, "stage": "ppo_update",
            "metrics": {"separate_value_loss": 0.45, "separate_grad_norm": 0.7,
                        "privileged_value_loss": 99.0, "privileged_grad_norm": 99.0}
        }) + "\n", encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        class CapturingHandler(make_handler(self.service)):
            def __init__(self, path):
                self.path = path
                self.wfile = io.BytesIO()
                self.headers_sent = []
                self.status = None

            def send_response(self, status):
                self.status = status

            def send_header(self, key, value):
                self.headers_sent.append((key, value))

            def end_headers(self):
                pass

        api = CapturingHandler("/api/run?id=http-updated")
        api.do_GET()
        self.assertEqual(200, api.status)
        payload = json.loads(api.wfile.getvalue().decode("utf-8"))
        summary = payload["manifest"]
        self.assertEqual(6, summary["requested_updates"])
        self.assertEqual(2, summary["executed_updates"])
        self.assertEqual(3, summary["requested_episodes"])
        self.assertEqual(1, summary["completed_episodes"])
        self.assertEqual("limit_reached:max_updates", summary["stop_reason"])
        self.assertEqual("diagnostic_updated", summary["model_status"])
        self.assertEqual("history_only_diagnostic", summary["critic_mode"])
        self.assertEqual(0.45, payload["metrics"][0]["metrics"]["separate_value_loss"])
        self.assertNotIn("privileged_value_loss", payload["metrics"][0]["metrics"])
        self.assertNotIn("privileged_grad_norm", payload["metrics"][0]["metrics"])
        self.assertEqual(8, summary["step_limit"])
        self.assertTrue(summary["limits_are_caps"])

        page = CapturingHandler("/dashboard.html")
        page.do_GET()
        self.assertEqual(200, page.status)
        html = page.wfile.getvalue().decode("utf-8")
        self.assertIn("Diagnostic mis à jour · entraînement non éligible", html)
        self.assertIn("Updates PPO réalisées", html)
        self.assertIn("Critiqueur séparé — historique joueur seul", html)
        self.assertIn("separate_value_loss", html)
        self.assertIn("separate_grad_norm", html)

    def test_complete_does_not_overwrite_a_prior_stop_or_failure(self):
        class Clock:
            value = 0.0

            def monotonic(self):
                return self.value

            def now(self):
                return datetime(2026, 10, 16, tzinfo=timezone.utc)

        clock = Clock()
        stopped = RunController(self.root / "controller-stopped", run_id="stopped",
                                max_seconds=5, max_updates=10, progress_timeout_seconds=None,
                                monotonic=clock.monotonic, utc_now=clock.now)
        clock.value = 5
        self.assertEqual("stopped", stopped.check())
        stopped.complete("late_completion_call")
        stopped_manifest = json.loads(stopped.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual("stopped", stopped_manifest["status"])
        self.assertEqual("time_limit", stopped_manifest["stop_reason"])

        failed = RunController(self.root / "controller-failed", run_id="failed",
                               max_seconds=5, max_updates=10, progress_timeout_seconds=None,
                               monotonic=clock.monotonic, utc_now=clock.now)
        failed.fail("native host failed before terminal observation", reason="host_error")
        failed.complete("caller_cleanup")
        failed_manifest = json.loads(failed.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual("failed", failed_manifest["status"])
        self.assertEqual("host_error", failed_manifest["stop_reason"])
        self.assertIn("native host failed", failed_manifest["error"])

    def test_dashboard_labels_recorded_inference_and_unexecuted_proposal(self):
        html = (Path(__file__).parents[1] / "civ6rl" / "learning" / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("Inférences/s mesurées", html)
        self.assertIn("inférence enregistrée, non exécutée", html)
        self.assertIn("visible_enemy_units", html)
        self.assertIn("updateRunOverview(d.manifest,metrics)", html)
        self.assertIn("updateRunHealth(d.manifest)", html)
        self.assertIn("transport_seconds", html)
        self.assertIn("worker_start_seconds", html)
        self.assertIn("gae_loss", html)
        self.assertIn("gradient_norm", html)
        self.assertIn("Aucune progression depuis le seuil fourni", html)
        self.assertIn("heartbeat et seuil de progression requis", html)
        self.assertIn("Erreur de chargement du replay : ", html)
        self.assertIn("Aucun replay publié pour ce run.", html)
        self.assertIn("Run avec erreur", html)
        self.assertIn("civ6rl.learning_run.v1", html)
        self.assertEqual(1, html.count("function drawReplay("))
        self.assertEqual(1, html.count("function showTimings("))
        self.assertIn("dernier ${nice(latest)}", html)
        self.assertIn("overflow-wrap:anywhere", html)
        self.assertIn("Plafond updates PPO", html)
        self.assertIn("Épisodes terminés", html)
        self.assertIn("Motif d’arrêt", html)
        self.assertIn("Diagnostic mis à jour · entraînement non éligible", html)
        self.assertIn("Modèle mis à jour en diagnostic (non qualifié)", html)
        self.assertIn("Plafond transitions", html)
        self.assertIn("Transitions observées", html)
        self.assertIn("Évaluations publiées", html)
        self.assertIn("showTaskContext(m?.task_context)", html)
        self.assertIn("Objectif d’exercice diagnostic", html)
        self.assertIn("Plafond de décisions collectées", html)
        self.assertIn("Plafond de tours collectés", html)
        self.assertIn("ils ne sont pas des échéances de victoire", html)
        self.assertIn("Atteindre l’objectif d’exercice ne signifie pas une victoire dans Civilization VI", html)
        self.assertIn("SHA-256 différent du scénario déclaré dans le manifeste", html)
        self.assertIn("Critiqueur limité aux observations joueur", html)
        self.assertIn("Critiqueur privilégié en diagnostic", html)
        self.assertIn("Mode critiqueur non précisé (privilège non confirmé)", html)
        self.assertIn("Temps écoulé (s)", html)
        self.assertIn("r.wall_seconds-minTime", html)
        self.assertNotIn("Dernière étape mesurée", html)
        self.assertIn("r.stage==='ppo_update'", html)


if __name__ == "__main__":
    unittest.main()
