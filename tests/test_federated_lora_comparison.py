"""Comparison contract tests. Scores below are unit-test fixtures, not results."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from federated_lora.comparison import comparison_configs
from federated_lora.config import load_config
from federated_lora.plotting import plot_comparison
from federated_lora.sessions import prepare_sessions, run_directory


PROJECT = Path(__file__).resolve().parents[1]


class ComparisonTests(unittest.TestCase):
    def settings(self):
        return load_config(PROJECT / "configs/federated-lora-sessions.yaml")

    def test_configs_differ_only_in_availability_and_mode(self):
        original = self.settings()
        snapshot = copy.deepcopy(original)
        candidates = list(comparison_configs(original))
        self.assertEqual(len(candidates), 4)
        self.assertEqual(original, snapshot)
        for candidate in candidates:
            self.assertEqual(candidate["training"]["local_epochs"], 1)
            self.assertEqual(candidate["training"]["micro_batch_size"] * candidate["training"]["grad_accum"], 32)
            self.assertEqual(candidate["model"]["target_modules"], "all-linear")
            candidate["sessions"]["availability"] = original["sessions"]["availability"]
            candidate["federation"]["mode"] = original["federation"]["mode"]
            self.assertEqual(candidate, original)

    def test_comparison_preserves_work_budget_and_rejects_once_policy(self):
        candidate = self.settings()
        candidate["training"]["local_epochs"] = 2
        for result in comparison_configs(candidate):
            self.assertEqual(result["training"], candidate["training"])
        candidate["sessions"]["participation"] = "once_per_window"
        with self.assertRaisesRegex(ValueError, "repeated"):
            list(comparison_configs(candidate))

    def test_invalid_intervals_and_availability_fail_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"  # JSON is valid YAML.
            for invalid in (0, -1, float("inf"), True, "hourly", 0.0000001):
                candidate = self.settings()
                candidate["evaluation"]["interval_seconds"] = invalid
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "interval_seconds"):
                    load_config(path)
            path.write_text(json.dumps(self.settings()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "availability"):
                load_config(path, availability_override="invented")


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_config(PROJECT / "configs/federated-lora-sessions.yaml")
        self.config["output_dir"] = "experiment"
        self.config["sessions"]["count"] = 2
        self.config["evaluation"]["max_samples"] = 4
        # Use the real session builder on the recorded traces, but write only to a
        # temporary directory. Synthetic scores are explicitly confined to tests.
        self.manifests = {}
        for candidate in comparison_configs(self.config):
            condition = candidate["sessions"]["availability"]
            if condition not in self.manifests:
                self.manifests[condition] = prepare_sessions(candidate, PROJECT)

    def write_fixture_results(self):
        for candidate in comparison_configs(self.config):
            condition, mode = candidate["sessions"]["availability"], candidate["federation"]["mode"]
            schedule = self.manifests[condition]
            directory = run_directory(candidate, self.root)
            directory.mkdir(parents=True)
            metrics = {"mmlu_accuracy": 0.5 if condition == "always" else 0.25,
                       "count": 4, "scope": "debug_subset"}
            points, sessions = [], []
            for session in schedule["sessions"]:
                for hour in range(1, 10):
                    points.append({"session": session["index"], "kind": "deadline" if hour == 9 else "periodic",
                                   "time_us": session["start_us"] + hour * 3_600_000_000,
                                   "elapsed_session_seconds": (session["index"] * 9 + hour) * 3600,
                                   "version": 1, "evaluation": metrics})
                sessions.append({"index": session["index"], "evaluation": metrics,
                                 "server_updates": 1, "trained_jobs": 1, "completed_uploads": 1,
                                 "canceled_jobs": 0, "incomplete_round_uploads": 0})
            report = {"availability": condition, "mode": mode, "participation": "repeated", "completed": True,
                      "scope": "debug_subset", "initial_evaluation": {**metrics, "mmlu_accuracy": 0.25},
                      "evaluations": points, "sessions": sessions}
            identity = {"scheduler_format_version": 2, "config": candidate, "sessions": schedule,
                        "data": {"fixture": True}, "client_content_sha256": "test-only-client-hash",
                        "initial_adapter_sha256": "test-only-initial-hash", "adapter_bytes": 100,
                        "evaluation": {"scope": "debug_subset", "evaluated_questions": 4}}
            (directory / "summary.json").write_text(json.dumps(report), encoding="utf-8")
            (directory / "run_manifest.json").write_text(json.dumps(identity), encoding="utf-8")

    def plot(self):
        with patch("federated_lora.plotting.prepare_sessions",
                   side_effect=lambda c, r: self.manifests[c["sessions"]["availability"]]):
            return plot_comparison(self.config, self.root)

    def test_empty_panels_contain_no_fabricated_accuracy(self):
        report = self.plot()
        self.assertFalse(report["complete"])
        directory = Path(report["directory"])
        for name in report["plots"]:
            self.assertGreater((directory / f"{name}.png").stat().st_size, 1000)
        self.assertIn("No measured MMLU results yet", (directory / "accuracy-trace.svg").read_text())
        with (directory / "accuracy.csv").open(newline="") as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])

    def test_measured_point_export_and_deadline_gaps(self):
        self.write_fixture_results()
        report = self.plot()
        self.assertTrue(report["complete"])
        directory = Path(report["directory"])
        with (directory / "accuracy.csv").open(newline="") as stream:
            points = list(csv.DictReader(stream))
        self.assertEqual(len(points), 4 * 19)
        self.assertEqual(max(float(p["elapsed_session_hours"]) for p in points), 18)
        self.assertEqual(max(float(p["recorded_elapsed_seconds"]) for p in points), 33 * 3600)
        with (directory / "deadline-comparison.csv").open(newline="") as stream:
            deadlines = list(csv.DictReader(stream))
        self.assertEqual(len(deadlines), 8)
        self.assertTrue(all(float(p["gap_to_always_pp"]) == 25 for p in deadlines if p["availability"] == "trace"))

    def test_mismatched_initial_adapter_is_rejected(self):
        self.write_fixture_results()
        candidate = next(c for c in comparison_configs(self.config) if c["sessions"]["availability"] == "trace")
        path = run_directory(candidate, self.root) / "run_manifest.json"
        identity = json.loads(path.read_text())
        identity["initial_adapter_sha256"] = "different-initial-adapter"
        path.write_text(json.dumps(identity), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "initial adapter"):
            self.plot()

    def test_missing_periodic_point_is_rejected(self):
        self.write_fixture_results()
        candidate = next(comparison_configs(self.config))
        path = run_directory(candidate, self.root) / "summary.json"
        report = json.loads(path.read_text())
        report["evaluations"].pop(0)
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evaluation points"):
            self.plot()


if __name__ == "__main__":
    unittest.main()
