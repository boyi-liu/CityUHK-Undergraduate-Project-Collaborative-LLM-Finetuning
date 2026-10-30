"""Exercise snapshot files and resume with tiny tensors and a stub evaluator.

No model downloads or GPU computation are used. The training dependencies are
needed because the production runner imports the real training module.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from federated_lora.config import load_config
from federated_lora.session_runner import run_sessions
from federated_lora.sessions import run_directory


DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("torch", "peft", "transformers"))


@unittest.skipUnless(DEPENDENCIES, "Requires the training environment (torch, peft, transformers)")
class SessionRunnerTests(unittest.TestCase):
    def test_periodic_files_do_not_overwrite_deadlines_and_completed_resume_does_not_evaluate_again(self):
        import torch

        project = Path(__file__).resolve().parents[1]
        config = load_config(project / "configs/federated-lora-sessions-smoke.yaml", "async", "always")
        config["output_dir"] = "experiment"
        config["sessions"].update(count=1, end_time="22:01")
        config["evaluation"].update(interval_seconds=20, max_samples=4)
        model = SimpleNamespace(adapter=None)
        initial = {"lora_A": torch.zeros(1), "lora_B": torch.zeros(1)}
        evaluation_calls = []

        class Evaluator:
            def __init__(self, *args):
                self.manifest = {"scope": "debug_subset", "evaluated_questions": 4}

            def evaluate(self, model, device, prediction_path):
                evaluation_calls.append(str(prediction_path.name))
                prediction_path.write_text('{"unit_test_fixture": true}\n', encoding="utf-8")
                return {"mmlu_accuracy": 0.5, "count": 4, "scope": "debug_subset"}

        def load_adapter(model, adapter):
            model.adapter = {k: v.clone() for k, v in adapter.items()}

        def train_client(*args):
            return {k: v + 0.01 for k, v in model.adapter.items()}, 1.0, 0.001

        # Keep real source coverage checks; all writes use a temporary output path.
        from federated_lora.sessions import prepare_sessions
        schedule = prepare_sessions(config, project)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch("federated_lora.data.load_prepared", return_value=([[]] * 10, {"fixture": True})),
                  patch("federated_lora.session_runner.prepare_sessions", return_value=schedule),
                  patch("federated_lora.training.load_training_model", return_value=(model, SimpleNamespace(pad_token_id=0), SimpleNamespace(type="cuda"))),
                  patch("federated_lora.training.adapter_state", return_value=initial),
                  patch("federated_lora.training.encode_rows", return_value=[]),
                  patch("federated_lora.training.load_adapter", side_effect=load_adapter),
                  patch("federated_lora.training.train_client", side_effect=train_client),
                  patch("federated_lora.evaluation.MMLUEvaluator", Evaluator),
                  patch("federated_lora.scheduler.job_durations", return_value={
                      "train_us": 1_000_000, "upload_us": 1_000_000, "adapter_bytes": 8,
                      "profile": "unit-test", "step_seconds": 0.05}),
                  patch("torch.cuda.get_device_name", return_value="Unit-test fixture")):
                report = run_sessions(config, root)
                directory = run_directory(config, root)
                self.assertEqual(len(evaluation_calls), 4)
                self.assertEqual([p["time_us"] for p in report["evaluations"]],
                                 [20_000_000, 40_000_000, 60_000_000])
                self.assertTrue((directory / "initial-evaluation.json").exists())
                self.assertTrue((directory / "session-01-evaluation.json").exists())
                self.assertEqual(len(list(directory.glob("session-01-at-*-evaluation.json"))), 2)
                self.assertGreater(report["version"], 10)
                resumed = run_sessions(config, root, resume=True)
                self.assertEqual(resumed["version"], report["version"])
                self.assertEqual(len(evaluation_calls), 4)
                saved = json.loads((directory / "summary.json").read_text())
                self.assertEqual(saved["evaluations"], report["evaluations"])


if __name__ == "__main__":
    unittest.main()
