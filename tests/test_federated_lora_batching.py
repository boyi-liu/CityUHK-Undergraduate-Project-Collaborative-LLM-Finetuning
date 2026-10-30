"""Check complete epochs, short batches, and the configured work budget."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from federated_lora.batching import epoch_batches, optimizer_steps
from federated_lora.config import load_config


PROJECT = Path(__file__).resolve().parents[1]


class EpochBatchTests(unittest.TestCase):
    settings = {"local_epochs": 1, "micro_batch_size": 1, "grad_accum": 32}

    def test_one_epoch_visits_every_row_once_without_padding(self):
        for size in (1, 12, 31, 32, 33, 64, 1000):
            groups = list(epoch_batches(size, self.settings, 42))
            visited = [i for group in groups for batch in group for i in batch]
            self.assertEqual(sorted(visited), list(range(size)))
            self.assertEqual(len(groups), optimizer_steps(size, self.settings))
            self.assertEqual(sum(len(batch) for batch in groups[-1]), (size - 1) % 32 + 1)

    def test_micro_batch_changes_do_not_change_optimizer_groups(self):
        original = [[i for batch in group for i in batch]
                    for group in epoch_batches(67, self.settings, 42)]
        changed = {**self.settings, "micro_batch_size": 4, "grad_accum": 8}
        actual = [[i for batch in group for i in batch]
                  for group in epoch_batches(67, changed, 42)]
        self.assertEqual(actual, original)
        self.assertEqual(len(list(epoch_batches(67, changed, 42))[-1][-1]), 3)

    def test_each_epoch_is_complete_and_shuffle_is_reproducible(self):
        settings = {**self.settings, "local_epochs": 2}
        groups = list(epoch_batches(67, settings, 42))
        self.assertEqual(groups, list(epoch_batches(67, settings, 42)))
        passes = [[i for group in groups[start:start + 3] for batch in group for i in batch]
                  for start in (0, 3)]
        for order in passes:
            self.assertEqual(sorted(order), list(range(67)))
        self.assertNotEqual(*passes)
        with self.assertRaises(ValueError):
            optimizer_steps(0, settings)

    def test_all_training_configs_use_the_requested_settings(self):
        for path in (PROJECT / "configs").glob("federated-lora-*.yaml"):
            config = load_config(path)
            self.assertEqual(config["training"]["local_epochs"], 1, path)
            if config.get("data", {}).get("dataset") == "mmlu":
                self.assertEqual(config["training"]["micro_batch_size"] * config["training"]["grad_accum"], 8, path)
                self.assertEqual(config["model"]["target_modules"], ["q_proj", "v_proj"], path)
            else:
                self.assertEqual(config["training"]["micro_batch_size"] * config["training"]["grad_accum"], 32, path)
                self.assertEqual(config["model"]["target_modules"], "all-linear", path)
            self.assertNotIn("local_steps", config["training"])

    def test_invalid_epoch_and_legacy_step_settings_are_rejected(self):
        base = load_config(PROJECT / "configs/federated-lora-full.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for key, value in (("local_epochs", 0), ("local_epochs", 1.5),
                               ("local_epochs", True), ("grad_accum", 0), ("local_steps", 20)):
                config = copy.deepcopy(base)
                config["training"][key] = value
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
