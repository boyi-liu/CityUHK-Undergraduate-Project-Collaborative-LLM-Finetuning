"""Data integrity, stopping boundaries, and complete round orchestration."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import torch
from federated_lora.config import load_config
from federated_lora.mmlu_data import load_pool, prepare, load_prepared, clean_pool, load_groups, MMLU_REVISION
from federated_lora.mmlu_runner import run
from federated_lora.mmlu_runtime import adapter_hash, encode_training, weighted_mean
from federated_lora.round_selection import AccuracySelection

PROJECT = Path(__file__).resolve().parents[1]


class FakeRuntime:
    scores = [40, 50, 60, 55]
    instances = []

    def __init__(self, config, root, directory, source):
        self.manifest = {"scope": "debug_subset", "count": 100}
        self.test_rows = []
        self.adapter = {"A": torch.tensor([0.]), "B": torch.tensor([0.])}
        self.calls, self.starts = [], []
        self.__class__.instances.append(self)

    def state(self):
        return {k: v.clone() for k, v in self.adapter.items()}

    def encode(self, rows):
        return [{"answer_tokens": 1} for _ in rows]

    def save(self, path, adapter):
        self.adapter = {k: v.clone() for k, v in adapter.items()}
        Path(path).mkdir(parents=True, exist_ok=True)

    def evaluate(self, adapter, rnd):
        self.calls.append(rnd)
        return {"round": rnd, "correct": self.scores[rnd], "count": 100, "accuracy": self.scores[rnd] / 100, "scope": "debug_subset"}

    def train(self, rows, adapter, rnd, cid):
        before = adapter_hash(adapter)
        self.starts.append((rnd, cid, before))
        update = {k: v + cid + 1 for k, v in adapter.items()}
        return update, {"start_adapter_sha256": before, "loss_sum": len(rows), "target_tokens": len(rows)}

    def verify_base(self):
        return True


class MMLUTests(unittest.TestCase):
    def config(self, root):
        cfg = load_config(PROJECT / "configs/federated-lora-full.yaml")
        for key in ("training_path", "manifest_path", "subject_groups_path"):
            cfg["data"][key] = str(PROJECT / cfg["data"][key])
        cfg["data"]["max_examples"] = 120
        cfg["data"]["output_dir"] = "prepared"
        cfg["output_dir"] = "run"
        return cfg

    def test_committed_data_groups_and_reproducible_clients(self):
        cfg = load_config(PROJECT / "configs/federated-lora-full.yaml")
        rows, groups, source = load_pool(cfg, PROJECT)
        self.assertEqual(len(rows), 1780)
        self.assertEqual(len(groups), 8)
        self.assertEqual(sum(map(len, groups.values())), 57)
        self.assertEqual(source["source_counts"], {"dev": 285, "validation": 1531})
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp))
            cfg["data"]["max_examples"] = None
            first = prepare(cfg, Path(tmp))
            self.assertEqual(first, prepare(cfg, Path(tmp)))
            self.assertEqual(first["client_counts"], [488, 304, 247, 232, 509])
            clients, _ = load_prepared(cfg, Path(tmp))
            self.assertEqual(len({r["id"] for c in clients for r in c}), 1780)
            path = Path(tmp) / "prepared/clients/0.jsonl"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "client file changed"):
                load_prepared(cfg, Path(tmp))

    def test_json_lines_preserve_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group_path = root / "groups.json"
            group_path.write_bytes((PROJECT / "dataset/mmlu_subject_groups.json").read_bytes())
            rows = [{"id": "dev:0", "source_split": "dev", "source_index": 0, "subject": "abstract_algebra",
                     "major_group": "Mathematics", "question": "first\u2028second\u0085third", "choices": ["a", "b", "c", "d"], "answer": 0}]
            source = root / "rows.jsonl"
            source.write_bytes((json.dumps(rows[0], ensure_ascii=False) + "\n").encode())
            manifest = {"revision": MMLU_REVISION, "training_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "subject_groups_sha256": hashlib.sha256(group_path.read_bytes()).hexdigest(),
                        "training_count": 1, "group_counts": {"Mathematics": 1}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            cfg = {"data": {"training_path": "rows.jsonl", "manifest_path": "manifest.json", "subject_groups_path": "groups.json", "revision": MMLU_REVISION}}
            self.assertEqual(load_pool(cfg, root)[0], rows)

    def test_pool_deduplicates_test_and_training_rows(self):
        groups = load_groups(PROJECT / "dataset/mmlu_subject_groups.json")
        row = lambda text, answer=0: {"question": text, "choices": ["a", "b", "c", "d"], "answer": answer, "subject": "abstract_algebra"}
        rows, removed = clean_pool([row("a"), row("a")], [row("b"), row("test"), row("a")], [row("test")], groups)
        self.assertEqual([r["id"] for r in rows], ["dev:0", "validation:0"])
        self.assertEqual([r["reason"] for r in removed], ["pool_duplicate", "test_duplicate", "pool_duplicate"])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            clean_pool([row("a")], [row("a", 1)], [], groups)

    def test_strict_drop_and_earliest_tie(self):
        rule = AccuracySelection.initial(40, 100)
        self.assertFalse(rule.observe(1, 50, 100))
        self.assertFalse(rule.observe(2, 50, 100))
        self.assertEqual(rule.best_round, 1)
        self.assertTrue(rule.observe(3, 49, 100))
        self.assertEqual(rule.best_round, 1)
        with self.assertRaises(ValueError):
            rule.observe(4, 70, 100)
        initial_wins = AccuracySelection.initial(40, 100)
        self.assertTrue(initial_wins.observe(1, 39, 100))
        self.assertEqual(initial_wins.best_round, 0)
        with self.assertRaisesRegex(ValueError, "same questions"):
            AccuracySelection.initial(40, 100).observe(1, 41, 101)

    def test_stop_saves_best_and_resume_does_not_train_again(self):
        FakeRuntime.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root)
            result = run(cfg, root, runtime_factory=FakeRuntime)
            self.assertEqual((result["stopped_round"], result["selected_round"]), (3, 2))
            self.assertEqual(FakeRuntime.instances[0].calls, [0, 1, 2, 3])
            starts = FakeRuntime.instances[0].starts
            self.assertEqual(len(starts), 15)
            for rnd in (1, 2, 3):
                self.assertEqual(len({h for r, _, h in starts if r == rnd}), 1)
            self.assertTrue(result["test_used_for_checkpoint_selection"])
            self.assertFalse(result["independent_final_test"])
            resumed = run(cfg, root, resume=True, runtime_factory=FakeRuntime)
            self.assertEqual(result, resumed)
            self.assertEqual(FakeRuntime.instances[-1].calls, [])
            self.assertEqual(FakeRuntime.instances[-1].starts, [])
            original_manifest = (root / "run/run_manifest.json").read_bytes()
            changed = copy.deepcopy(cfg)
            changed["training"]["learning_rate"] *= 2
            with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
                run(changed, root, resume=True, runtime_factory=FakeRuntime)
            self.assertEqual((root / "run/run_manifest.json").read_bytes(), original_manifest)

    def test_epoch_cap_without_accuracy_decrease(self):
        class Increasing(FakeRuntime):
            scores = [40, 41, 41]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp))
            cfg["federation"]["rounds"] = 2
            result = run(cfg, Path(tmp), runtime_factory=Increasing)
            self.assertEqual((result["stopped_round"], result["selected_round"]), (2, 1))
            self.assertEqual(result["stop_reason"], "maximum_rounds")

    def test_training_encoder_masks_prompt_and_reports_truncation(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(map(ord, text))
        row = {"question": "q", "choices": ["a", "b", "c", "d"], "answer": 1}
        short = encode_training(row, Tokenizer(), 128)
        self.assertGreater(short["answer_tokens"], 0)
        long = encode_training({**row, "question": "q" * 200}, Tokenizer(), 128)
        self.assertEqual(long["answer_tokens"], 0)
        self.assertEqual(len(long["input_ids"]), 128)

    def test_sample_weighted_factors_preserve_inputs(self):
        a, b = {"A": torch.tensor([1.]), "B": torch.tensor([2.])}, {"A": torch.tensor([5.]), "B": torch.tensor([6.])}
        before = adapter_hash(a)
        result = weighted_mean([a, b], [1, 3])
        self.assertEqual((result["A"].item(), result["B"].item()), (4., 5.))
        self.assertEqual(adapter_hash(a), before)


if __name__ == "__main__":
    unittest.main()
