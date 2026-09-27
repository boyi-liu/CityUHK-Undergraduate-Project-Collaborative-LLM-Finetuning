import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from federated_lora.data import load_prepared, load_task_groups, prepare, split_records


class FederatedLoraDataTests(unittest.TestCase):
    def test_all_rows_are_assigned_once_reproducibly_and_non_iid(self):
        groups = ("create_content", "transform_content", "explain_describe",
                  "find_fact", "summarize")
        raw = [{"instruction": f"Write item {i}",
                "input": "", "output": f"answer {i}"} for i in range(200)]
        labels = {i: groups[i % len(groups)] for i in range(len(raw))}
        options = dict(task_groups=labels, client_count=5, max_examples=None,
                       alpha=0.3, min_samples=8, seed=42)
        first = split_records(raw, **options)
        second = split_records(raw, **options)
        self.assertEqual(first, second)
        clients = first
        ids = [r["source_index"] for group in clients for r in group]
        self.assertEqual(set(ids), set(range(len(raw))))
        self.assertEqual(len(ids), len(raw))
        self.assertTrue(all(r["task_group"] == labels[r["source_index"]]
                            for group in clients for r in group))
        self.assertGreater(len({tuple(sorted(r["task_group"] for r in group))
                                for group in clients}), 1)

    def test_smoke_limit_uses_all_selected_rows(self):
        raw = [{"instruction": str(i), "input": "", "output": str(i)} for i in range(200)]
        clients = split_records(raw, task_groups={i: "create_content" for i in range(200)},
                                client_count=3, max_examples=150, alpha=0.5,
                                min_samples=5, seed=42)
        ids = [r["source_index"] for client in clients for r in client]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 150)

    def test_rejects_too_few_examples_and_invalid_limit(self):
        raw = [{"instruction": "Write", "input": "", "output": "x"}] * 10
        options = dict(task_groups={i: "create_content" for i in range(10)},
                       client_count=3, max_examples=10, alpha=0.5,
                       min_samples=4, seed=1)
        with self.assertRaises(ValueError):
            split_records(raw, **options)
        options["min_samples"] = 1
        options["max_examples"] = 0
        with self.assertRaises(ValueError):
            split_records(raw, **options)

    def test_prepared_manifest_matches_saved_client_files(self):
        raw = [{"instruction": f"Write item {i}", "input": "", "output": str(i)}
               for i in range(120)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset" / "alpaca_raw" / "alpaca_data.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(raw), encoding="utf-8")
            labels_path = root / "dataset" / "alpaca_task_groups.jsonl"
            labels = [{"source_index": i, "task_group": "find_fact" if i % 2 else "create_content"}
                      for i in range(len(raw))]
            labels_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
            label_manifest = labels_path.with_suffix(".manifest.json")
            label_manifest.write_text(json.dumps({
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_rows": len(raw), "labels": ["create_content", "find_fact"],
                "counts": dict(Counter(row["task_group"] for row in labels)),
            }), encoding="utf-8")
            config = {
                "seed": 42,
                "data": {"raw_path": "dataset/alpaca_raw/alpaca_data.json",
                         "task_groups_path": "dataset/alpaca_task_groups.jsonl",
                         "task_groups_manifest_path": "dataset/alpaca_task_groups.manifest.json",
                         "output_dir": "dataset/alpaca_client_splits_test", "max_examples": 100,
                         "dirichlet_alpha": 0.5, "min_samples_per_client": 5},
                "federation": {"client_count": 3},
            }
            manifest = prepare(config, root)
            clients, loaded = load_prepared(config, root)
            self.assertEqual(manifest, loaded)
            self.assertEqual(sum(map(len, clients)), 100)
            self.assertEqual(manifest["training_count"], 100)
            self.assertEqual(manifest["client_counts"], list(map(len, clients)))
            self.assertFalse((root / config["data"]["output_dir"] / "val.jsonl").exists())
            self.assertFalse((root / config["data"]["output_dir"] / "test.jsonl").exists())
            self.assertEqual(manifest["task_groups_sha256"],
                             hashlib.sha256(labels_path.read_bytes()).hexdigest())
            labels_path.write_text(labels_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Task-group labels changed"):
                load_prepared(config, root)

    def test_rejects_wrong_source_and_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.jsonl"
            manifest = root / "manifest.json"
            labels.write_text('{"source_index": 0, "task_group": "create_content"}\n'
                              '{"source_index": 0, "task_group": "create_content"}\n',
                              encoding="utf-8")
            metadata = {"source_sha256": hashlib.sha256(b"[]").hexdigest(),
                        "source_rows": 2, "labels": ["create_content"],
                        "counts": {"create_content": 2}}
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_task_groups(labels, manifest, b"[]", 2)
            metadata["source_sha256"] = "incorrect"
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different Alpaca file"):
                load_task_groups(labels, manifest, b"[]", 2)


if __name__ == "__main__":
    unittest.main()
