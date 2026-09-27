import json
import tempfile
import unittest
from pathlib import Path

from point2.data import load_prepared, prepare, split_records


class Point2DataTests(unittest.TestCase):
    def test_splits_are_disjoint_reproducible_and_non_iid(self):
        verbs = ("Write", "Translate", "Explain", "List", "Summarize")
        raw = [{"instruction": f"{verbs[i % len(verbs)]} item {i}",
                "input": "", "output": f"answer {i}"} for i in range(200)]
        options = dict(client_count=5, max_examples=150, train_ratio=0.8,
                       val_ratio=0.1, test_ratio=0.1, alpha=0.3,
                       min_samples=8, seed=42)
        first = split_records(raw, **options)
        second = split_records(raw, **options)
        self.assertEqual(first, second)
        clients, val, test = first
        self.assertEqual(sum(map(len, clients)), 120)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)
        ids = [r["source_index"] for group in [*clients, val, test] for r in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len({tuple(sorted(r["category"] for r in group))
                                for group in clients}), 1)

    def test_rejects_too_few_examples_and_invalid_ratios(self):
        raw = [{"instruction": "Write", "input": "", "output": "x"}] * 10
        options = dict(client_count=3, max_examples=10, train_ratio=0.8,
                       val_ratio=0.1, test_ratio=0.1, alpha=0.5,
                       min_samples=4, seed=1)
        with self.assertRaises(ValueError):
            split_records(raw, **options)
        options["min_samples"] = 1
        options["test_ratio"] = 0.2
        with self.assertRaises(ValueError):
            split_records(raw, **options)

    def test_prepared_manifest_matches_saved_client_files(self):
        raw = [{"instruction": f"Write item {i}", "input": "", "output": str(i)}
               for i in range(120)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            source = root / "dataset" / "alpaca_raw" / "alpaca_data.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(raw), encoding="utf-8")
            config = {
                "seed": 42,
                "data": {"raw_path": "dataset/alpaca_raw/alpaca_data.json",
                         "output_dir": "dataset/point2_alpaca_test", "max_examples": 100,
                         "train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2,
                         "dirichlet_alpha": 0.5, "min_samples_per_client": 5},
                "federation": {"client_count": 3},
            }
            manifest = prepare(config, root)
            clients, val, test, loaded = load_prepared(config, root)
            self.assertEqual(manifest, loaded)
            self.assertEqual(sum(map(len, clients)), 70)
            self.assertEqual(len(val), 10)
            self.assertEqual(len(test), 20)
            self.assertEqual(manifest["client_counts"], list(map(len, clients)))


if __name__ == "__main__":
    unittest.main()
