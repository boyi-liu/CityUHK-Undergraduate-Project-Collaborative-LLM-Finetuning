"""Prepare disjoint Alpaca splits with auditable non-IID client partitions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np


ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def category(instruction: str) -> str:
    text = instruction.strip().lower()
    groups = {
        "writing": ("write", "create", "generate", "compose", "draft"),
        "transform": ("translate", "convert", "transform", "rewrite"),
        "list": ("list", "give", "provide", "name", "identify"),
        "explain": ("explain", "describe", "what is", "what are", "define"),
        "question": ("how", "why", "when", "where", "who"),
        "classify": ("classify", "categorize", "determine", "evaluate", "judge"),
        "summarize": ("summarize", "summarise", "condense", "shorten"),
    }
    for label, prefixes in groups.items():
        if text.startswith(prefixes):
            return label
    return "other"


def partition_dirichlet(records: list[dict], client_count: int, alpha: float,
                        min_samples: int, seed: int, attempts: int = 1000) -> list[list[dict]]:
    if client_count < 2 or alpha <= 0 or min_samples < 1:
        raise ValueError("client_count >= 2, alpha > 0 and min_samples >= 1 are required")
    if len(records) < client_count * min_samples:
        raise ValueError("Too few training examples for the requested clients and minimum size")
    rng = np.random.default_rng(seed)
    labels = np.array([r["category"] for r in records])
    for _ in range(attempts):
        buckets: list[list[int]] = [[] for _ in range(client_count)]
        for label in sorted(set(labels)):
            indices = rng.permutation(np.flatnonzero(labels == label))
            proportions = rng.dirichlet(np.full(client_count, alpha))
            cuts = (np.cumsum(proportions[:-1]) * len(indices)).astype(int)
            for client_id, group in enumerate(np.split(indices, cuts)):
                buckets[client_id].extend(int(i) for i in group)
        if min(map(len, buckets)) >= min_samples:
            result = []
            for indices in buckets:
                rng.shuffle(indices)
                result.append([records[i] for i in indices])
            return result
    raise ValueError(f"Could not meet min_samples after {attempts} Dirichlet attempts")


def split_records(raw: list[dict], *, client_count: int, max_examples: int,
                  train_ratio: float, val_ratio: float, test_ratio: float,
                  alpha: float, min_samples: int, seed: int) -> tuple[list[list[dict]], list[dict], list[dict]]:
    if any(x <= 0 for x in (train_ratio, val_ratio, test_ratio)) or not np.isclose(
        train_ratio + val_ratio + test_ratio, 1.0
    ):
        raise ValueError("train, validation and test ratios must be positive and sum to 1")
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(raw))[:min(max_examples, len(raw))]
    records = []
    for index in selected:
        row = raw[int(index)]
        if not all(k in row for k in ("instruction", "input", "output")):
            raise ValueError("Alpaca records need instruction, input and output fields")
        records.append({
            "source_index": int(index),
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"],
            "category": category(row["instruction"]),
        })
    n_train = int(len(records) * train_ratio)
    n_val = int(len(records) * val_ratio)
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    if not val or not test:
        raise ValueError("Validation and test splits must each contain examples")
    clients = partition_dirichlet(train, client_count, alpha, min_samples, seed + 1)
    ids = [r["source_index"] for group in [*clients, val, test] for r in group]
    if len(ids) != len(set(ids)):
        raise AssertionError("Training, validation and test data overlap")
    return clients, val, test


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def split_config(config: dict) -> dict:
    data = config["data"]
    return {
        "client_count": config["federation"]["client_count"],
        "max_examples": data["max_examples"],
        "train_ratio": data["train_ratio"],
        "val_ratio": data["val_ratio"],
        "test_ratio": data["test_ratio"],
        "alpha": data["dirichlet_alpha"],
        "min_samples": data["min_samples_per_client"],
        "seed": config["seed"],
    }


def prepare(config: dict, root: Path) -> dict:
    data_dir = root / config["data"]["output_dir"]
    raw_path = root / config["data"]["raw_path"]
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(ALPACA_URL, timeout=120) as response:
            raw_path.write_bytes(response.read())
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    clients, val, test = split_records(raw, **split_config(config))
    for client_id, rows in enumerate(clients):
        _write_jsonl(data_dir / "clients" / f"{client_id}.jsonl", rows)
    _write_jsonl(data_dir / "val.jsonl", val)
    _write_jsonl(data_dir / "test.jsonl", test)
    manifest = {
        "source": ALPACA_URL,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "split_config": split_config(config),
        "client_counts": [len(rows) for rows in clients],
        "client_categories": [dict(Counter(r["category"] for r in rows)) for rows in clients],
        "validation_count": len(val),
        "test_count": len(test),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_prepared(config: dict, root: Path) -> tuple[list[list[dict]], list[dict], list[dict], dict]:
    data_dir = root / config["data"]["output_dir"]
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared data missing: run `python -m point2 prepare --config ...` first ({data_dir})")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["split_config"] != split_config(config):
        raise ValueError("Prepared data configuration differs from this experiment; prepare the data again")
    clients = [read_jsonl(data_dir / "clients" / f"{i}.jsonl")
               for i in range(config["federation"]["client_count"])]
    return clients, read_jsonl(data_dir / "val.jsonl"), read_jsonl(data_dir / "test.jsonl"), manifest
