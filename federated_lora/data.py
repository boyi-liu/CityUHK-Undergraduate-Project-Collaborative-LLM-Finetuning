"""Assign every selected Alpaca row to a non-IID training client."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np


ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
PREPARED_FORMAT_VERSION = 3


def load_task_groups(label_path: Path, manifest_path: Path, raw_bytes: bytes,
                     raw_count: int) -> tuple[dict[int, str], str]:
    """Reject labels generated for another Alpaca file or with missing rows."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError("Task-group labels were generated for a different Alpaca file")
    if manifest["source_rows"] != raw_count:
        raise ValueError("Task-group manifest row count differs from the Alpaca file")
    label_bytes = label_path.read_bytes()
    labels: dict[int, str] = {}
    allowed = set(manifest["labels"])
    for line in label_bytes.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index, group = row["source_index"], row["task_group"]
        if type(index) is not int or not 0 <= index < raw_count or index in labels:
            raise ValueError(f"Duplicate or invalid task-group source_index: {index}")
        if group not in allowed:
            raise ValueError(f"Invalid task group for row {index}: {group}")
        labels[index] = group
    if len(labels) != raw_count or dict(Counter(labels.values())) != manifest["counts"]:
        raise ValueError("Task-group labels are incomplete or differ from their manifest")
    return labels, hashlib.sha256(label_bytes).hexdigest()


def partition_dirichlet(records: list[dict], client_count: int, alpha: float,
                        min_samples: int, seed: int, attempts: int = 1000) -> list[list[dict]]:
    if client_count < 2 or alpha <= 0 or min_samples < 1:
        raise ValueError("client_count >= 2, alpha > 0 and min_samples >= 1 are required")
    if len(records) < client_count * min_samples:
        raise ValueError("Too few training examples for the requested clients and minimum size")
    rng = np.random.default_rng(seed)
    labels = np.array([r["task_group"] for r in records])
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


def split_records(raw: list[dict], *, task_groups: dict[int, str], client_count: int,
                  max_examples: int | None, alpha: float, min_samples: int,
                  seed: int) -> list[list[dict]]:
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive or null for all rows")
    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(raw))
    if max_examples is not None:
        selected = selected[:max_examples]
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
            "task_group": task_groups[int(index)],
        })
    clients = partition_dirichlet(records, client_count, alpha, min_samples, seed + 1)
    ids = [r["source_index"] for group in clients for r in group]
    if len(ids) != len(records) or len(ids) != len(set(ids)):
        raise AssertionError("Client training partitions lost or duplicated rows")
    return clients


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
    data = config["data"]
    labels, label_sha256 = load_task_groups(
        root / data["task_groups_path"], root / data["task_groups_manifest_path"],
        raw_bytes, len(raw),
    )
    clients = split_records(raw, task_groups=labels, **split_config(config))
    for client_id, rows in enumerate(clients):
        _write_jsonl(data_dir / "clients" / f"{client_id}.jsonl", rows)
    manifest = {
        "format_version": PREPARED_FORMAT_VERSION,
        "source": ALPACA_URL,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "task_groups_sha256": label_sha256,
        "split_config": split_config(config),
        "training_count": sum(map(len, clients)),
        "client_counts": [len(rows) for rows in clients],
        "client_task_groups": [dict(Counter(r["task_group"] for r in rows)) for rows in clients],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_prepared(config: dict, root: Path) -> tuple[list[list[dict]], dict]:
    data_dir = root / config["data"]["output_dir"]
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared data missing: run `python -m federated_lora prepare --config ...` first ({data_dir})")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != PREPARED_FORMAT_VERSION:
        raise ValueError("Prepared data format changed; prepare the data again")
    if manifest["split_config"] != split_config(config):
        raise ValueError("Prepared data configuration differs from this experiment; prepare the data again")
    label_path = root / config["data"]["task_groups_path"]
    if manifest.get("task_groups_sha256") != hashlib.sha256(label_path.read_bytes()).hexdigest():
        raise ValueError("Task-group labels changed since preparation; prepare the data again")
    clients = [read_jsonl(data_dir / "clients" / f"{i}.jsonl")
               for i in range(config["federation"]["client_count"])]
    ids = [row["source_index"] for client in clients for row in client]
    if len(ids) != manifest["training_count"] or len(ids) != len(set(ids)):
        raise ValueError("Prepared client files have missing or duplicate training rows")
    return clients, manifest
