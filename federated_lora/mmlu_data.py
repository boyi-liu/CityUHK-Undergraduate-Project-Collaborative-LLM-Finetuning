"""Load the committed MMLU pool and prepare reproducible subject-skewed clients."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def question_key(row):
    return row["question"].strip(), tuple(choice.strip() for choice in row["choices"])


def load_groups(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = document["groups"]
    subjects = [subject for names in groups.values() for subject in names]
    if len(groups) != 8 or len(subjects) != 57 or len(set(subjects)) != 57:
        raise ValueError("The subject map must assign 57 subjects to eight groups exactly once")
    return groups


def clean_pool(dev, validation, test, groups):
    subject_group = {subject: group for group, names in groups.items() for subject in names}
    test_keys = {question_key(row) for row in test}
    seen, rows, removed = {}, [], []
    for split, source in (("dev", dev), ("validation", validation)):
        for index, row in enumerate(source):
            key = question_key(row)
            reason = "test_duplicate" if key in test_keys else "pool_duplicate" if key in seen else None
            if reason:
                if reason == "pool_duplicate" and seen[key]["answer"] != row["answer"]:
                    raise ValueError("Conflicting answers for a repeated training question")
                removed.append({"source_split": split, "source_index": index, "reason": reason})
                continue
            result = {"id": f"{split}:{index}", "source_split": split, "source_index": index,
                      **row, "major_group": subject_group[row["subject"]]}
            seen[key] = result
            rows.append(result)
    return rows, removed


def rebuild(root, cache_dir):
    """Export only dev + validation. The test split stays in the local HF cache."""
    from datasets import load_dataset
    root = Path(root)
    groups_path = root / "dataset/mmlu_subject_groups.json"
    groups = load_groups(groups_path)
    kwargs = {"path": "cais/mmlu", "name": "all", "revision": MMLU_REVISION, "cache_dir": str(cache_dir)}
    dev, val, test = [[dict(row) for row in load_dataset(**kwargs, split=s)] for s in ("dev", "validation", "test")]
    rows, removed = clean_pool(dev, val, test, groups)
    path = root / "dataset/mmlu_train.jsonl"
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_bytes(text.encode("utf-8"))
    manifest = {"format_version": 1, "dataset": "cais/mmlu", "revision": MMLU_REVISION,
                "source": f"https://huggingface.co/datasets/cais/mmlu/tree/{MMLU_REVISION}",
                "license": "MIT", "license_file": "dataset/MMLU-LICENSE.txt",
                "training_splits": ["dev", "validation"], "source_counts": {"dev": len(dev), "validation": len(val)},
                "training_count": len(rows), "removed": removed,
                "training_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "subject_groups_sha256": hashlib.sha256(groups_path.read_bytes()).hexdigest(),
                "test_count": len(test), "test_content_sha256": digest(test),
                "source_content_sha256": {"dev": digest(dev), "validation": digest(val)},
                "group_counts": dict(Counter(row["major_group"] for row in rows)),
                "note": "Project-defined major groups. Exact question/choice duplicates and test overlaps are excluded."}
    (root / "dataset/mmlu_train.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_pool(config, root):
    settings, root = config["data"], Path(root)
    source = root / settings["training_path"]
    groups_path = root / settings["subject_groups_path"]
    manifest = json.loads((root / settings["manifest_path"]).read_text(encoding="utf-8"))
    if manifest["revision"] != settings["revision"] or manifest["revision"] != MMLU_REVISION:
        raise ValueError("Training data revision differs from the pinned MMLU source")
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["training_file_sha256"]:
        raise ValueError("The MMLU training file differs from its manifest")
    if hashlib.sha256(groups_path.read_bytes()).hexdigest() != manifest["subject_groups_sha256"]:
        raise ValueError("The MMLU subject map differs from its manifest")
    groups = load_groups(groups_path)
    mapping = {s: g for g, subjects in groups.items() for s in subjects}
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").split("\n") if line.strip()]
    ids, questions = set(), set()
    for row in rows:
        if (row["source_split"] not in ("dev", "validation") or type(row["source_index"]) is not int
                or row["source_index"] < 0 or row["id"] != f"{row['source_split']}:{row['source_index']}"
                or len(row["choices"]) != 4 or type(row["answer"]) is not int or not 0 <= row["answer"] < 4):
            raise ValueError("Invalid MMLU training record")
        if row["subject"] not in mapping or row["major_group"] != mapping[row["subject"]]:
            raise ValueError("MMLU row has the wrong major group")
        key = question_key(row)
        if row["id"] in ids or key in questions:
            raise ValueError("Duplicate MMLU training row")
        ids.add(row["id"])
        questions.add(key)
    if len(rows) != manifest["training_count"] or dict(Counter(r["major_group"] for r in rows)) != manifest["group_counts"]:
        raise ValueError("MMLU row counts differ from the manifest")
    return rows, groups, manifest


def partition(rows, groups, client_count, alpha, seed):
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(client_count)]
    for group in groups:
        indices = [i for i, row in enumerate(rows) if row["major_group"] == group]
        rng.shuffle(indices)
        counts = rng.multinomial(len(indices), rng.dirichlet(np.full(client_count, alpha)))
        cursor = 0
        for client, count in zip(clients, counts):
            client.extend(indices[cursor:cursor + count])
            cursor += count
    clients = [sorted(client) for client in clients]
    if not all(clients):
        raise ValueError("The partition has an empty client; use more rows or another seed")
    if sorted(i for client in clients for i in client) != list(range(len(rows))):
        raise AssertionError("Client partitions lost or repeated rows")
    return clients


def prepare(config, root):
    rows, groups, source = load_pool(config, root)
    maximum = config["data"].get("max_examples")
    if maximum is not None and maximum < len(rows):
        indices = sorted(np.random.default_rng(config["seed"]).choice(len(rows), maximum, replace=False).tolist())
        rows = [rows[i] for i in indices]
    clients = partition(rows, groups, config["federation"]["client_count"], config["data"]["dirichlet_alpha"], config["seed"])
    directory = Path(root) / config["data"]["output_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    hashes = []
    for cid, indices in enumerate(clients):
        path = directory / "clients" / f"{cid}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in indices).encode("utf-8"))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = {"format_version": 1, "dataset": "cais/mmlu", "revision": source["revision"],
                "source_sha256": source["training_file_sha256"], "subject_groups_sha256": source["subject_groups_sha256"],
                "seed": config["seed"], "alpha": config["data"]["dirichlet_alpha"], "maximum_examples": maximum,
                "training_count": len(rows), "client_counts": list(map(len, clients)), "client_sha256": hashes,
                "client_groups": [dict(Counter(rows[i]["major_group"] for i in indices)) for indices in clients]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_prepared(config, root):
    rows, _, source = load_pool(config, root)
    directory = Path(root) / config["data"]["output_dir"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["source_sha256"] != source["training_file_sha256"]
            or manifest["subject_groups_sha256"] != source["subject_groups_sha256"]
            or manifest["seed"] != config["seed"] or manifest["alpha"] != config["data"]["dirichlet_alpha"]
            or manifest["maximum_examples"] != config["data"].get("max_examples")
            or len(manifest["client_counts"]) != config["federation"]["client_count"]):
        raise ValueError("Prepared MMLU data differ from this configuration; prepare again")
    originals = {row["id"]: row for row in rows}
    clients = []
    for cid, expected in enumerate(manifest["client_sha256"]):
        path = directory / "clients" / f"{cid}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Prepared MMLU client file changed")
        client = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        if len(client) != manifest["client_counts"][cid] or any(originals.get(r["id"]) != r for r in client):
            raise ValueError("Prepared MMLU rows differ from the source")
        clients.append(client)
    ids = [row["id"] for client in clients for row in client]
    if len(ids) != manifest["training_count"] or len(ids) != len(set(ids)):
        raise ValueError("Prepared MMLU clients lost or repeated rows")
    return clients, manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild the pinned grouped MMLU training pool")
    parser.add_argument("--cache-dir", default="dataset/mmlu_raw")
    args = parser.parse_args()
    print(json.dumps(rebuild(Path(__file__).resolve().parents[1], args.cache_dir), indent=2))
