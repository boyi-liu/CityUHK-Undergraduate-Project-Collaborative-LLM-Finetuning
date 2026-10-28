"""Synchronous MMLU rounds with immediate accuracy-drop stopping."""
from __future__ import annotations

import json
import os
from pathlib import Path
import torch
from .mmlu_data import prepare, load_prepared, load_pool, question_key, digest
from .mmlu_runtime import MMLURuntime, weighted_mean, adapter_hash
from .round_selection import AccuracySelection


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run(config, root, resume=False, runtime_factory=MMLURuntime):
    root = Path(root)
    prepared_path = root / config["data"]["output_dir"] / "manifest.json"
    if not prepared_path.exists():
        prepare(config, root)
    clients, manifest = load_prepared(config, root)
    _, _, source = load_pool(config, root)
    directory = root / config["output_dir"]
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError("The MMLU run already exists; use --resume or a new output_dir")
    if resume and not checkpoint.exists():
        raise FileNotFoundError("No MMLU checkpoint is available to resume")
    directory.mkdir(parents=True, exist_ok=True)
    runtime = runtime_factory(config, root, directory, source)
    test_keys = {question_key(row) for row in runtime.test_rows}
    if any(question_key(row) in test_keys for client in clients for row in client):
        raise ValueError("A training question overlaps the MMLU selection split")
    encoded = [runtime.encode(rows) for rows in clients]
    coverage = [{"client": cid, "rows": len(rows), "rows_with_targets": sum(r["answer_tokens"] > 0 for r in rows),
                 "rows_without_targets": sum(r["answer_tokens"] == 0 for r in rows)} for cid, rows in enumerate(encoded)]
    identity = {"configuration": config, "data": manifest, "evaluation": runtime.manifest,
                "initial_adapter_sha256": adapter_hash(runtime.state())}
    fingerprint = digest(identity)
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint differs from the model, data, configuration, or evaluation questions")
        policy = AccuracySelection(**state["selection"])
    else:
        initial = runtime.state()
        runtime.save(directory / "round-0-adapter", initial)
        metrics = runtime.evaluate(initial, 0)
        write_json(directory / "round-0-evaluation.json", metrics)
        policy = AccuracySelection.initial(metrics["correct"], metrics["count"])
        state = {"fingerprint": fingerprint, "adapter": initial, "best_adapter": initial,
                 "selection": policy.to_dict(), "history": [metrics], "finished": False, "stop_reason": None}
        save_checkpoint(checkpoint, state)
    write_json(directory / "run_manifest.json", identity)
    write_json(directory / "token_coverage.json", coverage)
    write_json(directory / "resolved_config.json", config)
    while not policy.stopped and policy.last_round < config["federation"]["rounds"]:
        rnd = policy.last_round + 1
        starting = state["adapter"]
        updates, jobs = [], []
        for cid, rows in enumerate(encoded):
            adapter, job = runtime.train(rows, starting, rnd, cid)
            if job["start_adapter_sha256"] != adapter_hash(starting):
                raise RuntimeError("A client did not start from the current global adapter")
            runtime.save(directory / "clients" / f"round-{rnd}" / f"client-{cid}", adapter)
            write_json(directory / f"client-round-{rnd}-{cid}.json", job)
            updates.append(adapter)
            jobs.append(job)
        current = weighted_mean(updates, [len(rows) for rows in clients])
        runtime.save(directory / f"round-{rnd}-adapter", current)
        metrics = runtime.evaluate(current, rnd)
        previous_best = policy.best_round
        decreased = policy.observe(rnd, metrics["correct"], metrics["count"])
        if policy.best_round != previous_best:
            state["best_adapter"] = {name: tensor.clone() for name, tensor in current.items()}
        event = {**metrics, "decreased_from_previous": decreased, "selected_round": policy.best_round,
                 "global_adapter_sha256": adapter_hash(current), "client_start_adapter_sha256": adapter_hash(starting),
                 "training_loss": sum(j["loss_sum"] for j in jobs) / sum(j["target_tokens"] for j in jobs)}
        state["adapter"] = current
        state["selection"] = policy.to_dict()
        state["history"].append(event)
        write_json(directory / f"round-{rnd}-evaluation.json", event)
        state["finished"] = decreased or rnd == config["federation"]["rounds"]
        state["stop_reason"] = "first_accuracy_drop" if decreased else "maximum_rounds" if state["finished"] else None
        save_checkpoint(checkpoint, state)
        write_json(directory / "round-history.json", state["history"])
        print(json.dumps({"round": rnd, "correct": metrics["correct"], "count": metrics["count"],
                          "accuracy": metrics["accuracy"], "decreased": decreased, "selected_round": policy.best_round}), flush=True)
    runtime.verify_base()
    runtime.save(directory / "best-adapter", state["best_adapter"])
    selected = next(event for event in state["history"] if event["round"] == policy.best_round)
    result = {"completed": True, "stopped_round": policy.last_round, "stop_reason": state["stop_reason"],
              "selected_round": policy.best_round, "selected_accuracy": selected["accuracy"],
              "selected_correct": selected["correct"], "count": selected["count"],
              "baseline_accuracy": state["history"][0]["accuracy"], "round_history": state["history"],
              "selection_split": "test", "independent_final_test": False, "test_used_for_checkpoint_selection": True,
              "scope": runtime.manifest["scope"], "data": manifest, "token_coverage": coverage,
              "best_adapter_sha256": adapter_hash(state["best_adapter"]), "frozen_base_verified": True}
    write_json(directory / "summary.json", result)
    return result
