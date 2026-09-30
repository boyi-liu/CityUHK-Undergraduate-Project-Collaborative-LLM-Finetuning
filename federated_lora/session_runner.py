"""Integrate the shared trace scheduler with actual LoRA training and MMLU."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .scheduler import SessionScheduler
from .sessions import prepare_sessions, write_schedule


def _json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def dry_run(config: dict, root: Path) -> dict:
    """No GPU or model downloads. Placeholder adapters produce scheduling metrics only."""
    manifest = prepare_sessions(config, root)
    directory = root / config["output_dir"] / "dry-run" / config["federation"]["mode"]
    write_schedule(manifest, directory)

    def train(client, adapter, job):
        return {"adapter": adapter, "example_count": 1}

    engine = SessionScheduler(config, manifest, config["timing"]["dry_run_adapter_bytes"],
                              None, train, lambda current, updates, mode, stale: None)
    state = engine.run()
    report = {"simulation_only": True, "note": "No model trained or evaluated; adapter size is configured estimate",
              "sessions": state["summaries"]}
    _json(directory / "summary.json", report)
    (directory / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in state["history"]), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_sessions(config: dict, root: Path, resume=False) -> dict:
    import torch

    from .data import load_prepared
    from .evaluation import MMLUEvaluator
    from .runner import interpolate, weighted_mean
    from .training import adapter_state, encode_rows, load_adapter, load_training_model, train_client

    directory = root / config["output_dir"] / config["federation"]["mode"]
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists() != resume:
        raise ValueError("Existing run needs --resume; --resume requires an existing checkpoint")
    clients, data_manifest = load_prepared(config, root)
    manifest = prepare_sessions(config, root)
    # Bind checkpoint to actual prepared client contents, not only partition settings.
    client_hash = hashlib.sha256(json.dumps(clients, sort_keys=True).encode()).hexdigest()
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    model, tokenizer, device = load_training_model(config)
    if device.type != "cuda":
        raise RuntimeError("Session training requires a CUDA GPU; use dry-run for CPU scheduling checks")
    encoded = [encode_rows(rows, tokenizer, config["model"]["max_length"]) for rows in clients]
    evaluator = MMLUEvaluator(config["evaluation"], tokenizer, root)
    initial = adapter_state(model)
    adapter_bytes = sum(t.numel() * t.element_size() for t in initial.values())
    identity = {"config": config, "sessions": manifest, "data": data_manifest,
                "client_content_sha256": client_hash, "evaluation": evaluator.manifest,
                "adapter_bytes": adapter_bytes}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if resume else None
    if saved and saved["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint differs from the configuration, traces, client data or MMLU protocol")
    directory.mkdir(parents=True, exist_ok=True)
    write_schedule(manifest, directory)
    _json(directory / "run_manifest.json", identity)

    def evaluate(adapter, index):
        load_adapter(model, adapter)
        label = "initial" if index < 0 else f"session-{index + 1:02d}"
        adapter_path = directory / f"{label}-adapter.pt"
        temporary = adapter_path.with_suffix(".pt.tmp")
        torch.save(adapter, temporary)
        os.replace(temporary, adapter_path)
        metrics = evaluator.evaluate(model, device, directory / f"{label}-predictions.jsonl")
        _json(directory / f"{label}-evaluation.json", metrics)
        print(json.dumps({"snapshot": label, "scope": metrics["scope"],
                          "mmlu_accuracy": metrics["mmlu_accuracy"], "questions": metrics["count"]}), flush=True)
        return metrics

    initial_evaluation = saved["initial_evaluation"] if saved else evaluate(initial, -1)

    def train(client_id, adapter, job_id):
        load_adapter(model, adapter)
        trained, loss, seconds = train_client(model, encoded[client_id], config, device,
                                              tokenizer.pad_token_id, job_id, client_id)
        return {"adapter": trained, "example_count": len(clients[client_id]),
                "train_loss": loss, "real_train_seconds": seconds}

    def aggregate(adapter, updates, mode, staleness):
        if mode == "sync":
            return weighted_mean(updates)
        mix = config["federation"]["async_mix"] / (1 + staleness)
        return interpolate(adapter, updates[0]["adapter"], mix)

    def save(state):
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save({"fingerprint": fingerprint, "initial_evaluation": initial_evaluation,
                    "state": state}, temporary)
        os.replace(temporary, checkpoint)
        # Rebuild from checkpointed history, so retries do not duplicate log entries.
        temporary = directory / "events.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in state["history"]), encoding="utf-8")
        os.replace(temporary, directory / "events.jsonl")

    engine = SessionScheduler(config, manifest, adapter_bytes, initial, train, aggregate,
                              evaluate=evaluate, save=save, state=saved["state"] if saved else None)
    if not resume:
        save(engine.state)
    state = engine.run()
    torch.save(state["adapter"], directory / "adapter.pt")
    report = {"mode": config["federation"]["mode"], "version": state["version"],
              "scope": evaluator.manifest["scope"], "initial_evaluation": initial_evaluation,
              "sessions": state["summaries"], "gpu": torch.cuda.get_device_name(0)}
    previous = initial_evaluation["mmlu_accuracy"]
    for summary in report["sessions"]:
        accuracy = summary["evaluation"]["mmlu_accuracy"]
        summary["mmlu_change_from_previous_deadline"] = accuracy - previous
        previous = accuracy
    _json(directory / "summary.json", report)
    print(json.dumps({"mode": report["mode"], "version": report["version"],
                      "completed_sessions": len(report["sessions"]), "scope": report["scope"]}), flush=True)
    return report
