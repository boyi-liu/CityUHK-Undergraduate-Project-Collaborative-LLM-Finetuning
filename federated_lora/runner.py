"""Synchronous and event-driven federated LoRA runners on one physical GPU."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import random
from pathlib import Path

import torch
import yaml

from .data import load_prepared
from .training import adapter_state, encode_rows, load_adapter, load_training_model, train_client


def weighted_mean(updates: list[dict]) -> dict[str, torch.Tensor]:
    if not updates or any(u["example_count"] <= 0 for u in updates):
        raise ValueError("Aggregation needs updates with positive example counts")
    keys = set(updates[0]["adapter"])
    if any(set(u["adapter"]) != keys for u in updates):
        raise ValueError("Client adapters have different parameter keys")
    total = sum(u["example_count"] for u in updates)
    return {key: sum((u["adapter"][key].float() * (u["example_count"] / total)
                      for u in updates), torch.zeros_like(updates[0]["adapter"][key], dtype=torch.float32))
            for key in keys}


def interpolate(current: dict[str, torch.Tensor], incoming: dict[str, torch.Tensor],
                mix: float) -> dict[str, torch.Tensor]:
    if not 0 <= mix <= 1 or set(current) != set(incoming):
        raise ValueError("Invalid interpolation weight or adapter keys")
    return {key: current[key].float() * (1 - mix) + incoming[key].float() * mix
            for key in current}


def _save_progress(run_dir: Path, state: dict, config: dict, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metrics_tmp = run_dir / "metrics.jsonl.tmp"
    with metrics_tmp.open("w", encoding="utf-8") as file:
        for event in state["history"]:
            file.write(json.dumps(event, allow_nan=False) + "\n")
    os.replace(metrics_tmp, run_dir / "metrics.jsonl")
    checkpoint_tmp = run_dir / "checkpoint.pt.tmp"
    torch.save(state, checkpoint_tmp)
    os.replace(checkpoint_tmp, run_dir / "checkpoint.pt")


def _configuration_id(config: dict, manifest: dict) -> str:
    payload = json.dumps({"config": config, "format_version": manifest["format_version"],
                          "source_sha256": manifest["source_sha256"],
                          "task_groups_sha256": manifest["task_groups_sha256"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def run(config: dict, root: Path, resume: bool = False) -> dict:
    if config.get("sessions", {}).get("enabled"):
        from .session_runner import run_sessions
        return run_sessions(config, root, resume=resume)
    clients, manifest = load_prepared(config, root)
    config_id = _configuration_id(config, manifest)
    run_dir = root / config["output_dir"] / config["federation"]["mode"]
    checkpoint = run_dir / "checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"Run already exists at {run_dir}; use --resume or a new output_dir")
    if resume and not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint to resume at {checkpoint}")

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    model, tokenizer, device = load_training_model(config)
    if device.type != "cuda":
        raise RuntimeError("Real-model federated LoRA training requires a CUDA GPU")
    encoded_clients = [encode_rows(rows, tokenizer, config["model"]["max_length"]) for rows in clients]
    rng = random.Random(config["seed"])
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config_id"] != config_id:
            raise ValueError("Checkpoint was produced by a different configuration or source dataset")
        rng.setstate(state["rng_state"])
        load_adapter(model, state["adapter"])
    else:
        state = {
            "config_id": config_id,
            "adapter": adapter_state(model),
            "version": 0,
            "completed": 0,
            "launched": 0,
            "virtual_time": 0.0,
            "pending": [],
            "history": [],
            "rng_state": rng.getstate(),
        }
        _save_progress(run_dir, state, config, manifest)
    pad_id = tokenizer.pad_token_id
    fed = config["federation"]

    def train_job(client_id: int) -> dict:
        base_version = state["version"]
        load_adapter(model, state["adapter"])
        job_id = state["launched"]
        trained, loss, seconds = train_client(model, encoded_clients[client_id], config, device,
                                               pad_id, job_id, client_id)
        state["launched"] += 1
        return {
            "job_id": job_id,
            "client_id": client_id,
            "base_version": base_version,
            "adapter": trained,
            "example_count": len(clients[client_id]),
            "train_loss": loss,
            "real_train_seconds": seconds,
        }

    def record(event: dict) -> None:
        state["history"].append(event)
        state["rng_state"] = rng.getstate()
        _save_progress(run_dir, state, config, manifest)
        print(json.dumps(event, allow_nan=False), flush=True)

    if fed["mode"] == "sync":
        while state["completed"] < fed["rounds"]:
            selected = rng.sample(range(len(clients)), fed["clients_per_round"])
            updates = [train_job(client_id) for client_id in selected]
            state["adapter"] = weighted_mean(updates)
            state["version"] += 1
            state["completed"] += 1
            durations = [rng.uniform(fed["duration_min"], fed["duration_max"]) for _ in updates]
            state["virtual_time"] += max(durations)
            record({
                "mode": "sync", "round": state["completed"], "version": state["version"],
                "clients": selected, "base_versions": [u["base_version"] for u in updates],
                "client_losses": [u["train_loss"] for u in updates],
                "real_train_seconds": [u["real_train_seconds"] for u in updates],
                "virtual_time": state["virtual_time"],
            })
    else:
        def launch() -> None:
            busy = {item[2]["client_id"] for item in state["pending"]}
            idle = [i for i in range(len(clients)) if i not in busy]
            if not idle or state["launched"] >= fed["total_updates"]:
                return
            client_id = rng.choice(idle)
            update = train_job(client_id)
            finish = state["virtual_time"] + rng.uniform(fed["duration_min"], fed["duration_max"])
            heapq.heappush(state["pending"], (finish, update["job_id"], update))

        while len(state["pending"]) < fed["max_active"] and state["launched"] < fed["total_updates"]:
            launch()
        if state["pending"]:
            state["rng_state"] = rng.getstate()
            _save_progress(run_dir, state, config, manifest)
        while state["completed"] < fed["total_updates"]:
            finish, _, update = heapq.heappop(state["pending"])
            state["virtual_time"] = finish
            staleness = state["version"] - update["base_version"]
            mix = fed["async_mix"] / (1 + staleness)
            state["adapter"] = interpolate(state["adapter"], update["adapter"], mix)
            state["version"] += 1
            state["completed"] += 1
            launch()
            record({
                "mode": "async", "update": state["completed"], "version": state["version"],
                "client": update["client_id"], "base_version": update["base_version"],
                "staleness": staleness, "mix": mix, "train_loss": update["train_loss"],
                "real_train_seconds": update["real_train_seconds"],
                "virtual_time": state["virtual_time"],
            })

    train_losses = []
    for event in state["history"]:
        if "client_losses" in event:
            train_losses.extend(event["client_losses"])
        else:
            train_losses.append(event["train_loss"])
    final = {
        "mode": fed["mode"], "version": state["version"],
        "completed": state["completed"], "launched": state["launched"],
        "virtual_time": state["virtual_time"],
        "mean_client_train_loss": sum(train_losses) / len(train_losses),
        "gpu": torch.cuda.get_device_name(0),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state["adapter"], run_dir / "adapter.pt")
    (run_dir / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)
    return final
