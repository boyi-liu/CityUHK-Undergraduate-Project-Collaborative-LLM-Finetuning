"""Load and validate federated LoRA experiment configurations."""

from pathlib import Path
import math
import re

import yaml


def load_config(path: Path, mode_override: str | None = None, availability_override: str | None = None) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if mode_override:
        config["federation"]["mode"] = mode_override
    if config.get("data", {}).get("dataset") == "mmlu":
        if availability_override:
            if not config.get("sessions", {}).get("enabled"):
                raise ValueError("Availability override requires sessions.enabled: true")
            config["sessions"]["availability"] = availability_override
        from .mmlu_config import validate
        return validate(config)
    if availability_override:
        if not config.get("sessions", {}).get("enabled"):
            raise ValueError("Availability override requires sessions.enabled: true")
        config["sessions"]["availability"] = availability_override
    fed, train, model = config["federation"], config["training"], config["model"]
    if fed.get("algorithm") in ("ffa_lora", "flora", "fedasync_lora"):
        raise ValueError("Use the grouped MMLU round runner for FFA-LoRA/FLoRA, or its async runner for FedAsync-LoRA")
    if fed["mode"] not in ("sync", "async"):
        raise ValueError("federation.mode must be sync or async")
    if fed["client_count"] < 2 or not 1 <= fed["clients_per_round"] <= fed["client_count"]:
        raise ValueError("Invalid client count or clients_per_round")
    if not 1 <= fed["max_active"] <= fed["client_count"]:
        raise ValueError("Invalid max_active")
    if not 0 < fed["async_mix"] <= 1:
        raise ValueError("Invalid async mix")
    if not config.get("sessions", {}).get("enabled"):
        if fed["rounds"] < 1 or fed["total_updates"] < 1:
            raise ValueError("Rounds and total_updates must be positive")
        if fed["duration_min"] <= 0 or fed["duration_max"] < fed["duration_min"]:
            raise ValueError("Invalid synthetic duration range")
    if "local_steps" in train:
        raise ValueError("Replace training.local_steps with training.local_epochs")
    if any(type(train.get(k)) is not int or train[k] < 1
           for k in ("local_epochs", "micro_batch_size", "grad_accum")):
        raise ValueError("Local epochs, batch size and accumulation must be positive integers")
    targets = model.get("target_modules")
    if targets != "all-linear" and not (isinstance(targets, list) and targets
                                        and all(isinstance(t, str) and t for t in targets)):
        raise ValueError("model.target_modules must be all-linear or a list of module names")
    if train["learning_rate"] <= 0 or model["lora_rank"] < 1 or model["max_length"] < 16:
        raise ValueError("Invalid learning rate, LoRA rank or maximum sequence length")
    if config.get("sessions", {}).get("enabled"):
        sessions, timing, evaluation = config["sessions"], config["timing"], config["evaluation"]
        sessions.setdefault("availability", "trace")
        sessions.setdefault("participation", "repeated")
        evaluation.setdefault("interval_seconds", 3600)
        if sessions["availability"] not in ("trace", "always"):
            raise ValueError("sessions.availability must be trace or always")
        if sessions["participation"] not in ("repeated", "once_per_window"):
            raise ValueError("sessions.participation must be repeated or once_per_window")
        interval = evaluation["interval_seconds"]
        if interval is not None and (isinstance(interval, bool) or not isinstance(interval, (int, float))
                                     or not math.isfinite(interval) or interval < 0.000001):
            raise ValueError("evaluation.interval_seconds must be finite and positive, or null")
        ids = [str(x) for x in sessions["trace_ids"]]
        if len(ids) != fed["client_count"] or len(set(ids)) != len(ids):
            raise ValueError("sessions.trace_ids must map each client to a distinct device")
        if type(sessions["count"]) is not int or sessions["count"] < 1:
            raise ValueError("sessions.count must be a positive integer")
        if not math.isfinite(sessions.get("min_window_seconds", 60)) or sessions.get("min_window_seconds", 60) < 0:
            raise ValueError("Invalid minimum availability window")
        for key in ("upload_mbps", "speed_scale", "dry_run_adapter_bytes"):
            if not math.isfinite(timing[key]) or timing[key] <= 0:
                raise ValueError(f"timing.{key} must be finite and positive")
        if any(type(i) is not int or not 0 <= i < fed["client_count"] for i in timing["agx_client_ids"]):
            raise ValueError("Invalid AGX client ID")
        if not re.fullmatch(r"[0-9a-f]{40}", evaluation["revision"]):
            raise ValueError("Pin evaluation.revision to a full dataset commit SHA")
        limit = evaluation["max_samples"]
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("evaluation.max_samples must be positive or null for full MMLU")
        if type(evaluation["n_shot"]) is not int or evaluation["n_shot"] != 0:
            raise ValueError("New MMLU runs require zero-shot evaluation (n_shot: 0)")
        if evaluation["max_length"] < 16:
            raise ValueError("Invalid MMLU context limit")
    return config
