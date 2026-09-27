"""Load and validate federated LoRA experiment configurations."""

from pathlib import Path

import yaml


def load_config(path: Path, mode_override: str | None = None) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if mode_override:
        config["federation"]["mode"] = mode_override
    fed, train, model = config["federation"], config["training"], config["model"]
    if fed["mode"] not in ("sync", "async"):
        raise ValueError("federation.mode must be sync or async")
    if fed["client_count"] < 2 or not 1 <= fed["clients_per_round"] <= fed["client_count"]:
        raise ValueError("Invalid client count or clients_per_round")
    if not 1 <= fed["max_active"] <= fed["client_count"]:
        raise ValueError("Invalid max_active")
    if fed["rounds"] < 1 or fed["total_updates"] < 1:
        raise ValueError("Rounds and total_updates must be positive")
    if not 0 < fed["async_mix"] <= 1 or fed["duration_min"] <= 0 or fed["duration_max"] < fed["duration_min"]:
        raise ValueError("Invalid async mix or synthetic duration range")
    if any(train[k] < 1 for k in ("local_steps", "micro_batch_size", "grad_accum")):
        raise ValueError("Training steps, batch size and accumulation must be positive")
    if train["learning_rate"] <= 0 or model["lora_rank"] < 1 or model["max_length"] < 16:
        raise ValueError("Invalid learning rate, LoRA rank or maximum sequence length")
    return config
