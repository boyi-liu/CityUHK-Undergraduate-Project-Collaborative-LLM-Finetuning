"""Inherited device latency assumptions, independent of physical GPU elapsed time."""

import math
import random


# Original utils/time_utils.py, commit 9f5a4eb. Measurement setup unverified.
PROFILES = {
    "qwen": {"agx": (4.947, 4.962), "nano": (7.045, 7.087)},
    "tinyllama": {"agx": (8.36, 8.44), "nano": (14.538, 14.562)},
}


def job_durations(config: dict, client_id: int, attempt: int, adapter_bytes: int) -> dict:
    settings = config["timing"]
    name = config["model"]["name"].lower()
    family = next((key for key in PROFILES if key in name), None)
    if family is None:
        raise ValueError("Inherited timing profiles support Qwen and TinyLLaMA only")
    profile = "agx" if client_id in settings["agx_client_ids"] else "nano"
    rng = random.Random(config["seed"] + client_id * 1_000_003 + attempt * 1009)
    per_step = rng.uniform(*PROFILES[family][profile]) * settings.get("speed_scale", 1.0)
    return {"profile": profile, "step_seconds": per_step,
            "train_us": math.ceil(per_step * config["training"]["local_steps"] * 1_000_000),
            "upload_us": math.ceil(adapter_bytes * 8 / settings["upload_mbps"]),
            "adapter_bytes": adapter_bytes}
