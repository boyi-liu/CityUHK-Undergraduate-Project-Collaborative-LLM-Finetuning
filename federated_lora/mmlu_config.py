"""Validate the active synchronous MMLU experiment."""
import math
import re
from .mmlu_data import MMLU_REVISION


def validate(config):
    data, model, train, fed, evaluation, stopping = [config[k] for k in ("data", "model", "training", "federation", "evaluation", "stopping")]
    if data["revision"] != MMLU_REVISION or evaluation["revision"] != MMLU_REVISION:
        raise ValueError("Use the pinned MMLU revision for training and selection")
    if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("Pin the model revision to a full commit SHA")
    if fed["mode"] != "sync" or config.get("sessions", {}).get("enabled"):
        raise ValueError("The MMLU accuracy stopping rule requires synchronous rounds")
    for key in ("client_count", "clients_per_round", "rounds"):
        if type(fed[key]) is not int or fed[key] < 1:
            raise ValueError("Client and round counts must be positive integers")
    if fed["client_count"] < 2 or fed["clients_per_round"] != fed["client_count"]:
        raise ValueError("All MMLU clients must participate in each round")
    for key in ("local_epochs", "micro_batch_size", "grad_accum"):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError("Epochs and batch settings must be positive integers")
    if "local_steps" in train or train["grad_accum"] != 1 or train["optimizer"] != "Adam":
        raise ValueError("MMLU training uses complete local epochs, Adam, and no accumulation")
    for value in (train["learning_rate"], data["dirichlet_alpha"], model["lora_alpha"]):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rate, Dirichlet alpha, and LoRA alpha must be positive")
    if model["lora_rank"] < 1 or model["max_length"] < 16 or not 0 <= model["lora_dropout"] < 1:
        raise ValueError("Invalid LoRA rank, dropout, or training length")
    if not isinstance(model["target_modules"], list) or not model["target_modules"]:
        raise ValueError("Use explicit LoRA target modules")
    if train["weight_decay"] != 0:
        raise ValueError("This MMLU protocol uses Adam without weight decay")
    if type(evaluation["n_shot"]) is not int or evaluation["n_shot"] != 0:
        raise ValueError("MMLU evaluation must use zero shot")
    if evaluation["max_length"] < 16:
        raise ValueError("Invalid evaluation context limit")
    for value in (data.get("max_examples"), evaluation.get("max_samples")):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("Example limits must be positive integers or null")
    if stopping != {"rule": "first_accuracy_drop", "split": "test", "include_initial": True}:
        raise ValueError("Use the explicit first-drop test-selection rule, including the initial adapter")
    return config
