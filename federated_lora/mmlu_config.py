"""Validate grouped MMLU experiments and each baseline's own schedule."""
import math
import re
from .mmlu_data import MMLU_REVISION


def validate(config):
    data, model, train, fed, evaluation, stopping = [config[k] for k in ("data", "model", "training", "federation", "evaluation", "stopping")]
    if data["revision"] != MMLU_REVISION or evaluation["revision"] != MMLU_REVISION:
        raise ValueError("Use the pinned MMLU revision for training and selection")
    if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("Pin the model revision to a full commit SHA")
    algorithm = fed.get("algorithm", "fedit")
    if algorithm not in ("fedit", "ffa_lora", "flora", "fedasync_lora"):
        raise ValueError("Unknown grouped MMLU algorithm")
    asynchronous = algorithm == "fedasync_lora"
    trace = config.get("sessions", {}).get("enabled", False)
    if fed["mode"] != ("async" if asynchronous else "sync"):
        raise ValueError("FedAsync-LoRA requires async; the other MMLU baselines require synchronous rounds")
    count_keys = ("client_count",) if trace and asynchronous else ("client_count", "clients_per_round") if trace else (
        ("client_count", "max_updates") if asynchronous else ("client_count", "clients_per_round", "rounds"))
    for key in count_keys:
        if type(fed[key]) is not int or fed[key] < 1:
            raise ValueError("Client and round counts must be positive integers")
    if fed["client_count"] < 2 or (not asynchronous and not 1 <= fed["clients_per_round"] <= fed["client_count"]):
        raise ValueError("Invalid client participation count")
    if not trace and not asynchronous and fed["clients_per_round"] != fed["client_count"]:
        raise ValueError("All MMLU clients must participate in each standalone round")
    if asynchronous:
        validate_async(fed, train, trace=trace)
    if "client_ranks" in fed:
        ranks = fed["client_ranks"]
        if (algorithm != "flora" or not isinstance(ranks, list) or len(ranks) != fed["client_count"]
                or any(type(r) is not int or r < 1 for r in ranks)):
            raise ValueError("Only FLoRA accepts one positive integer rank per client")
    for key in ("local_epochs", "micro_batch_size", "grad_accum"):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError("Epochs and batch settings must be positive integers")
    expected_optimizer = "SGD" if asynchronous else "Adam"
    if "local_steps" in train or train["grad_accum"] != 1 or train["optimizer"] != expected_optimizer:
        raise ValueError(f"This method uses complete local epochs, {expected_optimizer}, and no accumulation")
    for value in (train["learning_rate"], data["dirichlet_alpha"], model["lora_alpha"]):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rate, Dirichlet alpha, and LoRA alpha must be positive")
    if type(model["lora_rank"]) is not int or model["lora_rank"] < 1 or model["max_length"] < 16 or not 0 <= model["lora_dropout"] < 1:
        raise ValueError("Invalid LoRA rank, dropout, or training length")
    if not isinstance(model["target_modules"], list) or not model["target_modules"]:
        raise ValueError("Use explicit LoRA target modules")
    if train["weight_decay"] != 0:
        raise ValueError("This MMLU protocol uses no weight decay")
    if type(evaluation["n_shot"]) is not int or evaluation["n_shot"] != 0:
        raise ValueError("MMLU evaluation must use zero shot")
    if evaluation["max_length"] < 16:
        raise ValueError("Invalid evaluation context limit")
    for value in (data.get("max_examples"), evaluation.get("max_samples")):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("Example limits must be positive integers or null")
    if trace:
        from .trace_baselines import validate_trace
        validate_trace(config)
    elif stopping != {"rule": "first_accuracy_drop", "split": "test", "include_initial": True}:
        raise ValueError("Use the explicit first-drop test-selection rule, including the initial adapter")
    return config


def validate_async(fed, train, trace=False):
    if any(k in fed for k in ("rounds", "clients_per_round", "client_ranks")):
        raise ValueError("FedAsync uses updates and matching ranks, without a round barrier")
    alpha, rho = fed["mixing_alpha"], train["proximal_rho"]
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("FedAsync mixing_alpha must be between zero and one")
    if isinstance(rho, bool) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("The FedAsync local objective requires positive proximal_rho")
    from .fedasync import staleness_weight
    decay = fed["staleness"]
    if decay["function"] not in ("constant", "polynomial", "hinge"):
        raise ValueError("Use a staleness function from the FedAsync paper")
    staleness_weight(0, decay)
    if trace:
        return
    simulation = fed["simulation"]
    if simulation["schedule"] == "uniform_staleness":
        value = simulation["max_staleness"]
        if type(value) is not int or value < 0:
            raise ValueError("max_staleness must be a nonnegative integer")
    elif simulation["schedule"] == "arrivals":
        durations = simulation["client_job_seconds"]
        if (not isinstance(durations, list) or len(durations) != fed["client_count"] or
                any(isinstance(v, bool) or not math.isfinite(v) or v < .000001 for v in durations)):
            raise ValueError("Provide one positive job duration per client, in simulated seconds")
    else:
        raise ValueError("Use uniform_staleness or arrivals for the FedAsync simulation")
