"""Run matched availability experiments without changing their training settings."""

from __future__ import annotations

import copy
from pathlib import Path

from .sessions import run_directory


CONDITIONS = ("always", "trace")
MODES = ("sync", "async")


def comparison_configs(config: dict):
    if not config.get("sessions", {}).get("enabled"):
        raise ValueError("Availability comparison requires sessions.enabled: true")
    if config["sessions"].get("participation", "repeated") != "repeated":
        raise ValueError("Availability comparison requires repeated participation")
    for condition in CONDITIONS:
        for mode in MODES:
            candidate = copy.deepcopy(config)
            candidate["sessions"]["availability"] = condition
            candidate["federation"]["mode"] = mode
            yield candidate


def run_comparison(config: dict, root: Path, resume=False, simulation_only=False):
    from .session_runner import dry_run, run_sessions
    from .plotting import plot_comparison

    candidates = list(comparison_configs(config))
    # Check every destination before starting expensive training.
    if not simulation_only and not resume:
        for candidate in candidates:
            if (run_directory(candidate, root) / "checkpoint.pt").exists():
                raise FileExistsError("Comparison already has checkpoints; use --resume or a new output_dir")
    # Check plotting dependencies and existing result identities before GPU work.
    plot_comparison(config, root)
    for candidate in candidates:
        if simulation_only:
            dry_run(candidate, root)
        else:
            exists = (run_directory(candidate, root) / "checkpoint.pt").exists()
            run_sessions(candidate, root, resume=resume and exists)
    return plot_comparison(config, root)
