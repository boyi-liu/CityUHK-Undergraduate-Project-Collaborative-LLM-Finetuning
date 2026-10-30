"""Compare one reference algorithm under two availability conditions."""
from __future__ import annotations

import copy
from pathlib import Path

CONDITIONS = ('always', 'trace')


def comparison_configs(config):
    from .mmlu_config import validate
    for condition in CONDITIONS:
        candidate = copy.deepcopy(config)
        candidate['sessions']['availability'] = condition
        candidate['output_dir'] = str(Path(config['output_dir']) / 'comparison' / condition)
        yield validate(candidate)


def run_comparison(config, root, resume=False, simulation_only=False, runtime_factory=None):
    from .trace_baselines import run_sessions, schedule
    from .trace_plotting import plot_comparison
    candidates = list(comparison_configs(config))
    if simulation_only:
        # Schedule counts never become measured accuracy points.
        return {c['sessions']['availability']: schedule(c, root, simulate=True) for c in candidates}
    for candidate in candidates:
        path = Path(root) / candidate['output_dir'] / 'checkpoint.pt'
        if path.exists() and not resume:
            raise FileExistsError(f'{path} exists; use --resume or a new output_dir')
    plot_comparison(config, root)  # Check plotting and existing identities before GPU work.
    for candidate in candidates:
        path = Path(root) / candidate['output_dir'] / 'checkpoint.pt'
        run_sessions(candidate, root, resume=resume and path.exists(), runtime_factory=runtime_factory,
                     on_report=lambda: plot_comparison(config, root))
    return plot_comparison(config, root)
