"""Check a completed baseline run and score its saved best adapter again."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from peft import PeftModel
from safetensors.torch import load_file

from .baseline_runtime import BaselineRuntime
from .baselines import factor_pairs, rank
from .mmlu_data import load_pool
from .mmlu_runner import run, write_json
from .mmlu_runtime import adapter_hash


def audit(config, root):
    root = Path(root)
    directory = root / config['output_dir']
    summary = json.loads((directory / 'summary.json').read_text())
    algorithm = config['federation']['algorithm']
    counts = summary['data']['client_counts']
    weights = [n / sum(counts) for n in counts]
    scale = config['model']['lora_alpha'] / config['model']['lora_rank']
    previous = load_file(directory / 'round-0-adapter/adapter_model.safetensors')
    fixed_A = {a: previous[a].clone() for a, _ in factor_pairs(previous)}
    checks = []
    for rnd in range(1, summary['stopped_round'] + 1):
        payloads = [torch.load(directory / f'clients/round-{rnd}/client-{cid}/upload.pt',
                               map_location='cpu', weights_only=True) for cid in range(len(counts))]
        current = load_file(directory / f'round-{rnd}-adapter/adapter_model.safetensors')
        largest_error = 0.
        for a, b in factor_pairs(current):
            if algorithm == 'ffa_lora':
                if not torch.equal(current[a], fixed_A[a]) or any(set(p) != {k for _, k in factor_pairs(current)} for p in payloads):
                    raise AssertionError('FFA-LoRA changed A or uploaded A')
                expected = sum(w * p[b] for w, p in zip(weights, payloads)) @ fixed_A[a] * scale
            else:
                old = previous[b] @ previous[a] * scale
                expected = old + sum(w * (p[b] @ p[a]) * scale for w, p in zip(weights, payloads))
            actual = current[b] @ current[a] * scale
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-4)
            largest_error = max(largest_error, float((actual - expected).abs().max()))
        checks.append({'round': rnd, 'checkpoint_rank': rank(current),
                       'largest_product_error': largest_error, 'product_check_passed': True})
        previous = current

    # Load the saved PEFT export on a new pinned base. Score it directly, without
    # using restore() to hide an incorrect saved rank or alpha.
    _, _, source = load_pool(config, root)
    runtime = BaselineRuntime(config, root, directory, source)
    runtime.model = PeftModel.from_pretrained(runtime._original_base(), directory / 'best-adapter')
    saved = runtime.state()
    exported = runtime.model.peft_config['default']
    if adapter_hash(saved) != summary['best_adapter_sha256'] or exported.lora_alpha / exported.r != scale:
        raise AssertionError('The exported best adapter differs from the selected model')
    best = summary['selected_round']
    predictions = directory / f'round-{best}-predictions.jsonl'
    before = predictions.read_bytes()
    with patch.object(runtime, 'restore', lambda adapter: None):
        evaluated = runtime.evaluate(saved, best)
    if predictions.read_bytes() != before or evaluated['correct'] != summary['selected_correct']:
        raise AssertionError('The reloaded model changed MMLU predictions or scores')
    runtime.verify_base()
    del runtime
    torch.cuda.empty_cache()
    before_resume = {p.name: p.stat().st_mtime_ns for p in directory.glob('round-*-predictions.jsonl')}
    resumed = run(config, root, resume=True)
    if resumed != summary or before_resume != {p.name: p.stat().st_mtime_ns for p in directory.glob('round-*-predictions.jsonl')}:
        raise AssertionError('Completed resume changed the run or repeated evaluation')
    result = {'algorithm': algorithm, 'round_products': checks,
              'fresh_export_predictions_and_scores_match': True, 'completed_resume_repeated_no_rounds': True,
              'original_frozen_base_verified': True, 'selected_correct': evaluated['correct'],
              'selection_count': evaluated['count'], 'independent_final_test': False}
    write_json(directory / 'baseline-audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    from .config import load_config
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(audit(load_config(root / args.config), root), indent=2))
