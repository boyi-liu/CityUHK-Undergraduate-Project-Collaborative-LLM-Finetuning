"""Arrival-wise FedAsync-LoRA training, selection, and atomic recovery."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import torch

from .async_schedule import AsyncSchedule
from .fedasync import mix
from .fedasync_runtime import FedAsyncRuntime
from .mmlu_data import prepare, load_prepared, load_pool, question_key, digest
from .mmlu_runtime import adapter_hash
from .mmlu_runner import write_json, save_checkpoint


@dataclass
class UpdateSelection:
    count: int
    previous_correct: int
    best_correct: int
    best_update: int = 0
    last_update: int = 0
    stopped: bool = False

    def observe(self, number, correct, count):
        if self.stopped or number != self.last_update + 1:
            raise ValueError('Accuracy selection requires consecutive updates')
        if count != self.count or type(correct) is not int or not 0 <= correct <= count:
            raise ValueError('Every update must score the same questions')
        decreased = correct < self.previous_correct
        if correct > self.best_correct:
            self.best_correct, self.best_update = correct, number
        self.previous_correct, self.last_update, self.stopped = correct, number, decreased
        return decreased


def run(config, root, resume=False, runtime_factory=None):
    from .mmlu_config import validate
    validate(config)
    root = Path(root)
    if not (root / config['data']['output_dir'] / 'manifest.json').exists():
        prepare(config, root)
    clients, manifest = load_prepared(config, root)
    _, _, source = load_pool(config, root)
    directory = root / config['output_dir']
    checkpoint = directory / 'checkpoint.pt'
    if checkpoint.exists() and not resume:
        raise FileExistsError('This FedAsync run exists; use --resume or a new output_dir')
    if resume and not checkpoint.exists():
        raise FileNotFoundError('No FedAsync checkpoint is available to resume')
    directory.mkdir(parents=True, exist_ok=True)
    runtime = (runtime_factory or FedAsyncRuntime)(config, root, directory, source)
    test_keys = {question_key(row) for row in runtime.test_rows}
    if any(question_key(row) in test_keys for rows in clients for row in rows):
        raise ValueError('A training question overlaps the MMLU selection split')
    encoded = [runtime.encode(rows) for rows in clients]
    coverage = [{'client': cid, 'rows': len(rows), 'rows_with_targets': sum(r['answer_tokens'] > 0 for r in rows),
                 'rows_without_targets': sum(r['answer_tokens'] == 0 for r in rows)} for cid, rows in enumerate(encoded)]
    initial = runtime.state()
    identity = {'configuration': config, 'data': manifest, 'evaluation': runtime.manifest,
                'initial_adapter_sha256': adapter_hash(initial)}
    fingerprint = digest(identity)
    if resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('format_version') != 1 or state['fingerprint'] != fingerprint:
            raise ValueError('FedAsync checkpoint differs from the model, data, protocol, or configuration')
        policy = UpdateSelection(**state['selection'])
        schedule = AsyncSchedule(config, initial, state['scheduler'])
    else:
        runtime.save(directory / 'update-0-adapter', initial)
        metric = runtime.evaluate(initial, 0)
        count, correct = metric['count'], metric['correct']
        if type(count) is not int or type(correct) is not int or not 0 <= correct <= count or count < 1:
            raise ValueError('Invalid initial evaluation counts')
        policy = UpdateSelection(count, correct, correct)
        schedule = AsyncSchedule(config, initial)
        state = {'format_version': 1, 'fingerprint': fingerprint, 'adapter': initial, 'best_adapter': initial,
                 'selection': asdict(policy), 'scheduler': schedule.checkpoint(), 'history': [metric],
                 'finished': False, 'stop_reason': None}
        write_json(directory / 'update-0-evaluation.json', metric)
        save_checkpoint(checkpoint, state)
    write_json(directory / 'run_manifest.json', identity)
    write_json(directory / 'resolved_config.json', config)
    write_json(directory / 'token_coverage.json', coverage)
    fed = config['federation']
    while not policy.stopped and policy.last_update < fed['max_updates']:
        version = policy.last_update
        job = schedule.next_job(version)
        downloaded_hash = adapter_hash(job['adapter'])
        local, training = runtime.train(encoded[job['client']], job['adapter'], job['job_id'] + 1, job['client'])
        if training['start_adapter_sha256'] != downloaded_hash:
            raise RuntimeError('A FedAsync client trained from the wrong download')
        current, mixing = mix(state['adapter'], local, version, job['downloaded_version'],
                              fed['mixing_alpha'], fed['staleness'])
        number = version + 1
        job_path = directory / 'jobs' / f"job-{job['job_id']}"
        job_path.mkdir(parents=True, exist_ok=True)
        save_checkpoint(job_path / 'upload.pt', local)
        details = {k: v for k, v in job.items() if k != 'adapter'}
        details.update(training)
        details.update({'accepted_update': number, **mixing, 'downloaded_adapter_sha256': downloaded_hash,
                        'upload_bytes': sum(v.numel() * v.element_size() for v in local.values())})
        write_json(job_path / 'training.json', details)
        runtime.save(directory / f'update-{number}-adapter', current)
        metrics = runtime.evaluate(current, number)
        old_best = policy.best_update
        decreased = policy.observe(number, metrics['correct'], metrics['count'])
        if old_best != policy.best_update:
            state['best_adapter'] = {k: v.clone() for k, v in current.items()}
        event = {**metrics, **mixing, 'job_id': job['job_id'], 'client': job['client'],
                 'dispatched_at_us': job['dispatched_at_us'], 'arrives_at_us': job['arrives_at_us'],
                 'downloaded_adapter_sha256': downloaded_hash, 'global_adapter_sha256': adapter_hash(current),
                 'decreased_from_previous': decreased, 'selected_update': policy.best_update,
                 'training_loss': training['training_loss'], 'successful_steps': training['successful_steps']}
        schedule.complete(job, current, number)
        state.update(adapter=current, selection=asdict(policy), scheduler=schedule.checkpoint())
        state['history'].append(event)
        state['finished'] = decreased or number == fed['max_updates']
        state['stop_reason'] = 'first_accuracy_drop' if decreased else 'maximum_updates' if state['finished'] else None
        write_json(directory / f'update-{number}-evaluation.json', event)
        save_checkpoint(checkpoint, state)
        write_json(directory / 'update-history.json', state['history'])
        print(json.dumps({k: event[k] for k in ('update','client','staleness','mixing_coefficient','correct','count','accuracy','decreased_from_previous','selected_update')}), flush=True)
    runtime.verify_base()
    runtime.save(directory / 'best-adapter', state['best_adapter'])
    selected = next(e for e in state['history'] if e['update'] == policy.best_update)
    result = {'algorithm': 'fedasync_lora', 'completed': True, 'selected_update': policy.best_update,
              'selected_correct': selected['correct'], 'selected_accuracy': selected['accuracy'], 'count': selected['count'],
              'baseline_accuracy': state['history'][0]['accuracy'], 'stopped_update': policy.last_update,
              'stop_reason': state['stop_reason'], 'update_history': state['history'], 'data': manifest,
              'selection_split': 'test', 'independent_final_test': False, 'test_used_for_checkpoint_selection': True,
              'scope': runtime.manifest['scope'], 'token_coverage': coverage, 'frozen_base_verified': True,
              'best_adapter_sha256': adapter_hash(state['best_adapter']), 'schedule': fed['simulation'],
              'pending_jobs_not_applied': len(schedule.state['queue']),
              'simulated_elapsed_us': schedule.state['now_us'] if schedule.kind == 'arrivals' else None}
    write_json(directory / 'summary.json', result)
    return result
