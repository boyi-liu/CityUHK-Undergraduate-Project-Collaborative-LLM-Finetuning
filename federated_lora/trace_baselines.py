"""Run the four baseline rules on recorded availability and fixed deadlines."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .sessions import prepare_sessions, write_schedule
from .scheduler import SessionScheduler
from .mmlu_data import prepare, load_prepared, load_pool, question_key, digest


def validate_trace(config):
    sessions, fed, timing = config['sessions'], config['federation'], config['timing']
    if sessions.get('availability') not in ('trace', 'always'):
        raise ValueError('Deadline runs require trace or always availability')
    if config['stopping'] != {'rule': 'session_deadline'}:
        raise ValueError('Trace runs stop at session deadlines, without accuracy-based stopping')
    if any(k in fed for k in ('max_updates', 'rounds', 'simulation')):
        raise ValueError('Trace sessions use the trace clock, without an update budget or synthetic schedule')
    if type(fed['max_active']) is not int or not 1 <= fed['max_active'] <= fed['client_count']:
        raise ValueError('Invalid number of active clients')
    ids = list(map(str, sessions['trace_ids']))
    if len(ids) != fed['client_count'] or len(ids) != len(set(ids)):
        raise ValueError('Map every client to one distinct trace')
    if sessions['time_basis'] != 'recorded_local':
        raise ValueError('Use the timestamps as recorded in the trace file')
    if type(sessions['count']) is not int or sessions['count'] < 1:
        raise ValueError('At least one session is required')
    if sessions.get('participation') not in ('repeated', 'once_per_window'):
        raise ValueError('Unknown session participation rule')
    minimum = sessions.get('min_window_seconds', 60)
    if isinstance(minimum, bool) or not math.isfinite(minimum) or minimum < 0:
        raise ValueError('Invalid minimum availability duration')
    for key in ('speed_scale', 'upload_mbps', 'dry_run_adapter_bytes'):
        value = timing[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'timing.{key} must be finite and positive')
    if any(type(cid) is not int or not 0 <= cid < fed['client_count'] for cid in timing['agx_client_ids']):
        raise ValueError('Invalid device profile assignment')
    interval = config['evaluation'].get('interval_seconds')
    if interval is not None and (isinstance(interval, bool) or not math.isfinite(interval) or interval < .000001):
        raise ValueError('Invalid optional reporting interval')


def _clients(config, root):
    if not (Path(root) / config['data']['output_dir'] / 'manifest.json').exists():
        prepare(config, root)
    return load_prepared(config, root)


def schedule(config, root, simulate=False):
    """Prepare trace windows; optionally simulate work without model accuracy."""
    from .mmlu_runner import write_json
    clients, data = _clients(config, root)
    manifest = prepare_sessions(config, root)
    directory = Path(root) / config['output_dir'] / ('dry-run' if simulate else 'schedule')
    write_schedule(manifest, directory)
    if not simulate:
        return {'directory': str(directory), 'sessions': len(manifest['sessions']), 'clients': len(clients)}
    def train(cid, adapter, job_id):
        return {'adapter': None, 'example_count': len(clients[cid])}
    count = config['federation']['client_count']
    ranks = config['federation'].get('client_ranks', [config['model']['lora_rank']] * count)
    sizes = [round(config['timing']['dry_run_adapter_bytes'] * r / config['model']['lora_rank']) for r in ranks]
    engine = SessionScheduler(config, manifest, sizes[0], None, train, lambda *args: None,
                              client_sizes=list(map(len, clients)), adapter_bytes_by_client=sizes)
    state = engine.run()
    result = {'simulation_only': True, 'accuracy_measured': False, 'adapter_bytes_are_estimates': True,
              'algorithm': config['federation']['algorithm'], 'data': data, 'sessions': state['summaries'],
              'note': 'Recorded availability; inherited step-time assumptions; upload-only network cost'}
    write_json(directory / 'summary.json', result)
    (directory / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in state['history']), encoding='utf-8')
    return result


def make_runtime(config, root, directory, source):
    from .mmlu_runtime import MMLURuntime
    algorithm = config['federation']['algorithm']
    if algorithm == 'fedasync_lora':
        from .fedasync_runtime import FedAsyncRuntime
        return FedAsyncRuntime(config, root, directory, source)
    if algorithm in ('ffa_lora', 'flora'):
        from .baseline_runtime import BaselineRuntime
        return BaselineRuntime(config, root, directory, source)
    class TraceFedITRuntime(MMLURuntime):
        def initialize_adapter(self):
            # All trace baselines share PEFT's Kaiming A and zero B.
            pass
    return TraceFedITRuntime(config, root, directory, source)


def run_sessions(config, root, resume=False, runtime_factory=None, max_events=None, on_report=None):
    import torch
    from .baselines import upload, aggregate
    from .fedasync import mix
    from .mmlu_runtime import adapter_hash, weighted_mean
    from .mmlu_runner import write_json, save_checkpoint
    from .mmlu_config import validate
    validate(config)
    root = Path(root)
    directory = root / config['output_dir']
    checkpoint = directory / 'checkpoint.pt'
    if checkpoint.exists() and not resume:
        raise FileExistsError('This trace run exists; use --resume or a new output_dir')
    if resume and not checkpoint.exists():
        raise FileNotFoundError('No trace checkpoint is available to resume')
    clients, data = _clients(config, root)
    _, _, source = load_pool(config, root)
    manifest = prepare_sessions(config, root)
    directory.mkdir(parents=True, exist_ok=True)
    runtime = (runtime_factory or make_runtime)(config, root, directory, source)
    runtime.manifest.update({'algorithm': config['federation']['algorithm'], 'role': 'deadline_report',
                            'independent_final_test': False, 'test_used_for_checkpoint_selection': False,
                            'initialization': 'peft_kaiming_A_zero_B', 'trace_protocol_version': 1})
    test_keys = {question_key(row) for row in runtime.test_rows}
    if any(question_key(row) in test_keys for rows in clients for row in rows):
        raise ValueError('A training question overlaps the MMLU test split')
    encoded = [runtime.encode(rows) for rows in clients]
    coverage = [{'client': cid, 'rows': len(rows), 'rows_with_targets': sum(r['answer_tokens'] > 0 for r in rows)}
                for cid, rows in enumerate(encoded)]
    initial = runtime.state()
    algorithm = config['federation']['algorithm']
    wire = {k: v for k, v in initial.items() if algorithm != 'ffa_lora' or '.lora_B.' in k}
    wire_bytes = sum(v.numel() * v.element_size() for v in wire.values())
    ranks = config['federation'].get('client_ranks', [config['model']['lora_rank']] * len(clients))
    sizes = [round(wire_bytes * r / config['model']['lora_rank']) for r in ranks]
    identity = {'trace_protocol_version': 1, 'configuration': config, 'data': data, 'sessions': manifest,
                'evaluation': runtime.manifest, 'initial_adapter_sha256': adapter_hash(initial),
                'client_upload_bytes': sizes, 'timing_source': 'inherited unverified per-step profiles; upload-only costs'}
    fingerprint = digest(identity)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if resume else None
    if saved and saved['fingerprint'] != fingerprint:
        raise ValueError('Trace checkpoint differs from the model, data, traces, or configuration')
    write_schedule(manifest, directory)
    write_json(directory / 'run_manifest.json', identity)
    write_json(directory / 'resolved_config.json', config)
    write_json(directory / 'token_coverage.json', coverage)

    def evaluate_snapshot(adapter, label, index=None):
        runtime.save(directory / f'{label}-adapter', adapter)
        runtime.prediction_name = lambda number: f'{label}-predictions.jsonl'
        metric = runtime.evaluate(adapter, 0 if index is None else index + 1)
        metric.pop('round', None)
        metric.pop('update', None)
        metric.update({'snapshot': label, 'session': index, 'role': 'deadline_report',
                       'independent_final_test': False, 'test_used_for_checkpoint_selection': False,
                       'adapter_sha256': adapter_hash(adapter)})
        write_json(directory / f'{label}-evaluation.json', metric)
        print(json.dumps({'snapshot': label, 'correct': metric['correct'], 'count': metric['count'],
                          'accuracy': metric['accuracy'], 'scope': metric['scope']}), flush=True)
        return metric

    initial_evaluation = saved['initial_evaluation'] if saved else evaluate_snapshot(initial, 'initial')

    def train(cid, adapter, job_id):
        starting_hash = adapter_hash(adapter)
        local, metrics = runtime.train(encoded[cid], adapter, job_id + 1, cid)
        if metrics['start_adapter_sha256'] != starting_hash:
            raise RuntimeError('A trace client used the wrong global download')
        payload = upload(algorithm, local, adapter) if algorithm in ('ffa_lora', 'flora') else local
        byte_count = sum(v.numel() * v.element_size() for v in payload.values())
        if byte_count != sizes[cid]:
            raise RuntimeError('The simulated upload size differs from the local payload')
        folder = directory / 'jobs' / f'job-{job_id}'
        folder.mkdir(parents=True, exist_ok=True)
        save_checkpoint(folder / 'local-result.pt', payload)
        metadata = {**metrics, 'job_id': job_id, 'client': cid, 'downloaded_version': engine.state['version'],
                    'downloaded_adapter_sha256': starting_hash, 'upload_bytes': byte_count}
        write_json(folder / 'training.json', metadata)
        return {'adapter': payload, 'example_count': len(clients[cid]),
                'train_loss': metrics['training_loss'], 'real_train_seconds': metrics['seconds'],
                'downloaded_adapter_sha256': starting_hash}

    def combine(adapter, jobs, mode, stale):
        version = engine.state['version']
        if algorithm == 'fedasync_lora':
            current, details = mix(adapter, jobs[0]['adapter'], version, jobs[0]['base_version'],
                                   config['federation']['mixing_alpha'], config['federation']['staleness'])
            if details['staleness'] != stale:
                raise RuntimeError('The scheduler and FedAsync staleness differ')
        else:
            counts = [j['example_count'] for j in jobs]
            values = [j['adapter'] for j in jobs]
            current = weighted_mean(values, counts) if algorithm == 'fedit' else aggregate(algorithm, adapter, values, counts)
            details = {'server_version_before': version, 'sample_counts': counts}
        updates = directory / 'server-updates'
        updates.mkdir(exist_ok=True)
        write_json(updates / f'{version + 1}.json', {'version': version + 1, 'algorithm': algorithm,
            'time_us': engine.state['now_us'], 'session': engine.state['session_index'], 'jobs': [j['job_id'] for j in jobs],
            'downloaded_versions': [j['base_version'] for j in jobs], 'before_sha256': adapter_hash(adapter),
            'after_sha256': adapter_hash(current), **details})
        return current

    def evaluate(adapter, index, point=None):
        label = f'session-{index + 1:02d}'
        if point is not None:
            label += f"-at-{point['time_us']:016d}us"
        return evaluate_snapshot(adapter, label, index)

    plotted_evaluations = -1
    base_verified = False

    def write_report(state):
        report = {'algorithm': algorithm, 'mode': config['federation']['mode'], 'completed': state['finished'],
                  'version': state['version'], 'stop_rule': 'session_deadline', 'initial_evaluation': initial_evaluation,
                  'sessions': state['summaries'], 'evaluations': state['evaluations'], 'data': data, 'token_coverage': coverage,
                  'trace_manifest_sha256': digest(manifest), 'scope': runtime.manifest['scope'],
                  'test_used_for_checkpoint_selection': False, 'independent_final_test': False,
                  'frozen_base_verified': base_verified, 'final_adapter_sha256': adapter_hash(state['adapter']),
                  'completed_deadlines': sum(e['kind'] == 'deadline' for e in state['evaluations'])}
        write_json(directory / 'summary.json', report)
        (directory / 'events.jsonl').write_text(''.join(json.dumps(e, allow_nan=False) + '\n' for e in state['history']), encoding='utf-8')
        from .trace_plotting import plot_run
        plot_run(config, root)
        if on_report is not None:
            on_report()
        return report

    def save(state):
        nonlocal plotted_evaluations
        save_checkpoint(checkpoint, {'fingerprint': fingerprint, 'initial_evaluation': initial_evaluation, 'state': state})
        if len(state['evaluations']) != plotted_evaluations:
            write_report(state)
            plotted_evaluations = len(state['evaluations'])

    engine = SessionScheduler(config, manifest, wire_bytes, initial, train, combine, evaluate=evaluate,
                              evaluate_periodic=evaluate, save=save, state=saved['state'] if saved else None,
                              client_sizes=list(map(len, clients)), adapter_bytes_by_client=sizes)
    if not resume:
        save(engine.state)
    state = engine.run(max_events=max_events)
    runtime.verify_base()
    base_verified = True
    if state['finished']:
        runtime.save(directory / 'final-adapter', state['adapter'])
    return write_report(state)
