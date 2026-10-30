"""Plot measured MMLU scores and client losses for one baseline algorithm.

This module reads JSON only. It needs no CUDA device or saved model weights.
It never turns schedule predictions into accuracy or interpolated measurements.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path

from .mmlu_data import digest
from .sessions import prepare_sessions
from .trace_comparison import CONDITIONS, comparison_configs

NAMES = {'fedit': 'FedIT', 'ffa_lora': 'FFA-LoRA', 'flora': 'FLoRA', 'fedasync_lora': 'FedAsync-LoRA'}
COLORS = {'always': '#2764a5', 'trace': '#198377'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def elapsed(manifest, session, time_us):
    prior = sum(s['deadline_us'] - s['start_us'] for s in manifest['sessions'][:session])
    return (prior + time_us - manifest['sessions'][session]['start_us']) / 3_600_000_000


def expected_points(config, manifest):
    interval = config['evaluation'].get('interval_seconds')
    points = []
    for session in manifest['sessions']:
        start, end = session['start_us'], session['deadline_us']
        if interval is not None:
            points.extend((session['index'], t, 'periodic') for t in range(start + round(interval * 1e6), end, round(interval * 1e6)))
        points.append((session['index'], end, 'deadline'))
    return points


def load_run(config, root):
    folder = Path(root) / config['output_dir']
    if not (folder / 'summary.json').exists():
        return None
    report, identity, manifest = [read(folder / p) for p in ('summary.json', 'run_manifest.json', 'session_manifest.json')]
    if (identity['configuration'] != config or report.get('algorithm') != config['federation']['algorithm']
            or report.get('mode') != config['federation']['mode'] or report.get('simulation_only')
            or report.get('test_used_for_checkpoint_selection') is not False
            or identity['data'] != report['data'] or identity['sessions'] != manifest
            or digest(manifest) != report['trace_manifest_sha256']
            or manifest != prepare_sessions(config, Path(root))):
        raise ValueError(f'Unmatched measured experiment: {folder}')
    actual = [(p['session'], p['time_us'], p['kind']) for p in report['evaluations']]
    expected = expected_points(config, manifest)
    if actual != expected[:len(actual)] or (report['completed'] and actual != expected):
        raise ValueError('Missing, duplicate, or mistimed evaluation points')
    points = [{'session': -1, 'kind': 'initial', 'time_us': 0, 'version': 0,
               'elapsed_session_seconds': 0, 'evaluation': report['initial_evaluation']}] + report['evaluations']
    for point in points:
        metric = point['evaluation']
        count, correct, accuracy = metric['count'], metric['correct'], metric['accuracy']
        if (type(count) is not int or type(correct) is not int or not 0 <= correct <= count or count < 1
                or not math.isfinite(accuracy) or not math.isclose(accuracy, correct / count, abs_tol=1e-12)
                or metric['scope'] != report['scope']):
            raise ValueError('Invalid measured accuracy, count, or scope')
        if count != points[0]['evaluation']['count']:
            raise ValueError('Evaluation question count changed within a run')
        if 'indices' in identity['evaluation'] and count != len(identity['evaluation']['indices']):
            raise ValueError('The score count differs from the pinned evaluation subset')
        if point['kind'] != 'initial' and not math.isclose(point['elapsed_session_seconds'] / 3600,
                elapsed(manifest, point['session'], point['time_us']), abs_tol=1e-12):
            raise ValueError('Evaluation uses the wrong simulated clock')
    return {'folder': folder, 'config': config, 'report': report, 'identity': identity, 'manifest': manifest, 'points': points}


def comparable(run):
    identity = run['identity']
    configuration = copy.deepcopy(identity['configuration'])
    configuration.pop('output_dir')
    configuration['sessions'].pop('availability')
    return {'configuration': configuration, **{k: identity[k] for k in
            ('data', 'evaluation', 'initial_adapter_sha256', 'client_upload_bytes')}}


def loss_rows(run):
    folder, manifest = run['folder'], run['manifest']
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text(encoding='utf-8').split('\n') if line]
    starts = {e['job_id']: e for e in events if e['event'] == 'job_started'}
    accepted = {j for e in events if e['event'] == 'aggregation' for j in e['jobs']}
    completed, batches = [], []
    for event in events:
        if event['event'] != 'training_done':
            continue
        job_id = event['job_id']
        start = starts[job_id]
        metrics = read(folder / 'jobs' / f'job-{job_id}' / 'training.json')
        if (metrics['job_id'] != job_id or metrics['client'] != start['client']
                or metrics['downloaded_version'] != start['base_version']):
            raise ValueError('A loss record differs from its scheduled job')
        value = metrics['training_loss']
        if not math.isfinite(value):
            raise ValueError('Non-finite measured training loss')
        row = {'availability': manifest['availability'], 'client': start['client'], 'job_id': job_id,
               'session': event['session'] + 1, 'elapsed_session_hours': elapsed(manifest, event['session'], event['time_us']),
               'training_loss': value, 'accepted_by_server': job_id in accepted,
               'actual_gpu_seconds': metrics['seconds']}
        completed.append(row)
        for index, batch in enumerate(metrics.get('batches', [])):
            batches.append({'availability': manifest['availability'], 'client': start['client'], 'job_id': job_id,
                'batch': index + 1, 'epoch': batch['epoch'] + 1, 'target_tokens': batch['target_tokens'],
                'data_loss': batch.get('data_loss', batch.get('loss')),
                'objective_loss': batch.get('objective_loss'), 'proximal_penalty': batch.get('proximal_penalty')})
    # Speculative results from jobs that never finish in simulated time are excluded.
    return completed, batches


def _save(fig, folder, name, plt):
    for extension in ('png', 'svg'):
        fig.savefig(folder / f'{name}.{extension}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _csv(folder, name, rows, fields):
    with (folder / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(config, root, runs, folder):
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(folder / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    trace_config = copy.deepcopy(config)
    trace_config['sessions']['availability'] = 'trace'
    manifest = prepare_sessions(trace_config, Path(root))
    algorithm = NAMES[config['federation']['algorithm']]
    sessions = manifest['sessions']
    deadlines = [elapsed(manifest, i, s['deadline_us']) for i, s in enumerate(sessions)]
    fig, axes = plt.subplots(len(sessions), 1, figsize=(11, 1.4 + 2.4 * len(sessions)), squeeze=False, layout='constrained')
    for ax, session in zip(axes[:, 0], sessions):
        hours = (session['deadline_us'] - session['start_us']) / 3_600_000_000
        for client in manifest['clients']:
            y = client['client_id']
            ax.broken_barh([(0, hours)], (y - .32, .64), facecolors='#edf1f5')
            windows = [(max(w['start_us'], session['start_us']), min(w['end_us'], session['deadline_us'])) for w in client['windows']]
            ax.broken_barh([((a - session['start_us']) / 3_600_000_000, (b-a) / 3_600_000_000)
                            for a, b in windows if b > a], (y - .32, .64), facecolors=COLORS['trace'])
        ax.set(yticks=[c['client_id'] for c in manifest['clients']],
               yticklabels=[f"Client {c['client_id']} / trace {c['trace_id']}" for c in manifest['clients']],
               xlim=(0, hours), ylim=(len(manifest['clients']) - .5, -.6), xlabel='Hours since session start',
               title=f"Session {session['index']+1}: {session['start']} to {session['deadline']}")
        ax.axvline(hours, color='#9a3340', linestyle='--', clip_on=False)
    fig.suptitle('Recorded device availability\nGreen: available; grey: unavailable; dashed line: deadline')
    _save(fig, folder, 'device-availability', plt)
    all_scores = [100*p['evaluation']['accuracy'] for r in runs.values() if r for p in r['points']]
    limits = (max(0, min(all_scores)-3), min(100, max(all_scores)+3)) if all_scores else (0, 100)
    accuracy_rows, deadline_rows, losses, batch_rows = [], [], [], []
    plots = ['device-availability']
    for condition in CONDITIONS:
        run = runs.get(condition)
        title = 'All devices available' if condition == 'always' else 'Recorded availability'
        scope = run['report']['scope'].replace('_', ' ') if run else 'no measured run'
        for kind in ('accuracy', 'training-loss', 'evaluation-loss'):
            fig, ax = plt.subplots(figsize=(10, 4.8))
            fig.subplots_adjust(bottom=.22, top=.83)
            ax.set(title=f'{algorithm}: {title} ({scope})',
                   xlabel='Simulated session time (hours; gaps between sessions excluded)',
                   xlim=(0, deadlines[-1]*1.03), ylabel={'accuracy':'Zero-shot MMLU accuracy (%)',
                   'training-loss':'Local answer loss (mean per target token)', 'evaluation-loss':'MMLU answer loss (negative log probability)'}[kind])
            ax.grid(alpha=.18)
            for i, hour in enumerate(deadlines):
                ax.axvline(hour, color='#9a3340', linestyle='--', alpha=.5)
                ax.text(hour, .97, f'Deadline {i+1}', transform=ax.get_xaxis_transform(), ha='right', va='top', fontsize=8,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':2})
            has_data = False
            if run and kind in ('accuracy', 'evaluation-loss'):
                points = run['points']
                key = 'accuracy' if kind == 'accuracy' else 'answer_nll'
                factor = 100 if kind == 'accuracy' else 1
                available = [p for p in points if key in p['evaluation']]
                if available:
                    has_data = True
                    ax.plot([p['elapsed_session_seconds']/3600 for p in available],
                            [factor*p['evaluation'][key] for p in available], 'o-', color=COLORS[condition], markersize=4)
                    for p in available:
                        if p['kind'] == 'deadline':
                            ax.scatter(p['elapsed_session_seconds']/3600, factor*p['evaluation'][key], marker='D', color=COLORS[condition])
            if kind == 'accuracy':
                ax.set_ylim(*limits)
                if run:
                    for p in run['points']:
                        accuracy_rows.append({'availability': condition, 'algorithm': algorithm, 'kind': p['kind'],
                            'session': p['session']+1, 'elapsed_session_hours': p['elapsed_session_seconds']/3600,
                            'version': p['version'], 'accuracy': p['evaluation']['accuracy'],
                            'answer_nll': p['evaluation'].get('answer_nll'), 'questions': p['evaluation']['count'],
                            'scope': run['report']['scope']})
                        if p['kind'] == 'deadline':
                            control = runs.get('always')
                            peer = next((q for q in control['points'] if q['kind']=='deadline' and q['session']==p['session']), None) if control else None
                            deadline_rows.append({'availability': condition, 'algorithm': algorithm, 'session':p['session']+1,
                                'deadline_hours':p['elapsed_session_seconds']/3600, 'accuracy':p['evaluation']['accuracy'],
                                'change_from_initial_pp':100*(p['evaluation']['accuracy']-run['points'][0]['evaluation']['accuracy']),
                                'gap_to_always_pp':100*(peer['evaluation']['accuracy']-p['evaluation']['accuracy']) if peer else None,
                                'version':p['version'], 'scope':run['report']['scope']})
            if run and kind == 'training-loss':
                local, batches = loss_rows(run)
                losses.extend(local)
                batch_rows.extend(batches)
                for cid in sorted({r['client'] for r in local}):
                    rows = sorted((r for r in local if r['client']==cid), key=lambda r:r['elapsed_session_hours'])
                    ax.plot([r['elapsed_session_hours'] for r in rows], [r['training_loss'] for r in rows], 'o-', markersize=3, label=f'Client {cid}')
                has_data = bool(local)
                if has_data:
                    ax.legend(loc='best', ncol=2, fontsize=8)
            if not has_data:
                ax.text(.5, .5, 'No measured data yet', transform=ax.transAxes, ha='center', va='center', fontsize=15)
            footer = ('Each point is one completed local job. Client data differ. Some uploads may be discarded.' if kind=='training-loss'
                      else 'Points are measured scores. Lines only connect points. GPU evaluation time is excluded.')
            if run and not run['report']['completed']:
                footer += '\nRun in progress; later times have no score yet.'
            fig.text(.5, .035, footer, ha='center', fontsize=8, color='#566270')
            name = f'{kind}-{condition}'
            _save(fig, folder, name, plt)
            plots.append(name)
    _csv(folder, 'accuracy.csv', accuracy_rows, ['availability','algorithm','kind','session','elapsed_session_hours','version','accuracy','answer_nll','questions','scope'])
    _csv(folder, 'deadline-comparison.csv', deadline_rows, ['availability','algorithm','session','deadline_hours','accuracy','change_from_initial_pp','gap_to_always_pp','version','scope'])
    _csv(folder, 'training-loss.csv', losses, ['availability','client','job_id','session','elapsed_session_hours','training_loss','accepted_by_server','actual_gpu_seconds'])
    _csv(folder, 'batch-loss.csv', batch_rows, ['availability','client','job_id','batch','epoch','target_tokens','data_loss','objective_loss','proximal_penalty'])
    result = {'algorithm':algorithm, 'complete':all(runs.get(c) and runs[c]['report']['completed'] for c in CONDITIONS),
              'missing_measured_runs':[c for c in CONDITIONS if not runs.get(c)], 'plots':plots,
              'time_axis':'Cumulative simulated session hours; gaps and GPU evaluation runtime excluded',
              'loss_scope':'Completed local jobs only; no global training loss; CSV marks accepted updates',
              'directory':str(folder)}
    (folder/'plot_manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def plot_run(config, root):
    run = load_run(config, root)
    return render(config, root, {config['sessions']['availability']:run}, Path(root)/config['output_dir']/'plots')


def plot_comparison(config, root):
    runs = {c['sessions']['availability']:load_run(c, root) for c in comparison_configs(config)}
    measured = [r for r in runs.values() if r]
    if measured and any(comparable(r)!=comparable(measured[0]) for r in measured[1:]):
        raise ValueError('Comparison runs differ in initial adapter, training data, or evaluation protocol')
    return render(config, root, runs, Path(root)/config['output_dir']/'comparison'/'plots')
