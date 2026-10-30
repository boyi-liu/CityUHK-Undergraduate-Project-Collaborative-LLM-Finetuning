"""Replay accepted trace updates and verify the last deadline's saved model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from peft import PeftModel
from safetensors.torch import load_file

from .baselines import aggregate, factor_pairs
from .mmlu_runtime import adapter_hash, weighted_mean
from .mmlu_data import load_pool
from .mmlu_runner import write_json
from .trace_baselines import make_runtime, run_sessions


def audit(config, root):
    root=Path(root)
    folder=root/config['output_dir']
    result=json.loads((folder/'summary.json').read_text())
    if not result['completed']:
        raise ValueError('Only a completed trace run can be audited')
    manifest=json.loads((folder/'session_manifest.json').read_text())
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    state=checkpoint['state']
    history=state['history']
    algorithm=config['federation']['algorithm']
    current=load_file(folder/'initial-adapter/adapter_model.safetensors')
    hashes={0:adapter_hash(current)}
    starts={event['job_id']:event for event in history if event['event']=='job_started'}
    uploads={event['job_id']:event for event in history if event['event']=='upload_done'}
    accepted=[]
    for event in history:
        if event['event']!='aggregation':
            continue
        version=event['version']
        ledger=json.loads((folder/'server-updates'/f'{version}.json').read_text())
        if ledger['before_sha256']!=hashes[version-1] or ledger['jobs']!=event['jobs']:
            raise AssertionError('The server ledger differs from the accepted event history')
        values,counts=[],[]
        for job_id in event['jobs']:
            job=json.loads((folder/'jobs'/f'job-{job_id}'/'training.json').read_text())
            payload=torch.load(folder/'jobs'/f'job-{job_id}'/'local-result.pt',map_location='cpu',weights_only=True)
            start,arrival=starts[job_id],uploads[job_id]
            session=manifest['sessions'][arrival['session']]
            window=next(w for w in manifest['clients'][start['client']]['windows'] if w['id']==start['window'])
            if not (max(window['start_us'],session['start_us'])<=start['time_us']<min(window['end_us'],session['deadline_us'])):
                raise AssertionError('A client trained outside its recorded availability')
            if not (arrival['time_us']==start['training_finish_us']+start['upload_us']
                    and arrival['time_us']<=min(window['end_us'],session['deadline_us'])):
                raise AssertionError('An unavailable or late upload changed the server')
            if job['downloaded_adapter_sha256']!=hashes[job['downloaded_version']]:
                raise AssertionError('A client downloaded the wrong global model')
            if job['start_adapter_sha256']!=job['downloaded_adapter_sha256']:
                raise AssertionError('Local training used a different adapter from the download')
            if job['upload_bytes']!=sum(v.numel()*v.element_size() for v in payload.values()):
                raise AssertionError('Upload byte accounting differs from the payload')
            values.append(payload)
            counts.append(result['data']['client_counts'][job['client']])
        if algorithm=='fedasync_lora':
            age=version-1-ledger['downloaded_versions'][0]
            decay=config['federation']['staleness']
            weight=(1. if decay['function']=='constant' else (age+1.)**(-decay['a']) if decay['function']=='polynomial'
                    else 1. if age<=decay['b'] else 1./(decay['a']*(age-decay['b'])+1.))
            coefficient=config['federation']['mixing_alpha']*weight
            if len(values)!=1 or age!=ledger['staleness'] or coefficient!=ledger['mixing_coefficient']:
                raise AssertionError('The asynchronous rule differs from FedAsync')
            new={k:(1-coefficient)*current[k]+coefficient*values[0][k] for k in current}
        else:
            if len(values)!=config['federation']['clients_per_round'] or any(v!=version-1 for v in ledger['downloaded_versions']):
                raise AssertionError('A synchronous round used incomplete or stale clients')
            new=weighted_mean(values,counts) if algorithm=='fedit' else aggregate(algorithm,current,values,counts)
            if algorithm=='flora':
                scale=config['model']['lora_alpha']/config['model']['lora_rank']
                for a,b in factor_pairs(new):
                    expected=current[b]@current[a]*scale+sum(n/sum(counts)*(p[b]@p[a])*scale for p,n in zip(values,counts))
                    torch.testing.assert_close(new[b]@new[a]*scale,expected,atol=2e-7,rtol=2e-4)
        if adapter_hash(new)!=ledger['after_sha256']:
            raise AssertionError('Replayed factors differ from the accepted global update')
        current=new
        hashes[version]=adapter_hash(current)
        accepted.append(version)
    for point in state['evaluations']:
        if point['kind']!='deadline':
            continue
        session=manifest['sessions'][point['session']]
        if point['time_us']!=session['deadline_us'] or point['evaluation']['adapter_sha256']!=hashes[point['version']]:
            raise AssertionError('A deadline report used the wrong clock or global state')
    if result['test_used_for_checkpoint_selection'] or (folder/'best-adapter').exists():
        raise AssertionError('Deadline reporting must not select a model with test accuracy')
    if adapter_hash(current)!=result['final_adapter_sha256'] or len(state['evaluations'])<len(manifest['sessions']):
        raise AssertionError('Final model or deadline coverage differs')
    _,_,source=load_pool(config,root)
    runtime=make_runtime(config,root,folder,source)
    runtime.model=PeftModel.from_pretrained(runtime.model.unload(),folder/'final-adapter')
    final=runtime.state()
    exported=runtime.model.peft_config['default']
    if adapter_hash(final)!=adapter_hash(current) or exported.lora_alpha/exported.r!=config['model']['lora_alpha']/config['model']['lora_rank']:
        raise AssertionError('The final PEFT export has incorrect weights or scale')
    label=f"session-{len(manifest['sessions']):02d}"
    path=folder/f'{label}-predictions.jsonl'
    before=path.read_bytes()
    runtime.prediction_name=lambda number:f'{label}-predictions.jsonl'
    with patch.object(runtime,'restore',lambda adapter:None):
        metric=runtime.evaluate(final,0)
    if before!=path.read_bytes() or metric['correct']!=result['sessions'][-1]['evaluation']['correct']:
        raise AssertionError('The final model did not reproduce its deadline predictions and scores')
    runtime.verify_base()
    del runtime
    torch.cuda.empty_cache()
    times={p.name:p.stat().st_mtime_ns for p in folder.glob('*-predictions.jsonl')}
    resumed=run_sessions(config,root,resume=True)
    if resumed!=result or times!={p.name:p.stat().st_mtime_ns for p in folder.glob('*-predictions.jsonl')}:
        raise AssertionError('Completed resume repeated training or deadline evaluation')
    checks={'algorithm':algorithm,'accepted_versions_replayed':accepted,'recorded_availability_checked':True,
            'late_uploads_excluded':True,'all_deadline_states_verified':True,'frozen_base_verified':True,
            'fresh_final_export_predictions_and_scores_match':True,'completed_resume_repeated_no_events':True,
            'test_used_for_checkpoint_selection':False,'independent_final_test':False}
    write_json(folder/'trace-audit.json',checks)
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    from .config import load_config
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(audit(load_config(root/args.config),root),indent=2))
