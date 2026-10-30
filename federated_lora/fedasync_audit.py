"""Independent update replay and saved-model checks for a completed run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from peft import PeftModel
from safetensors.torch import load_file

from .fedasync_runtime import FedAsyncRuntime
from .mmlu_data import load_pool
from .mmlu_runtime import adapter_hash
from .mmlu_runner import run, write_json


def audit(config, root):
    root=Path(root)
    directory=root/config['output_dir']
    summary=json.loads((directory/'summary.json').read_text())
    history=summary['update_history']
    decay=config['federation']['staleness']
    checks=[]
    positive_penalty=False
    for event in history[1:]:
        number=event['update']
        previous=load_file(directory/f'update-{number-1}-adapter/adapter_model.safetensors')
        current=load_file(directory/f'update-{number}-adapter/adapter_model.safetensors')
        downloaded=load_file(directory/f"update-{event['downloaded_version']}-adapter/adapter_model.safetensors")
        job_path=directory/'jobs'/f"job-{event['job_id']}"
        local=torch.load(job_path/'upload.pt',map_location='cpu',weights_only=True)
        job=json.loads((job_path/'training.json').read_text())
        age=number-1-event['downloaded_version']
        if decay['function']=='constant':
            weight=1.
        elif decay['function']=='polynomial':
            weight=(age+1.)**(-decay['a'])
        else:
            weight=1. if age<=decay['b'] else 1./(decay['a']*(age-decay['b'])+1.)
        coefficient=config['federation']['mixing_alpha']*weight
        expected={k:(1-coefficient)*previous[k]+coefficient*local[k] for k in previous}
        if (any(not torch.equal(current[k],expected[k]) for k in current) or event['staleness']!=age
                or event['mixing_coefficient']!=coefficient):
            raise AssertionError('The server update differs from the FedAsync equation')
        if adapter_hash(downloaded)!=job['start_adapter_sha256'] or adapter_hash(downloaded)!=event['downloaded_adapter_sha256']:
            raise AssertionError('A client used the wrong stale starting model')
        if job['optimizer']!='SGD' or job['proximal_rho']!=config['training']['proximal_rho']:
            raise AssertionError('The local optimizer differs from the reference objective')
        batches=[b for b in job['batches'] if b['target_tokens']]
        if batches[0]['proximal_penalty']!=0:
            raise AssertionError('The proximal anchor differs from the downloaded model')
        positive_penalty |= job['max_proximal_penalty']>0
        indices=[i for b in job['batches'] for i in b['indices']]
        expected_indices=list(range(summary['data']['client_counts'][job['client']]))*config['training']['local_epochs']
        if sorted(indices)!=sorted(expected_indices):
            raise AssertionError('The local job omitted or repeated assigned rows')
        checks.append({'update':number,'client':event['client'],'downloaded_version':event['downloaded_version'],
                       'staleness':age,'coefficient':coefficient,'exact_factor_replay':True,
                       'max_proximal_penalty':job['max_proximal_penalty']})
    best=min(history,key=lambda e:(-e['correct'],e['update']))
    if summary['selected_update']!=best['update']:
        raise AssertionError('The selected update is not the earliest best model')
    _,_,source=load_pool(config,root)
    runtime=FedAsyncRuntime(config,root,directory,source)
    base=runtime.model.unload()
    runtime.model=PeftModel.from_pretrained(base,directory/'best-adapter')
    saved=runtime.state()
    exported=runtime.model.peft_config['default']
    settings=config['model']
    if (adapter_hash(saved)!=summary['best_adapter_sha256'] or exported.r!=settings['lora_rank']
            or exported.lora_alpha!=settings['lora_alpha']):
        raise AssertionError('The saved best adapter has incorrect weights or scale')
    path=directory/f"update-{best['update']}-predictions.jsonl"
    before=path.read_bytes()
    with patch.object(runtime,'restore',lambda adapter:None):
        metric=runtime.evaluate(saved,best['update'])
    if before!=path.read_bytes() or metric['correct']!=best['correct']:
        raise AssertionError('The saved model did not reproduce the selected predictions and scores')
    runtime.verify_base()
    del runtime
    torch.cuda.empty_cache()
    times={p.name:p.stat().st_mtime_ns for p in directory.glob('update-*-predictions.jsonl')}
    resumed=run(config,root,resume=True)
    if resumed!=summary or times!={p.name:p.stat().st_mtime_ns for p in directory.glob('update-*-predictions.jsonl')}:
        raise AssertionError('Completed resume repeated updates or evaluation')
    result={'algorithm':'fedasync_lora','updates':checks,'frozen_base_verified':True,
            'positive_local_proximal_penalty_observed':positive_penalty,
            'fresh_export_predictions_and_scores_match':True,'completed_resume_repeated_no_updates':True,
            'selected_update':best['update'],'selected_correct':best['correct'],'count':best['count'],
            'scope':summary['scope'],'independent_final_test':False}
    write_json(directory/'fedasync-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    from .config import load_config
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(audit(load_config(root/args.config),root),indent=2))
