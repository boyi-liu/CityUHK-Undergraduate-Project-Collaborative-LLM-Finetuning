"""Baseline rules on trace events, deadline reporting, and process recovery."""
import copy
import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from federated_lora.config import load_config
from federated_lora.mmlu_runtime import adapter_hash
from federated_lora.mmlu_config import validate
from federated_lora.trace_baselines import run_sessions, schedule

PROJECT = Path(__file__).resolve().parents[1]
A, B = 'model.q_proj.lora_A.weight', 'model.q_proj.lora_B.weight'


def manifest():
    return {'format_version':2,'availability':'trace','availability_rule':'unit fixture',
        'sessions':[{'index':0,'start':'2020-01-30T06:30:00','deadline':'2020-01-30T06:31:00','start_us':0,'deadline_us':60_000_000},
                    {'index':1,'start':'2020-01-31T06:30:00','deadline':'2020-01-31T06:31:00','start_us':86_400_000_000,'deadline_us':86_460_000_000}],
        'clients':[{'client_id':cid,'trace_id':str(cid),'windows':[
            {'id':0,'start':'2020-01-30T06:30:00','end':'2020-01-30T06:31:00','start_us':0,'end_us':60_000_000}]} for cid in range(2)]}


def costs(config, cid, attempt, adapter_bytes, count):
    return {'train_us':10_000_000,'upload_us':5_000_000,'adapter_bytes':adapter_bytes,
            'profile':'fixture','step_seconds':1.,'optimizer_steps':1,'training_examples':count}


class Runtime:
    instances=[]
    def __init__(self,config,root,directory,source):
        self.config=config
        self.directory=directory
        self.manifest={'scope':'debug_subset'}
        self.test_rows=[]
        self.adapter={A:torch.ones(1,1),B:torch.zeros(1,1)}
        self.calls=[]
        self.__class__.instances.append(self)
    def state(self):
        return copy.deepcopy(self.adapter)
    def encode(self,rows):
        return [{'answer_tokens':1} for _ in rows]
    def save(self,path,adapter):
        self.adapter=copy.deepcopy(adapter)
        Path(path).mkdir(parents=True,exist_ok=True)
        torch.save(adapter,Path(path)/'fixture.pt')
    def train(self,rows,adapter,job,cid):
        if self.config['federation']['algorithm']=='flora':
            local={A:torch.ones(1,1),B:torch.tensor([[.01*job]])}
        else:
            local={A:adapter[A].clone(),B:adapter[B]+.01}
        return local,{'start_adapter_sha256':adapter_hash(adapter),'training_loss':1.,'seconds':1.}
    def evaluate(self,adapter,number):
        self.calls.append(number)
        (self.directory/self.prediction_name(number)).write_text('{}\n')
        score=[80,70,60][number]
        return {'round':number,'correct':score,'count':100,'accuracy':score/100,'scope':'debug_subset'}
    def verify_base(self):
        return True


class TraceBaselineTests(unittest.TestCase):
    def setUp(self):
        # Chart integration has its own tests; keep scheduler checks small.
        self.plots = patch('federated_lora.trace_plotting.plot_run').start()
        self.addCleanup(patch.stopall)
    def test_default_cli_uses_trace_deadlines_and_dispatches_before_standalone_async(self):
        from federated_lora.__main__ import main
        from federated_lora.mmlu_runner import run
        with patch.object(sys,'argv',['federated_lora','run']),patch('federated_lora.mmlu_runner.run',return_value={'fixture':True}) as call:
            main()
            cfg=call.call_args.args[0]
            self.assertEqual(cfg['stopping'],{'rule':'session_deadline'})
            self.assertTrue(cfg['sessions']['enabled'])
        with patch('federated_lora.trace_baselines.run_sessions',return_value={'fixture':True}) as call:
            cfg=self.config('fedasync_lora')
            self.assertEqual(run(cfg,PROJECT,runtime_factory=Runtime),{'fixture':True})
            self.assertIs(call.call_args.kwargs['runtime_factory'],Runtime)

    def test_simulation_only_flag_cannot_silently_start_model_training(self):
        from federated_lora.__main__ import main
        with patch.object(sys,'argv',['federated_lora','run','--simulation-only']),patch('federated_lora.mmlu_runner.run') as call:
            with self.assertRaises(SystemExit):
                main()
            call.assert_not_called()

    def config(self,algorithm):
        cfg=load_config(PROJECT/f"configs/mmlu-trace-{algorithm.replace('_','-')}-smoke.yaml")
        for key in ('training_path','manifest_path','subject_groups_path'):
            cfg['data'][key]=str(PROJECT/cfg['data'][key])
        cfg['data']['output_dir']='prepared'
        cfg['output_dir']='run'
        cfg['federation'].update(client_count=2,max_active=2)
        if algorithm!='fedasync_lora':
            cfg['federation']['clients_per_round']=2
        cfg['sessions'].update(trace_ids=[0,1],participation='repeated')
        return cfg

    def test_real_trace_configs_and_no_accuracy_stop_or_mode_swap(self):
        for algorithm in ('fedit','ffa-lora','flora','fedasync-lora'):
            for suffix in ('','-smoke','-pilot'):
                load_config(PROJECT/f'configs/mmlu-trace-{algorithm}{suffix}.yaml')
        cfg=self.config('fedasync_lora')
        for section,key,value in [('stopping','rule','first_accuracy_drop'),('federation','max_updates',5),
                                  ('sessions','availability','unknown'),('federation','mode','sync')]:
            bad=copy.deepcopy(cfg)
            bad[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate(bad)
        cfg=self.config('flora')
        cfg['federation']['mode']='async'
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_all_four_train_and_aggregate_then_report_quiet_deadline_despite_lower_scores(self):
        for algorithm in ('fedit','ffa_lora','flora','fedasync_lora'):
            with self.subTest(algorithm=algorithm),tempfile.TemporaryDirectory() as tmp:
                cfg=self.config(algorithm)
                with patch('federated_lora.trace_baselines.prepare_sessions',return_value=manifest()),patch('federated_lora.scheduler.job_durations',side_effect=costs):
                    result=run_sessions(cfg,Path(tmp),runtime_factory=Runtime)
                self.assertTrue(result['completed'])
                self.assertEqual(result['completed_deadlines'],2)
                self.assertGreater(result['version'],0)
                self.assertEqual(result['sessions'][1]['server_updates'],0)
                self.assertEqual(result['sessions'][1]['start_version'],result['sessions'][0]['version'])
                self.assertEqual(result['sessions'][0]['evaluation']['adapter_sha256'],result['sessions'][1]['evaluation']['adapter_sha256'])
                self.assertEqual([s['evaluation']['correct'] for s in result['sessions']],[70,60])
                self.assertFalse(result['test_used_for_checkpoint_selection'])
                self.assertFalse((Path(tmp)/'run/best-adapter').exists())
                self.assertTrue((Path(tmp)/'run/final-adapter').is_dir())
                self.assertTrue((Path(tmp)/'run/initial-predictions.jsonl').exists())
                self.assertTrue((Path(tmp)/'run/session-01-predictions.jsonl').exists())
                self.assertTrue((Path(tmp)/'run/session-02-predictions.jsonl').exists())
                records=[json.loads(p.read_text()) for p in (Path(tmp)/'run/server-updates').glob('*.json')]
                if algorithm=='fedasync_lora':
                    self.assertTrue(any(r['staleness']>0 for r in records))
                    self.assertTrue(all(len(r['jobs'])==1 for r in records))
                else:
                    self.assertTrue(all(len(r['jobs'])==2 for r in records))
                if algorithm=='ffa_lora':
                    payload=torch.load(Path(tmp)/'run/jobs/job-0/local-result.pt',weights_only=True)
                    self.assertEqual(set(payload),{B})

    def test_resume_preserves_pending_jobs_and_completed_run_does_not_evaluate_again(self):
        for algorithm in ('fedasync_lora','flora'):
            with self.subTest(algorithm=algorithm),tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryDirectory() as other:
                cfg=self.config(algorithm)
                with patch('federated_lora.trace_baselines.prepare_sessions',return_value=manifest()),patch('federated_lora.scheduler.job_durations',side_effect=costs):
                    partial=run_sessions(cfg,Path(tmp),runtime_factory=Runtime,max_events=4)
                    self.assertFalse(partial['completed'])
                    state=torch.load(Path(tmp)/'run/checkpoint.pt',weights_only=False)['state']
                    self.assertTrue(state['jobs'])
                    resumed=run_sessions(cfg,Path(tmp),resume=True,runtime_factory=Runtime)
                    expected=run_sessions(cfg,Path(other),runtime_factory=Runtime)
                    self.assertEqual(resumed,expected)
                    completed=run_sessions(cfg,Path(tmp),resume=True,runtime_factory=Runtime)
                    self.assertEqual(completed,resumed)
                    self.assertEqual(Runtime.instances[-1].calls,[])
                    previous=(Path(tmp)/'run/run_manifest.json').read_bytes()
                    changed=copy.deepcopy(cfg)
                    changed['timing']['speed_scale']*=2
                    with self.assertRaisesRegex(ValueError,'differs'):
                        run_sessions(changed,Path(tmp),resume=True,runtime_factory=Runtime)
                    self.assertEqual((Path(tmp)/'run/run_manifest.json').read_bytes(),previous)

    def test_real_availability_never_becomes_an_always_available_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.config('fedasync_lora')
            cfg['sessions'].update(trace_path=str(PROJECT/'dataset/device_usage_traces.json'),trace_ids=[0,1],
                                   start_time='22:00',end_time='22:10',start_date='2020-01-30',count=2)
            result=schedule(cfg,Path(tmp),simulate=True)
            self.assertTrue(result['simulation_only'])
            self.assertFalse(result['accuracy_measured'])
            self.assertTrue(all(s['server_updates']==0 for s in result['sessions']))


if __name__=='__main__':
    unittest.main()
