"""Reference equations, stale downloads, arrival order, and crash recovery."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from federated_lora.async_schedule import AsyncSchedule
from federated_lora.config import load_config
from federated_lora.fedasync import mix, proximal_loss, staleness_weight
from federated_lora.mmlu_config import validate
from federated_lora.mmlu_runner import run
from federated_lora.mmlu_runtime import adapter_hash

PROJECT = Path(__file__).resolve().parents[1]
A, B = 'model.q_proj.lora_A.weight', 'model.q_proj.lora_B.weight'


class EquationTests(unittest.TestCase):
    def test_three_paper_staleness_functions_and_hinge_boundary(self):
        self.assertEqual(staleness_weight(100, {'function':'constant'}), 1.)
        poly = {'function':'polynomial', 'a':.5}
        self.assertEqual(staleness_weight(0, poly), 1.)
        self.assertEqual(staleness_weight(3, poly), .5)
        hinge = {'function':'hinge', 'a':10., 'b':4}
        self.assertEqual(staleness_weight(4, hinge), 1.)
        self.assertEqual(staleness_weight(5, hinge), 1/11)
        self.assertEqual(staleness_weight(6, hinge), 1/21)
        for decay in (poly, hinge):
            values = [staleness_weight(s, decay) for s in range(30)]
            self.assertTrue(all(a >= b > 0 for a, b in zip(values, values[1:])))
        for age in (-1, True, .5):
            with self.assertRaises(ValueError):
                staleness_weight(age, poly)

    def test_full_factor_interpolation_uses_current_server_not_stale_anchor(self):
        current = {A:torch.tensor([[2.]]), B:torch.tensor([[3.]])}
        local = {A:torch.tensor([[8.]]), B:torch.tensor([[9.]])}
        before = adapter_hash(current), adapter_hash(local)
        result, event = mix(current, local, 7, 4, .6, {'function':'polynomial','a':.5})
        self.assertEqual(event['staleness'], 3)
        self.assertAlmostEqual(event['mixing_coefficient'], .3)
        torch.testing.assert_close(result[A], .7*current[A]+.3*local[A])
        torch.testing.assert_close(result[B], .7*current[B]+.3*local[B])
        self.assertEqual(before, (adapter_hash(current),adapter_hash(local)))
        # This baseline intentionally retains product cross terms.
        exact_products = .7*current[B]@current[A]+.3*local[B]@local[A]
        self.assertFalse(torch.allclose(result[B]@result[A], exact_products))
        with self.assertRaisesRegex(ValueError,'version'):
            mix(current,local,2,3,.6,{'function':'constant'})
        with self.assertRaises(ValueError):
            mix(current,{A:torch.zeros(2,1),B:torch.zeros(1,2)},2,1,.6,{'function':'constant'})
        with self.assertRaises(FloatingPointError):
            mix(current,{A:torch.full((1,1),float('nan')),B:local[B]},2,1,.6,{'function':'constant'})

    def test_proximal_sum_and_sgd_match_the_paper_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([2.,4.]))
        anchor = torch.tensor([1.,1.], requires_grad=True)
        rho, learning_rate = .5, .1
        penalty = proximal_loss({'A':parameter},{'A':anchor},rho)
        self.assertEqual(float(penalty.detach()), 2.5)
        data = .5 * parameter.square().sum()
        expected = parameter.detach() - learning_rate*(parameter.detach()+rho*(parameter.detach()-anchor.detach()))
        optimizer = torch.optim.SGD([parameter],lr=learning_rate)
        (data+penalty).backward()
        self.assertIsNone(anchor.grad)
        optimizer.step()
        torch.testing.assert_close(parameter,expected)
        self.assertEqual(float(proximal_loss({'A':parameter},{'A':parameter.detach().clone()},rho)),0.)


class ScheduleTests(unittest.TestCase):
    def config(self, kind):
        return {'seed':42, 'federation':{'client_count':2, 'simulation':
            {'schedule':'uniform_staleness','max_staleness':4} if kind=='uniform_staleness'
            else {'schedule':'arrivals','client_job_seconds':[1.,3.]}}}

    def test_arrivals_have_no_round_barrier_and_preserve_stale_downloads(self):
        cfg = self.config('arrivals')
        original = {'value':0}
        schedule = AsyncSchedule(cfg,original)
        original['value']=999
        first = schedule.next_job(0)
        self.assertEqual((first['client'],first['downloaded_version'],first['adapter']['value']),(0,0,0))
        schedule.complete(first,{'value':1},1)
        second = schedule.next_job(1)
        self.assertEqual((second['client'],second['downloaded_version'],second['adapter']['value']),(0,1,1))
        schedule.complete(second,{'value':2},2)
        third = schedule.next_job(2)
        self.assertEqual((third['client'],third['downloaded_version'],third['adapter']['value']),(1,0,0))
        self.assertEqual(third['arrives_at_us'],3_000_000)

    def test_uniform_versions_and_both_schedule_resumes_are_exact(self):
        for kind in ('uniform_staleness','arrivals'):
            cfg = self.config(kind)
            schedule = AsyncSchedule(cfg,{'value':0})
            observed=[]
            for version in range(20):
                restored = AsyncSchedule(cfg,{'value':999},schedule.checkpoint())
                job = schedule.next_job(version)
                self.assertEqual(restored.next_job(version),job)
                self.assertEqual(job['adapter']['value'],job['downloaded_version'])
                observed.append(version-job['downloaded_version'])
                if kind=='uniform_staleness':
                    self.assertTrue(0 <= observed[-1] <= min(4,version))
                schedule.complete(job,{'value':version+1},version+1)
            self.assertGreater(max(observed),0)
            if kind=='uniform_staleness':
                self.assertLessEqual(len(schedule.state['snapshots']),5)


class FakeRuntime:
    scores = [40,41,42,41]
    instances=[]
    def __init__(self, config, root, directory, source):
        self.manifest={'scope':'debug_subset'}
        self.test_rows=[]
        self.adapter={A:torch.ones(1,1),B:torch.zeros(1,1)}
        self.starts=[]
        self.evaluations=[]
        self.__class__.instances.append(self)
    def state(self):
        return copy.deepcopy(self.adapter)
    def encode(self, rows):
        return [{'answer_tokens':1} for _ in rows]
    def save(self, path, adapter):
        self.adapter=copy.deepcopy(adapter)
        Path(path).mkdir(parents=True,exist_ok=True)
    def train(self, rows, adapter, number, cid):
        self.starts.append((number,cid,adapter_hash(adapter)))
        return {k:v+number+cid for k,v in adapter.items()}, {'start_adapter_sha256':adapter_hash(adapter),
                'training_loss':1.,'successful_steps':1}
    def evaluate(self, adapter, number):
        self.evaluations.append(number)
        correct=self.scores[number]
        return {'update':number,'correct':correct,'count':100,'accuracy':correct/100}
    def verify_base(self):
        return True


class ConfigAndRunnerTests(unittest.TestCase):
    def config(self, kind):
        cfg=load_config(PROJECT/('configs/fedasync-lora-smoke.yaml' if kind=='uniform_staleness' else 'configs/fedasync-lora-arrivals-smoke.yaml'))
        for key in ('training_path','subject_groups_path','manifest_path'):
            cfg['data'][key]=str(PROJECT/cfg['data'][key])
        cfg['data']['output_dir'],cfg['output_dir']='prepared','run'
        cfg['federation']['max_updates']=3
        return cfg

    def test_invalid_configs_and_sync_async_schedules(self):
        for label in ('fedasync-lora','fedasync-lora-smoke','fedasync-lora-pilot','fedasync-lora-arrivals-smoke'):
            load_config(PROJECT/f'configs/{label}.yaml')
        cfg=self.config('uniform_staleness')
        changes=[('training','optimizer','Adam'),('training','proximal_rho',0),('federation','mode','sync'),
                 ('federation','mixing_alpha',1.),('federation','max_updates',True),('federation','rounds',10),
                 ('federation','client_ranks',[8]*5)]
        for section,key,value in changes:
            bad=copy.deepcopy(cfg)
            bad[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate(bad)
        for method in ('fedit','ffa_lora','flora'):
            bad=load_config(PROJECT/'configs/federated-lora-full.yaml')
            bad['federation'].update(algorithm=method,mode='async')
            with self.assertRaises(ValueError):
                validate(bad)
        bad=copy.deepcopy(cfg)
        bad['federation']['simulation']['max_staleness']=-1
        with self.assertRaises(ValueError):
            validate(bad)

    def test_stopping_saved_best_and_completed_resume(self):
        for kind in ('uniform_staleness','arrivals'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                cfg=self.config(kind)
                result=run(cfg,Path(tmp),runtime_factory=FakeRuntime)
                self.assertEqual((result['stopped_update'],result['selected_update']),(3,2))
                self.assertFalse(result['independent_final_test'])
                resumed=run(cfg,Path(tmp),resume=True,runtime_factory=FakeRuntime)
                self.assertEqual(resumed,result)
                self.assertEqual(FakeRuntime.instances[-1].starts,[])
                self.assertEqual(FakeRuntime.instances[-1].evaluations,[])
                original=(Path(tmp)/'run/run_manifest.json').read_bytes()
                changed=copy.deepcopy(cfg)
                changed['training']['proximal_rho']*=2
                with self.assertRaisesRegex(ValueError,'differs'):
                    run(changed,Path(tmp),resume=True,runtime_factory=FakeRuntime)
                self.assertEqual(original,(Path(tmp)/'run/run_manifest.json').read_bytes())

    def test_interrupted_job_recovers_pending_downloads_and_rng(self):
        class Interrupted(FakeRuntime):
            def train(self, rows, adapter, number, cid):
                if number>1:
                    raise RuntimeError('simulated interruption')
                return super().train(rows,adapter,number,cid)
        for kind in ('uniform_staleness','arrivals'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryDirectory() as clean:
                cfg=self.config(kind)
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    run(cfg,Path(tmp),runtime_factory=Interrupted)
                state=torch.load(Path(tmp)/'run/checkpoint.pt',weights_only=False)
                self.assertEqual(state['selection']['last_update'],1)
                resumed=run(cfg,Path(tmp),resume=True,runtime_factory=FakeRuntime)
                expected=run(cfg,Path(clean),runtime_factory=FakeRuntime)
                self.assertEqual(resumed,expected)

    def test_ties_keep_initial_model_and_update_budget_ends_run(self):
        class Tied(FakeRuntime):
            scores=[40]*4
        with tempfile.TemporaryDirectory() as tmp:
            result=run(self.config('uniform_staleness'),Path(tmp),runtime_factory=Tied)
            self.assertEqual((result['selected_update'],result['stopped_update']),(0,3))
            self.assertEqual(result['stop_reason'],'maximum_updates')


if __name__=='__main__':
    unittest.main()
