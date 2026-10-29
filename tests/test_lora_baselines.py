"""Product identities, real PEFT state changes, uploads, and resume."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from federated_lora.baselines import aggregate, factor_pairs, rank, upload
from federated_lora.baseline_runtime import BaselineRuntime
from federated_lora.config import load_config
from federated_lora.mmlu_config import validate
from federated_lora.mmlu_runner import run
from federated_lora.mmlu_runtime import MMLURuntime, adapter_hash

PROJECT = Path(__file__).resolve().parents[1]
A, B = 'layer.q_proj.lora_A.weight', 'layer.q_proj.lora_B.weight'


def factors(r, seed):
    g = torch.Generator().manual_seed(seed)
    return {A: torch.randn(r, 3, generator=g), B: torch.randn(4, r, generator=g)}


class ProductTests(unittest.TestCase):
    def test_ffa_only_averages_B_and_preserves_A_exactly(self):
        initial = factors(2, 1)
        x, y = factors(2, 2), factors(2, 3)
        x[A], y[A] = initial[A].clone(), initial[A].clone()
        payloads = [upload('ffa_lora', p, initial) for p in (x, y)]
        self.assertEqual([set(p) for p in payloads], [{B}, {B}])
        out = aggregate('ffa_lora', initial, payloads, [1, 3])
        self.assertTrue(torch.equal(out[A], initial[A]))
        torch.testing.assert_close(out[B] @ out[A], .25 * x[B] @ x[A] + .75 * y[B] @ y[A])
        x[A][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'fixed A'):
            upload('ffa_lora', x, initial)
        with self.assertRaisesRegex(ValueError, 'B only'):
            aggregate('ffa_lora', initial, [initial], [1])

    def test_flora_heterogeneous_products_and_two_rounds(self):
        initial = factors(2, 0)
        initial[B].zero_()
        x, y = factors(1, 1), factors(3, 2)
        before = adapter_hash(x)
        first = aggregate('flora', initial, [x, y], [1, 3])
        expected = .25 * x[B] @ x[A] + .75 * y[B] @ y[A]
        self.assertEqual(rank(first), 4)
        torch.testing.assert_close(first[B] @ first[A], expected)
        second = aggregate('flora', first, [x, y], [3, 1])
        self.assertEqual(rank(second), 8)
        torch.testing.assert_close(second[B] @ second[A], expected + .75 * x[B] @ x[A] + .25 * y[B] @ y[A])
        self.assertEqual(adapter_hash(x), before)
        # Separately averaged factors do not reproduce this product.
        u, v = factors(2, 4), factors(2, 5)
        stack = aggregate('flora', initial, [u, v], [1, 1])
        wrong = ((u[B] + v[B]) / 2) @ ((u[A] + v[A]) / 2)
        self.assertFalse(torch.allclose(stack[B] @ stack[A], wrong))

    def test_reject_invalid_factors_counts_and_client_ranks(self):
        initial = factors(2, 0)
        for bad in ({A: initial[A]}, {**initial, 'extra': torch.zeros(1)}, {A: torch.zeros(0, 3), B: torch.zeros(4, 0)}):
            with self.assertRaises(ValueError):
                factor_pairs(bad)
        for counts in ([0], [-1], [float('nan')], [True], []):
            with self.assertRaises(ValueError):
                aggregate('flora', initial, [initial], counts)
        cfg = load_config(PROJECT / 'configs/flora.yaml')
        cfg['federation']['client_ranks'] = [1, 2, 3, 4, 5]
        validate(cfg)
        for changes in ({'mode': 'async'}, {'algorithm': 'ffa_lora'}, {'client_ranks': [1, 2]}, {'client_ranks': [1, 2, True, 4, 5]}):
            bad = copy.deepcopy(cfg)
            bad['federation'].update(changes)
            with self.assertRaises(ValueError):
                validate(bad)


def toy_runtime(algorithm):
    cfg = {'seed': 42, 'model': {'lora_rank': 2, 'lora_alpha': 8, 'lora_dropout': 0.,
                               'target_modules': ['q_proj', 'v_proj']},
           'federation': {'algorithm': algorithm, 'client_count': 2, 'client_ranks': [1, 3]}}

    def init(self, config, *args):
        self.config = config
        torch.manual_seed(42)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
        self.original = copy.deepcopy(base)
        self.model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=8,
                                                   target_modules=['q_proj', 'v_proj'], lora_dropout=0.))
        self.initialize_adapter()
        self.frozen_hashes = self.base_hashes()
        self.manifest = {}

    with patch.object(MMLURuntime, '__init__', init):
        return BaselineRuntime(cfg)


class PeftTests(unittest.TestCase):
    def test_ffa_optimizer_cannot_change_A_or_base(self):
        runtime = toy_runtime('ffa_lora')
        initial = runtime.state()
        params = [(n, p) for n, p in runtime.model.named_parameters() if p.requires_grad]
        self.assertTrue(params)
        self.assertTrue(all('.lora_B.' in n for n, _ in params))
        runtime.model.train()
        optimizer = torch.optim.Adam([p for _, p in params], lr=.01)
        runtime.model(input_ids=torch.tensor([[1, 2, 3]])).logits.square().sum().backward()
        self.assertTrue(all(p.grad is None for n, p in runtime.model.named_parameters() if '.lora_A.' in n))
        optimizer.step()
        payload = upload('ffa_lora', runtime.state(), initial)
        self.assertTrue(any(torch.count_nonzero(v) for v in payload.values()))
        runtime.verify_base()

    def test_flora_fresh_local_rank_merge_scale_and_saved_reload(self):
        runtime = toy_runtime('flora')
        current = runtime.state()
        ids = torch.tensor([[1, 2, 3]])
        for rnd in (1, 2):
            runtime.restore(current)
            runtime.model.eval()
            expected = runtime.model(input_ids=ids).logits.detach()
            updates = []
            for cid, local_rank in enumerate((1, 3)):
                self.assertEqual(runtime.start_client(current, rnd, cid), adapter_hash(current))
                self.assertEqual(rank(runtime.state()), local_rank)
                self.assertTrue(all(torch.count_nonzero(runtime.state()[b]) == 0 for _, b in factor_pairs(runtime.state())))
                runtime.model.eval()
                torch.testing.assert_close(runtime.model(input_ids=ids).logits, expected, atol=1e-6, rtol=1e-5)
                optimizer = torch.optim.Adam([p for p in runtime.model.parameters() if p.requires_grad], lr=.01)
                runtime.model(input_ids=ids).logits.square().sum().backward()
                optimizer.step()
                updates.append(runtime.state())
            current = aggregate('flora', current, updates, [1, 3])
            self.assertEqual(rank(current), rnd * 4)
            runtime.restore(current)
            self.assertEqual(runtime.model.peft_config['default'].lora_alpha / rank(current), 4)
            runtime.verify_base()
        runtime.model.eval()
        expected = runtime.model(input_ids=ids).logits.detach()
        with tempfile.TemporaryDirectory() as tmp:
            runtime.save(tmp, current)
            loaded = PeftModel.from_pretrained(copy.deepcopy(runtime.original), tmp)
            loaded.eval()
            torch.testing.assert_close(loaded(input_ids=ids).logits, expected, atol=0, rtol=0)
            self.assertEqual(loaded.peft_config['default'].r, 8)


class FakeBaseline:
    def __init__(self, config, root, directory, source):
        self.config = config
        self.manifest = {'scope': 'debug_subset'}
        self.test_rows = []
        self.adapter = factors(2, 42)
        self.adapter[B].zero_()
    def state(self):
        return copy.deepcopy(self.adapter)
    def encode(self, rows):
        return [{'answer_tokens': 1} for _ in rows]
    def save(self, folder, adapter):
        self.adapter = copy.deepcopy(adapter)
        Path(folder).mkdir(parents=True, exist_ok=True)
    def train(self, rows, adapter, rnd, cid):
        local = factors(2, rnd * 100 + cid)
        if self.config['federation']['algorithm'] == 'ffa_lora':
            local[A] = adapter[A].clone()
        return local, {'start_adapter_sha256': adapter_hash(adapter), 'local_rank': 2,
                       'loss_sum': len(rows), 'target_tokens': len(rows)}
    def evaluate(self, adapter, rnd):
        score = [40, 50, 49][rnd]
        return {'round': rnd, 'correct': score, 'count': 100, 'accuracy': score / 100}
    def verify_base(self):
        return True


class RunnerTests(unittest.TestCase):
    def test_trace_runner_rejects_unsupported_baseline(self):
        cfg = yaml.safe_load((PROJECT / 'configs/federated-lora-sessions-smoke.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            for algorithm in ('ffa_lora', 'flora'):
                cfg['federation']['algorithm'] = algorithm
                path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'grouped MMLU round runner'):
                    load_config(path)

    def test_incomplete_flora_resume_uses_the_saved_global_product(self):
        class Interrupted(FakeBaseline):
            def train(self, rows, adapter, rnd, cid):
                if rnd == 2 and cid == 1:
                    raise RuntimeError('simulated interruption')
                return super().train(rows, adapter, rnd, cid)
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as clean:
            cfg = load_config(PROJECT / 'configs/flora-smoke.yaml')
            for key in ('training_path', 'manifest_path', 'subject_groups_path'):
                cfg['data'][key] = str(PROJECT / cfg['data'][key])
            cfg['data']['output_dir'], cfg['output_dir'] = 'prepared', 'run'
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                run(cfg, Path(tmp), runtime_factory=Interrupted)
            state = torch.load(Path(tmp) / 'run/checkpoint.pt', weights_only=False)
            self.assertEqual(state['selection']['last_round'], 1)
            self.assertEqual(rank(state['adapter']), 10)
            resumed = run(cfg, Path(tmp), resume=True, runtime_factory=FakeBaseline)
            complete = run(cfg, Path(clean), runtime_factory=FakeBaseline)
            self.assertEqual(resumed, complete)

    def test_both_baselines_save_payloads_select_and_resume(self):
        for algorithm in ('ffa_lora', 'flora'):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = load_config(PROJECT / f"configs/{algorithm.replace('_', '-')}-smoke.yaml")
                for key in ('training_path', 'manifest_path', 'subject_groups_path'):
                    cfg['data'][key] = str(PROJECT / cfg['data'][key])
                cfg['data']['output_dir'] = 'prepared'
                cfg['output_dir'] = 'run'
                result = run(cfg, root, runtime_factory=FakeBaseline)
                self.assertEqual((result['stopped_round'], result['selected_round']), (2, 1))
                self.assertEqual(result['best_checkpoint_rank'], 2 if algorithm == 'ffa_lora' else 10)
                payload = torch.load(root / 'run/clients/round-1/client-0/upload.pt', weights_only=True)
                self.assertEqual(set(payload), {B} if algorithm == 'ffa_lora' else {A, B})
                self.assertEqual(run(cfg, root, resume=True, runtime_factory=FakeBaseline), result)


if __name__ == '__main__':
    unittest.main()
