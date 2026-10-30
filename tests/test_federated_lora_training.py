"""Test the real trainer on a small random model, without model downloads."""

import copy
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("torch", "peft", "transformers"))


@unittest.skipUnless(DEPENDENCIES, "Requires torch, peft, and transformers")
class LocalEpochTrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def model(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from federated_lora.training import load_training_model

        config = {"seed": 42, "model": {
            "name": "local-test-model", "lora_rank": 2, "lora_alpha": 4,
            "lora_dropout": 0.0, "target_modules": "all-linear"},
            "training": {"local_epochs": 1, "micro_batch_size": 1,
                         "grad_accum": 32, "learning_rate": 0.001}}
        torch.manual_seed(42)
        base = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
            attention_dropout=0.0, tie_word_embeddings=True))
        tokenizer = SimpleNamespace(pad_token="pad", pad_token_id=0)
        with (patch("federated_lora.training.AutoModelForCausalLM.from_pretrained", return_value=base),
              patch("federated_lora.training.AutoTokenizer.from_pretrained", return_value=tokenizer),
              patch("torch.cuda.is_available", return_value=False)):
            model, _, device = load_training_model(config)
        return model, config, device

    def test_all_linear_covers_attention_and_feed_forward_and_freezes_backbone(self):
        model, _, _ = self.model()
        targets = {name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                   if hasattr(module, "lora_A")}
        self.assertEqual(targets, {"q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"})
        self.assertTrue(all("lora_" in name for name, p in model.named_parameters() if p.requires_grad))
        self.assertFalse(hasattr(model.get_output_embeddings(), "lora_A"))

    def test_accumulation_matches_batch_32_and_visits_35_rows_once(self):
        import torch
        from federated_lora.training import adapter_state, train_client

        model, config, device = self.model()
        whole_batch = copy.deepcopy(model)
        other_config = copy.deepcopy(config)
        other_config["training"].update(micro_batch_size=32, grad_accum=1)
        rows = [{"input_ids": [i + 3, 1, 2] + [4] * (i % 3),
                 "labels": [-100, -100, 2] + [4] * (i % 3)} for i in range(35)]
        seen = []
        hook = model.register_forward_pre_hook(
            lambda _, args, kwargs: seen.extend(kwargs["input_ids"][:, 0].tolist()), with_kwargs=True)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        step_calls = []
        original_step = torch.optim.AdamW.step

        def step(optimizer, *args, **kwargs):
            step_calls.append(1)
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.AdamW, "step", step):
            accumulated, loss, _ = train_client(model, rows, config, device, 0, 0, 0)
        hook.remove()
        expected, other_loss, _ = train_client(whole_batch, rows, other_config, device, 0, 0, 0)
        self.assertEqual(len(step_calls), 2)
        self.assertEqual(sorted(seen), list(range(3, 38)))
        self.assertTrue(math.isfinite(loss))
        self.assertAlmostEqual(loss, other_loss, places=5)
        for name in accumulated:
            torch.testing.assert_close(accumulated[name], expected[name], atol=2e-5, rtol=2e-4)
        for name, param in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(param, frozen[name]), name)
        self.assertTrue(any(torch.count_nonzero(tensor) for name, tensor in adapter_state(model).items()
                            if "lora_B" in name))

    def test_qwen_adapter_payload_matches_dry_run_estimate(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from federated_lora.config import load_config

        # Official Qwen2.5-0.5B dimensions. Meta tensors need no model weights.
        with torch.device("meta"):
            base = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=151936, hidden_size=896, intermediate_size=4864,
                num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
                tie_word_embeddings=True))
            model = get_peft_model(base, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                target_modules="all-linear", bias="none"))
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(count, 4_399_104)
        project = Path(__file__).resolve().parents[1]
        for name in ("sessions", "sessions-smoke"):
            config = load_config(project / f"configs/federated-lora-{name}.yaml")
            self.assertEqual(config["timing"]["dry_run_adapter_bytes"], count * 4)


if __name__ == "__main__":
    unittest.main()
