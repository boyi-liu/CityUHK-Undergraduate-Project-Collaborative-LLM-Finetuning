"""PEFT state handling for the two synchronous reference methods."""
from __future__ import annotations

import torch
from .baselines import method, rank
from .mmlu_runtime import MMLURuntime, adapter_hash


class BaselineRuntime(MMLURuntime):
    def initialize_adapter(self):
        # PEFT initializes A with Kaiming uniform and B with zeros.
        # FFA-LoRA Appendix A.8 reports this initialization.
        if method(self.config) == "ffa_lora":
            for name, parameter in self.model.named_parameters():
                if ".lora_A." in name:
                    parameter.requires_grad_(False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm = method(self.config)
        self.fixed_A = {k: v.clone() for k, v in self.state().items() if ".lora_A." in k}
        # Retain only the target base weights. Other base weights never change.
        self.target_weights = {name: module.get_base_layer().weight.detach().cpu().clone()
                               for name, module in self.model.get_base_model().named_modules()
                               if hasattr(module, "lora_A")}
        self.manifest.update({"algorithm": self.algorithm, "schedule": "synchronous",
                              "initialization": "peft_kaiming_A_zero_B", "differential_privacy": False,
                              "local_ranks": self.local_ranks(),
                              "lora_scale": self.scale,
                              "checkpoint_form": "cumulative_stack_on_original_base" if self.algorithm == "flora" else "fixed_A_and_global_B"})

    @property
    def scale(self):
        cfg = self.config["model"]
        return cfg["lora_alpha"] / cfg["lora_rank"]

    def local_ranks(self):
        return self.config["federation"].get("client_ranks", [self.config["model"]["lora_rank"]] * self.config["federation"]["client_count"])

    def _wrap(self, base, local_rank):
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = self.config["model"]
        return get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=local_rank,
            lora_alpha=self.scale * local_rank, lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"], bias="none"))

    def _original_base(self):
        base = self.model.unload()
        modules = dict(base.named_modules())
        with torch.no_grad():
            for name, value in self.target_weights.items():
                modules[name].weight.copy_(value)
        return base

    def restore(self, adapter):
        if self.algorithm == "flora":
            self.model = self._wrap(self._original_base(), rank(adapter))
        else:
            if any(not torch.equal(adapter[k], value) for k, value in self.fixed_A.items()):
                raise ValueError("FFA-LoRA attempted to replace its fixed A")
        super().restore(adapter)

    def start_client(self, adapter, round_number, client_id):
        if self.algorithm != "flora":
            return super().start_client(adapter, round_number, client_id)
        self.restore(adapter)
        # Follow FLoRA: merge the preceding global product, then start a fresh
        # local adapter. Never train at the growing checkpoint rank.
        base = self.model.merge_and_unload(safe_merge=True)
        seed = self.config["seed"] + 1000 * round_number + client_id
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model = self._wrap(base, self.local_ranks()[client_id])
        initial = self.state()
        if any(torch.count_nonzero(v).item() for k, v in initial.items() if ".lora_B." in k):
            raise RuntimeError("FLoRA must start each local adapter with zero B")
        self.local_initial_hash = adapter_hash(initial)
        return adapter_hash(adapter)

    def train(self, encoded, adapter, round_number, client_id):
        try:
            result, job = super().train(encoded, adapter, round_number, client_id)
            if self.algorithm == "ffa_lora":
                if any(not torch.equal(result[k], value) for k, value in self.fixed_A.items()):
                    raise RuntimeError("Local training changed a fixed A factor")
            job.update({"algorithm": self.algorithm, "local_rank": rank(result),
                        "lora_scale": self.scale,
                        "fresh_local_adapter": self.algorithm == "flora",
                        "local_initial_adapter_sha256": self.local_initial_hash if self.algorithm == "flora" else job["start_adapter_sha256"],
                        "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)})
            return result, job
        finally:
            # Leave the runtime on the original frozen base, including failures.
            if self.algorithm == "flora":
                self.restore(adapter)

    def verify_base(self):
        if self.algorithm == "ffa_lora":
            if any(not torch.equal(self.state()[k], value) for k, value in self.fixed_A.items()):
                raise RuntimeError("The fixed FFA-LoRA A changed")
        return super().verify_base()
