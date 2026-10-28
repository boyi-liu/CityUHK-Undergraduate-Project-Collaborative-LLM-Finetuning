"""Pinned LoRA training and complete-question zero-shot MMLU scoring."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
import torch
from .mmlu_data import digest


def prompt(row):
    return "Question: " + row["question"] + "\n" + "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"])) + "\nAnswer:"


def encode_training(row, tokenizer, width):
    prefix = tokenizer.encode(prompt(row), add_special_tokens=False)
    whole = tokenizer.encode(prompt(row) + " " + "ABCD"[row["answer"]], add_special_tokens=False)
    if whole[:len(prefix)] != prefix:
        raise ValueError("Tokenizer changed the training prompt boundary")
    ids = whole[:width]
    labels = [-100] * min(len(prefix), width) + ids[len(prefix):]
    return {"input_ids": ids, "labels": labels, "answer_tokens": sum(x != -100 for x in labels),
            "full_length": len(whole)}


def adapter_hash(adapter):
    result = hashlib.sha256()
    for name in sorted(adapter):
        value = adapter[name].detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def weighted_mean(updates, counts):
    if not updates or len(updates) != len(counts) or any(n <= 0 for n in counts):
        raise ValueError("Adapter aggregation requires positive client counts")
    if any(set(update) != set(updates[0]) for update in updates):
        raise ValueError("Client adapter keys differ")
    total = sum(counts)
    return {name: sum((update[name].float() * (n / total) for update, n in zip(updates, counts)),
                      torch.zeros_like(updates[0][name], dtype=torch.float32)) for name in updates[0]}


class MMLURuntime:
    def __init__(self, config, root, directory, source_manifest):
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config, self.directory = config, Path(directory)
        if not torch.cuda.is_available():
            raise RuntimeError("MMLU model training requires a CUDA GPU")
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        settings = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["revision"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(settings["name"], revision=settings["revision"], torch_dtype=torch.float32).cuda()
        base.config.use_cache = False
        self.model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=settings["lora_rank"],
            lora_alpha=settings["lora_alpha"], lora_dropout=settings["lora_dropout"], target_modules=settings["target_modules"], bias="none"))
        with torch.no_grad():
            for name, value in self.model.named_parameters():
                if "lora_A" in name:
                    value.uniform_(-0.01, 0.01)
                elif "lora_B" in name:
                    value.zero_()
        if any(p.requires_grad for n, p in self.model.named_parameters() if "lora_" not in n):
            raise RuntimeError("The base model must remain frozen")
        self.frozen_hashes = self.base_hashes()
        evaluation = config["evaluation"]
        test = [dict(r) for r in load_dataset("cais/mmlu", "all", revision=evaluation["revision"],
                                            cache_dir=str(Path(root) / evaluation["cache_dir"]), split="test")]
        if len(test) != source_manifest["test_count"] or digest(test) != source_manifest["test_content_sha256"]:
            raise ValueError("MMLU test data differ from the training-source audit")
        self.test_rows = test
        indices = list(range(len(test)))
        if evaluation["max_samples"] is not None:
            indices = sorted(random.Random(evaluation["subset_seed"]).sample(indices, min(len(indices), evaluation["max_samples"])))
        self.encoded_test = []
        for index in indices:
            row = test[index]
            text = prompt(row)
            prefix = self.tokenizer.encode(text, add_special_tokens=False)
            answers = [self.tokenizer.encode(text + " " + letter, add_special_tokens=False) for letter in "ABCD"]
            if not all(a[:len(prefix)] == prefix for a in answers):
                raise ValueError("Tokenizer changed the evaluation prompt boundary")
            tails = [a[len(prefix):] for a in answers]
            if not all(a and a[:-1] == tails[0][:-1] for a in tails):
                raise ValueError("The MMLU evaluator requires a shared prefix before the answer token")
            ids = prefix + tails[0][:-1]
            if len(ids) >= evaluation["max_length"]:
                raise ValueError("A complete MMLU question exceeds the evaluation context")
            self.encoded_test.append((index, row["subject"], row["answer"], ids, [a[-1] for a in tails]))
        self.manifest = {"dataset": "cais/mmlu", "revision": evaluation["revision"], "source_split": "test",
            "role": "checkpoint_selection", "independent_final_test": False, "n_shot": 0, "indices": indices,
            "test_content_sha256": digest(test), "encoded_prompts_sha256": digest(self.encoded_test),
            "scope": "full_mmlu" if len(indices) == len(test) else "debug_subset",
            "model_revision": settings["revision"], "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name()}

    def base_hashes(self):
        return {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for name, p in self.model.named_parameters() if "lora_" not in name}

    def state(self):
        from peft import get_peft_model_state_dict
        return {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(self.model).items()}

    def restore(self, adapter):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, adapter)
        if adapter_hash(self.state()) != adapter_hash(adapter):
            raise RuntimeError("Loaded adapter differs from the saved adapter")

    def save(self, folder, adapter):
        self.restore(adapter)
        self.model.save_pretrained(folder)

    def encode(self, rows):
        return [encode_training(row, self.tokenizer, self.config["model"]["max_length"]) for row in rows]

    def train(self, encoded, adapter, round_number, client_id):
        self.restore(adapter)
        start_hash = adapter_hash(self.state())
        cfg = self.config["training"]
        seed = self.config["seed"] + 1000 * round_number + client_id
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.train()
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=cfg["learning_rate"], weight_decay=0)
        scaler = torch.amp.GradScaler("cuda", init_scale=512)
        loss_sum = target_count = successful = skipped = empty = 0
        events = []
        started = time.perf_counter()
        width, size = self.config["model"]["max_length"], cfg["micro_batch_size"]
        for epoch in range(cfg["local_epochs"]):
            order = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            for start in range(0, len(order), size):
                indices = order[start:start + size]
                rows = [encoded[i] for i in indices]
                ids = torch.tensor([r["input_ids"] + [self.tokenizer.pad_token_id] * (width - len(r["input_ids"])) for r in rows], device="cuda")
                mask = torch.tensor([[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows], device="cuda")
                labels = torch.tensor([r["labels"] + [-100] * (width - len(r["labels"])) for r in rows], device="cuda")
                targets = int((labels[:, 1:] != -100).sum())
                optimizer.zero_grad(set_to_none=True)
                value = None
                if targets:
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = self.model(input_ids=ids, attention_mask=mask, labels=labels).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite MMLU client loss")
                    value = float(loss.detach())
                    loss_sum += value * targets
                    target_count += targets
                    scaler.scale(loss).backward()
                    scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped += int(scaler.get_scale() < scale)
                    successful += int(scaler.get_scale() >= scale)
                else:
                    empty += 1
                events.append({"epoch": epoch, "indices": indices, "target_tokens": targets, "loss": value})
        torch.cuda.synchronize()
        self.model.zero_grad(set_to_none=True)
        if target_count == 0:
            raise ValueError("A client has no answer targets at the configured token limit")
        result = {"client": client_id, "round": round_number, "training_loss": loss_sum / target_count,
                  "loss_sum": loss_sum, "target_tokens": target_count, "successful_steps": successful,
                  "skipped_steps": skipped, "no_target_batches": empty, "seconds": time.perf_counter() - started,
                  "start_adapter_sha256": start_hash, "batches": events}
        return self.state(), result

    @torch.inference_mode()
    def evaluate(self, adapter, round_number):
        self.restore(adapter)
        self.model.eval()
        core = self.model.get_base_model()
        groups = []
        for row in sorted(self.encoded_test, key=lambda r: len(r[3])):
            if not groups or len(groups[-1]) >= 8 or len(row[3]) * (len(groups[-1]) + 1) > 2048:
                groups.append([])
            groups[-1].append(row)
        predictions, totals = [], defaultdict(lambda: {"correct": 0, "count": 0})
        nll_sum = 0.0
        started = time.perf_counter()
        for group in groups:
            width = max(len(row[3]) for row in group)
            ids = torch.tensor([r[3] + [self.tokenizer.pad_token_id] * (width - len(r[3])) for r in group], device="cuda")
            mask = torch.tensor([[1] * len(r[3]) + [0] * (width - len(r[3])) for r in group], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = core.model(input_ids=ids, attention_mask=mask).last_hidden_state
                positions = torch.tensor([len(r[3]) - 1 for r in group], device="cuda")
                logits = core.lm_head(hidden[torch.arange(len(group), device="cuda"), positions]).float()
            log_probs = logits.log_softmax(-1)
            for b, (index, subject, answer, _, candidates) in enumerate(group):
                scores = logits[b, candidates].log_softmax(-1).tolist()
                nll = -float(log_probs[b, candidates[answer]])
                if not math.isfinite(nll) or not all(math.isfinite(s) for s in scores):
                    raise FloatingPointError("Non-finite MMLU evaluation score")
                prediction = max(range(4), key=scores.__getitem__)
                totals[subject]["correct"] += int(prediction == answer)
                totals[subject]["count"] += 1
                nll_sum += nll
                predictions.append({"index": index, "subject": subject, "answer": answer,
                                    "prediction": prediction, "scores": scores, "answer_nll": nll})
        predictions.sort(key=lambda r: r["index"])
        count, correct = len(predictions), sum(r["correct"] for r in totals.values())
        path = self.directory / f"round-{round_number}-predictions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
        return {"round": round_number, "correct": correct, "count": count, "accuracy": correct / count,
                "answer_nll": nll_sum / count, "per_subject": dict(sorted(totals.items())),
                "scope": self.manifest["scope"], "role": "checkpoint_selection", "independent_final_test": False,
                "seconds": time.perf_counter() - started}

    def verify_base(self):
        if self.base_hashes() != self.frozen_hashes:
            raise RuntimeError("The frozen base weights changed")
        return True
