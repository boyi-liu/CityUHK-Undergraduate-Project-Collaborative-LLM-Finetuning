"""Single-GPU LoRA training and held-out validation for simulated clients."""

from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_training_model(config: dict):
    settings = config["model"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(settings["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(settings["name"], torch_dtype=dtype)
    base.config.use_cache = False
    base.config.pad_token_id = tokenizer.pad_token_id
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings["lora_rank"],
        lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        target_modules=settings["target_modules"],
        bias="none",
    )
    model = get_peft_model(base, lora).to(device)
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("The LoRA adapter has no trainable parameters")
    if any(p.requires_grad for name, p in model.named_parameters() if "lora_" not in name):
        raise RuntimeError("A base-model parameter is unexpectedly trainable")
    return model, tokenizer, device


def adapter_state(model) -> dict[str, torch.Tensor]:
    state = {name: tensor.detach().to(device="cpu", dtype=torch.float32).clone()
             for name, tensor in model.state_dict().items() if "lora_" in name}
    if not state or not any("lora_A" in key for key in state) or not any("lora_B" in key for key in state):
        raise RuntimeError("Both LoRA A and B tensors must be present")
    return state


def load_adapter(model, state: dict[str, torch.Tensor]) -> None:
    expected = set(adapter_state(model))
    if set(state) != expected:
        raise ValueError("Adapter keys do not match this model and LoRA configuration")
    model.load_state_dict(state, strict=False)


def encode_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    result = []
    for row in rows:
        prompt = f"### Instruction:\n{row['instruction'].strip()}\n"
        if row["input"].strip():
            prompt += f"### Input:\n{row['input'].strip()}\n"
        prompt += "### Response:\n"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_length // 2]
        answer_ids = tokenizer.encode(row["output"].strip() + tokenizer.eos_token,
                                      add_special_tokens=False)[:max_length - len(prompt_ids)]
        if not answer_ids:
            continue
        result.append({
            "input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids,
        })
    if not result:
        raise ValueError("No examples have usable response tokens")
    return result


def collate(rows: list[dict], pad_token_id: int, device: torch.device) -> dict:
    width = max(len(row["input_ids"]) for row in rows)
    ids = [row["input_ids"] + [pad_token_id] * (width - len(row["input_ids"])) for row in rows]
    labels = [row["labels"] + [-100] * (width - len(row["labels"])) for row in rows]
    masks = [[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in rows]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def train_client(model, encoded: list[dict], config: dict, device: torch.device,
                 pad_token_id: int, job_id: int, client_id: int) -> tuple[dict, float, float]:
    settings = config["training"]
    torch.manual_seed(config["seed"] + job_id * 1009 + client_id)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"] + job_id * 1009 + client_id)
    rng = random.Random(config["seed"] + job_id * 1009 + client_id)
    order = list(range(len(encoded)))
    rng.shuffle(order)
    cursor = 0
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=settings["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    losses = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(settings["local_steps"]):
        for _ in range(settings["grad_accum"]):
            selected = []
            for _ in range(settings["micro_batch_size"]):
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor = 0
                selected.append(encoded[order[cursor]])
                cursor += 1
            batch = collate(selected, pad_token_id, device)
            with _autocast(device):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite client {client_id} training loss")
            losses.append(float(loss.detach()))
            scaler.scale(loss / settings["grad_accum"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return adapter_state(model), sum(losses) / len(losses), elapsed


@torch.no_grad()
def evaluate(model, encoded: list[dict], config: dict, device: torch.device,
             pad_token_id: int) -> dict[str, float]:
    model.eval()
    limit = min(config["training"]["eval_limit"], len(encoded))
    total_loss = 0.0
    total_tokens = 0
    for row in encoded[:limit]:
        batch = collate([row], pad_token_id, device)
        with _autocast(device):
            loss = model(**batch).loss
        tokens = int((batch["labels"][:, 1:] != -100).sum())
        if tokens:
            total_loss += float(loss) * tokens
            total_tokens += tokens
    if total_tokens == 0:
        raise ValueError("Evaluation examples contain no response tokens")
    mean_loss = total_loss / total_tokens
    if not math.isfinite(mean_loss):
        raise FloatingPointError("Non-finite evaluation loss")
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss < 20 else None,
        "examples": limit,
        "response_tokens": total_tokens,
    }
