"""Pinned MMLU evaluation of global adapters; no Alpaca holdout is introduced."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict


CHOICES = "ABCD"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def format_question(row: dict, answer: bool = False) -> str:
    text = row["question"] + "\n"
    text += "\n".join(f"{letter}. {choice}" for letter, choice in zip(CHOICES, row["choices"]))
    text += "\nAnswer:"
    if answer:
        text += f" {CHOICES[row['answer']]}\n\n"
    return text


def candidate_tokens(tokenizer, prompt: str) -> tuple[list[int], list[list[int]]]:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    completions = []
    for letter in CHOICES:
        whole = tokenizer.encode(prompt + " " + letter, add_special_tokens=False)
        if whole[:len(prefix)] != prefix or len(whole) <= len(prefix):
            raise ValueError("Tokenizer changes the prompt boundary when adding an MMLU answer")
        completions.append(whole[len(prefix):])
    return prefix, completions


def fit_question(tokenizer, row: dict, demos: list[dict], n_shot: int, max_length: int):
    header = ("The following are multiple choice questions (with answers) about "
              + row["subject"].replace("_", " ") + ".\n\n")
    if len(demos) < n_shot:
        raise ValueError(f"Not enough dev demonstrations for {row['subject']}")
    for count in range(n_shot, -1, -1):
        prompt = header + "".join(format_question(d, True) for d in demos[:count]) + format_question(row)
        prefix, completions = candidate_tokens(tokenizer, prompt)
        if len(prefix) + max(map(len, completions)) <= max_length:
            return prefix, completions, count
    # Never silently remove the scored question or its Answer: suffix.
    raise ValueError(f"MMLU question in {row['subject']} exceeds evaluation context even at zero shot")


def score_candidates(model, prefix: list[int], completions: list[list[int]], device):
    import torch

    def forward(ids):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        return model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits

    with torch.inference_mode():
        if all(len(answer) == 1 for answer in completions):
            # Most target tokenizers need one token: share the expensive prefix pass.
            logprobs = forward(prefix)[0, -1].float().log_softmax(-1)
            return [float(logprobs[answer[0]]) for answer in completions]
        scores = []
        for answer in completions:
            logits = forward(prefix + answer[:-1])
            positions = logits[0, len(prefix) - 1:len(prefix) + len(answer) - 1].float()
            targets = torch.tensor(answer, device=device).unsqueeze(-1)
            scores.append(float(positions.log_softmax(-1).gather(-1, targets).sum()))
        return scores


class MMLUEvaluator:
    def __init__(self, settings: dict, tokenizer, root):
        from datasets import load_dataset

        self.settings, self.tokenizer = settings, tokenizer
        kwargs = {"path": "cais/mmlu", "name": "all", "revision": settings["revision"],
                  "cache_dir": str(root / settings["cache_dir"])}
        dev = load_dataset(**kwargs, split="dev")
        test = load_dataset(**kwargs, split="test")
        self.dev = defaultdict(list)
        for row in dev:
            self.dev[row["subject"]].append(row)
        chosen = list(range(len(test)))
        limit = settings["max_samples"]
        if limit is not None:
            chosen = sorted(random.Random(settings["subset_seed"]).sample(chosen, min(limit, len(chosen))))
        self.rows = [(index, dict(test[index])) for index in chosen]
        raw = json.dumps({"dev": dict(self.dev), "test": self.rows}, sort_keys=True).encode()
        self.manifest = {
            "dataset": "cais/mmlu", "revision": settings["revision"],
            "demonstration_split": "dev", "scored_split": "test", "n_shot": settings["n_shot"],
            "max_length": settings["max_length"], "test_indices": chosen,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "full_mmlu" if len(chosen) == len(test) else "debug_subset",
            "total_benchmark_questions": len(test), "evaluated_questions": len(chosen),
            "scoring": "summed conditional log probability of the complete answer label",
            "overall_aggregation": "question_weighted_accuracy",
        }
        # Tokenize once so every deadline uses exactly the same prompts and subset.
        self.encoded = []
        for index, row in self.rows:
            prefix, completions, shots = fit_question(tokenizer, row, self.dev[row["subject"]],
                                                      settings["n_shot"], settings["max_length"])
            self.encoded.append((index, row["subject"], row["answer"], prefix, completions, shots))

    def evaluate(self, model, device, prediction_path):
        started = time.perf_counter()
        was_training = model.training
        model.eval()
        totals = defaultdict(lambda: {"correct": 0, "count": 0})
        reduced = 0
        predictions = []
        try:
            for index, subject, answer, prefix, completions, shots in self.encoded:
                scores = score_candidates(model, prefix, completions, device)
                prediction = max(range(4), key=scores.__getitem__)
                totals[subject]["count"] += 1
                totals[subject]["correct"] += int(prediction == answer)
                reduced += shots < self.settings["n_shot"]
                predictions.append({"test_index": index, "subject": subject, "prediction": prediction,
                                    "answer": answer, "scores": scores, "shots_used": shots})
        finally:
            model.train(was_training)
        count = sum(x["count"] for x in totals.values())
        correct = sum(x["correct"] for x in totals.values())
        if not count:
            raise ValueError("MMLU evaluation has no questions")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in predictions), encoding="utf-8")
        return {"scope": self.manifest["scope"], "mmlu_accuracy": correct / count,
                "correct": correct, "count": count, "reduced_shot_questions": reduced,
                "per_subject": {key: {**value, "accuracy": value["correct"] / value["count"]}
                                for key, value in sorted(totals.items())},
                "real_eval_seconds": time.perf_counter() - started}
