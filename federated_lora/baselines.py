"""Synchronous FFA-LoRA and FLoRA rules. See reports/lora-baselines.md."""
from __future__ import annotations

import math
import torch


def method(config):
    return config["federation"].get("algorithm", "fedit")


def factor_pairs(adapter):
    """Return PEFT A/B keys and reject incomplete or invalid factor pairs."""
    pairs = []
    for a in sorted(adapter):
        if ".lora_A." not in a:
            continue
        b = a.replace(".lora_A.", ".lora_B.")
        if b not in adapter:
            raise ValueError("A LoRA factor has no matching B factor")
        A, B = adapter[a], adapter[b]
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] < 1 or A.shape[0] != B.shape[1]:
            raise ValueError("Invalid LoRA factor dimensions")
        pairs.append((a, b))
    if not pairs or {k for pair in pairs for k in pair} != set(adapter):
        raise ValueError("Use paired LoRA A/B weights without extra parameters")
    return pairs


def rank(adapter):
    ranks = {adapter[a].shape[0] for a, _ in factor_pairs(adapter)}
    if len(ranks) != 1:
        raise ValueError("All target layers must use the same adapter rank")
    return ranks.pop()


def upload(algorithm, local, starting):
    """FFA sends B only. FLoRA sends both local factors."""
    pairs = factor_pairs(local)
    if algorithm == "ffa_lora":
        if set(local) != set(starting):
            raise ValueError("FFA-LoRA client keys differ from the shared adapter")
        if any(not torch.equal(local[a], starting[a]) for a, _ in pairs):
            raise ValueError("FFA-LoRA changed a fixed A factor")
        return {b: local[b].clone() for _, b in pairs}
    if algorithm != "flora":
        raise ValueError("Unknown synchronous baseline")
    return {k: v.clone() for k, v in local.items()}


def aggregate(algorithm, starting, payloads, counts):
    """Apply sample weights once. All factors use the same alpha/r scale."""
    if (not payloads or len(payloads) != len(counts) or
            any(isinstance(n, bool) or not math.isfinite(n) or n <= 0 for n in counts)):
        raise ValueError("Aggregation requires positive client counts")
    pairs = factor_pairs(starting)
    total = sum(counts)
    weights = [n / total for n in counts]
    if algorithm == "ffa_lora":
        expected = {b for _, b in pairs}
        if any(set(p) != expected for p in payloads):
            raise ValueError("FFA-LoRA uploads must contain B only")
        result = {a: starting[a].clone() for a, _ in pairs}
        for _, b in pairs:
            if any(p[b].shape != starting[b].shape for p in payloads):
                raise ValueError("FFA-LoRA B dimensions differ")
            result[b] = sum((p[b].float() * w for p, w in zip(payloads, weights)),
                            torch.zeros_like(starting[b], dtype=torch.float32))
        return result
    if algorithm != "flora":
        raise ValueError("Unknown synchronous baseline")
    for p in payloads:
        factor_pairs(p)
        if set(p) != set(starting):
            raise ValueError("FLoRA target layers differ")
    # Preserve earlier merged rounds in a portable adapter on the original base.
    # This is a checkpoint representation, not rank compression or a local rank.
    keep_previous = any(torch.count_nonzero(starting[b]).item() for _, b in pairs)
    result = {}
    for a, b in pairs:
        if any(p[a].shape[1] != starting[a].shape[1] or p[b].shape[0] != starting[b].shape[0]
               for p in payloads):
            raise ValueError("FLoRA outer dimensions differ")
        As = ([starting[a]] if keep_previous else []) + [p[a] * w for p, w in zip(payloads, weights)]
        Bs = ([starting[b]] if keep_previous else []) + [p[b] for p in payloads]
        result[a], result[b] = torch.cat(As, dim=0), torch.cat(Bs, dim=1)
    return result
