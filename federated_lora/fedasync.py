"""FedAsync Algorithm 1 on the concatenated LoRA A/B parameter vector."""
from __future__ import annotations

import math
import torch
from .baselines import factor_pairs

PAPER = "https://arxiv.org/abs/1903.03934v5"
REFERENCE_COMMIT = "912ecf33e0fb811ac8551c4f3cdab344cdedfb53"


def staleness_weight(age, settings):
    if type(age) is not int or age < 0:
        raise ValueError("Staleness must be a nonnegative version count")
    kind = settings["function"]
    if kind == "constant":
        return 1.
    a = settings["a"]
    if isinstance(a, bool) or not math.isfinite(a) or a <= 0:
        raise ValueError("The decay parameter a must be finite and positive")
    if kind == "polynomial":
        return (age + 1.) ** (-a)
    if kind == "hinge":
        b = settings["b"]
        if isinstance(b, bool) or not math.isfinite(b) or b <= 0:
            raise ValueError("The hinge threshold b must be finite and positive")
        return 1. if age <= b else 1. / (a * (age - b) + 1.)
    raise ValueError("Use constant, polynomial, or hinge staleness weighting")


def mix(global_adapter, local_adapter, version, downloaded_version, alpha, decay):
    """Interpolate full factors with one coefficient; do not add a stale delta."""
    if type(version) is not int or type(downloaded_version) is not int or not 0 <= downloaded_version <= version:
        raise ValueError("Invalid server or download version")
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("FedAsync alpha must be between zero and one")
    factor_pairs(global_adapter)
    factor_pairs(local_adapter)
    if set(global_adapter) != set(local_adapter):
        raise ValueError("FedAsync adapter keys differ")
    if any(global_adapter[k].shape != local_adapter[k].shape for k in global_adapter):
        raise ValueError("FedAsync requires matching local and global factor shapes")
    age = version - downloaded_version
    weight = staleness_weight(age, decay)
    coefficient = alpha * weight
    result = {k: global_adapter[k].float() * (1. - coefficient) + local_adapter[k].float() * coefficient
              for k in global_adapter}
    if any(not torch.isfinite(v).all().item() for v in result.values()):
        raise FloatingPointError("FedAsync produced a non-finite adapter")
    return result, {"server_version_before": version, "downloaded_version": downloaded_version,
                    "staleness": age, "staleness_weight": weight, "mixing_coefficient": coefficient}


def proximal_loss(parameters, anchor, rho):
    """rho/2 times the SUM of squared distances from the downloaded factors."""
    if isinstance(rho, bool) or not math.isfinite(rho) or rho < 0 or not parameters:
        raise ValueError("Invalid proximal objective")
    if set(parameters) != set(anchor) or any(parameters[k].shape != anchor[k].shape for k in parameters):
        raise ValueError("The proximal anchor must match all trainable factors")
    return .5 * rho * sum((p.float() - anchor[k].detach().float()).square().sum()
                          for k, p in parameters.items())
