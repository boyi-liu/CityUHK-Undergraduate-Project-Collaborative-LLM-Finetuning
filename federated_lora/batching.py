"""Plan full local epochs without repeating rows to fill a batch."""

import math
import random


def optimizer_steps(example_count: int, settings: dict) -> int:
    if type(example_count) is not int or example_count < 1:
        raise ValueError("A local epoch needs at least one usable example")
    batch_size = settings["micro_batch_size"] * settings["grad_accum"]
    return settings["local_epochs"] * math.ceil(example_count / batch_size)


def epoch_batches(example_count: int, settings: dict, seed: int):
    """Yield one group of micro-batches per optimizer update, for each epoch."""
    optimizer_steps(example_count, settings)
    rng = random.Random(seed)
    batch_size = settings["micro_batch_size"] * settings["grad_accum"]
    for _ in range(settings["local_epochs"]):
        order = list(range(example_count))
        rng.shuffle(order)
        for start in range(0, example_count, batch_size):
            selected = order[start:start + batch_size]
            yield [selected[i:i + settings["micro_batch_size"]]
                   for i in range(0, len(selected), settings["micro_batch_size"])]
