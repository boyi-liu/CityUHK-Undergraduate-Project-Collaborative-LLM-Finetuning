import time
import random
from functools import wraps


# Per-gradient-step wall-clock latency (seconds) measured on real devices,
# keyed by model family.  AGX = Jetson AGX Orin (client ids 0, 7); the rest
# are Jetson Orin Nano.
PER_STEP_LATENCY = {
    'qwen':      {'agx': (4.947, 4.962), 'nano': (7.087, 7.045)},
    'tinyllama': {'agx': (8.36, 8.44),   'nano': (14.538, 14.562)},
}

AGX_CLIENT_IDS = (0, 7)


def _model_family(model: str) -> str:
    name = (model or '').lower()
    if 'tinyllama' in name:
        return 'tinyllama'
    if 'qwen' in name:
        return 'qwen'
    raise ValueError(f'step_latency: no per-step latency profile for model {model!r}')


def step_latency(client_id, model, factor=1.0):
    table = PER_STEP_LATENCY[_model_family(model)]
    low, high = table['agx'] if client_id in AGX_CLIENT_IDS else table['nano']
    return random.uniform(low, high) * factor

def time_record(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        self.training_time = (end_time - start_time) * self.delay
        
        step_num = min(self.args.step, self.args.epoch * len(self.trainer.train_loader) // self.args.grad_accum)
        self.training_time = step_latency(self.id, self.args.model) * step_num
        print(f'Client {self.id} - Step num: {step_num}, Training time: {self.training_time:.2f} seconds')
        return result
    return wrapper