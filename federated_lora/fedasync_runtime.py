"""SGD on the FedAsync proximal objective, restricted to LoRA factors."""
from __future__ import annotations

import time
import torch
from .fedasync import PAPER, REFERENCE_COMMIT, proximal_loss
from .mmlu_runtime import MMLURuntime, adapter_hash


class FedAsyncRuntime(MMLURuntime):
    def initialize_adapter(self):
        # Keep PEFT's Kaiming A and zero B, as in the other new baselines.
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest.update({'algorithm': 'fedasync_lora', 'paper': PAPER, 'reference_commit': REFERENCE_COMMIT,
            'adaptation': 'x=concatenated_A_B;factor_proximal_SGD;shared_factor_mixing',
            'initialization': 'peft_kaiming_A_zero_B', 'optimizer': 'SGD',
            'proximal_rho': self.config['training']['proximal_rho'],
            'schedule': self.config['federation'].get('simulation', {'schedule': 'trace_sessions'}),
            'product_consistent': False, 'protocol_version': 1})

    def train(self, encoded, adapter, job_number, client_id):
        self.restore(adapter)
        start_hash = adapter_hash(self.state())
        cfg = self.config['training']
        seed = self.config['seed'] + 1000 * job_number + client_id
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.train()
        parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        if not parameters or any('.lora_A.' not in n and '.lora_B.' not in n for n in parameters):
            raise RuntimeError('FedAsync-LoRA must train A/B factors only')
        anchor = {n: p.detach().clone() for n, p in parameters.items()}
        optimizer = torch.optim.SGD(parameters.values(), lr=cfg['learning_rate'], momentum=0, weight_decay=0)
        scaler = torch.amp.GradScaler('cuda', init_scale=512)
        loss_sum = objective_sum = penalty_sum = max_penalty = 0.
        target_count = successful = skipped = empty = 0
        events = []
        started = time.perf_counter()
        width, size = self.config['model']['max_length'], cfg['micro_batch_size']
        for epoch in range(cfg['local_epochs']):
            order = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            for start in range(0, len(order), size):
                indices = order[start:start + size]
                rows = [encoded[i] for i in indices]
                ids = torch.tensor([r['input_ids'] + [self.tokenizer.pad_token_id] * (width - len(r['input_ids'])) for r in rows], device='cuda')
                mask = torch.tensor([[1] * len(r['input_ids']) + [0] * (width - len(r['input_ids'])) for r in rows], device='cuda')
                labels = torch.tensor([r['labels'] + [-100] * (width - len(r['labels'])) for r in rows], device='cuda')
                targets = int((labels[:, 1:] != -100).sum())
                optimizer.zero_grad(set_to_none=True)
                raw = penalty = objective = None
                if targets:
                    with torch.autocast('cuda', dtype=torch.float16):
                        data_loss = self.model(input_ids=ids, attention_mask=mask, labels=labels).loss
                    regularizer = proximal_loss(parameters, anchor, cfg['proximal_rho'])
                    loss = data_loss.float() + regularizer
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError('Non-finite FedAsync local objective')
                    raw, penalty, objective = float(data_loss.detach()), float(regularizer.detach()), float(loss.detach())
                    loss_sum += raw * targets
                    objective_sum += objective * targets
                    penalty_sum += penalty
                    max_penalty = max(max_penalty, penalty)
                    target_count += targets
                    scaler.scale(loss).backward()
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped += int(scaler.get_scale() < previous_scale)
                    successful += int(scaler.get_scale() >= previous_scale)
                else:
                    empty += 1
                events.append({'epoch': epoch, 'indices': indices, 'target_tokens': targets,
                               'data_loss': raw, 'proximal_penalty': penalty, 'objective_loss': objective})
        torch.cuda.synchronize()
        self.model.zero_grad(set_to_none=True)
        if not target_count:
            raise ValueError('A FedAsync client has no answer targets at this token limit')
        return self.state(), {'client': client_id, 'job_seed_index': job_number, 'optimizer': 'SGD',
            'proximal_rho': cfg['proximal_rho'], 'training_loss': loss_sum / target_count,
            'objective_loss': objective_sum / target_count, 'loss_sum': loss_sum, 'target_tokens': target_count,
            'sum_proximal_penalties': penalty_sum, 'max_proximal_penalty': max_penalty,
            'successful_steps': successful, 'skipped_steps': skipped, 'no_target_batches': empty,
            'seconds': time.perf_counter() - started, 'start_adapter_sha256': start_hash, 'batches': events}

    def prediction_name(self, number):
        return f'update-{number}-predictions.jsonl'

    def evaluate(self, adapter, number):
        result = super().evaluate(adapter, number)
        result['update'] = result.pop('round')
        return result
