import os
import math
import json
import numpy as np
from collections import defaultdict


def split_uniform(all_data, config):
    dir_path = config['dir_path']
    train_ratio = config['train_ratio']
    client_num = config['client_num']

    np.random.shuffle(all_data)

    size = len(all_data)
    chunk_size = math.ceil(size / client_num)

    for i in range(client_num):
        start = i * chunk_size
        end = min(start + chunk_size, size)
        chunk = all_data[start:end]

        split_idx = int(len(chunk) * train_ratio)
        train_chunk = chunk[:split_idx]
        test_chunk = chunk[split_idx:]

        save_file(train_chunk, f"{dir_path}/train/{i}.jsonl")
        save_file(test_chunk, f"{dir_path}/test/{i}.jsonl")

def split_dir(all_data, config):
    num_clients = int(config['client_num'])
    train_ratio = float(config['train_ratio'])
    dir_path = config['dir_path']
    alpha = float(config['alpha'])
    min_samples = int(config.get('min_samples_per_client', 10))
    max_attempts = int(config.get('max_split_attempts', 1000))
    rng = np.random.default_rng(config.get('seed', 42))

    if num_clients < 1 or not 0 < train_ratio < 1 or alpha <= 0:
        raise ValueError('client_num, train_ratio, and alpha must be positive; train_ratio must be below 1')
    if min_samples < 2 or max_attempts < 1:
        raise ValueError('min_samples_per_client must be at least 2 and max_split_attempts must be positive')
    if len(all_data) < num_clients * min_samples:
        raise ValueError('Not enough records to satisfy min_samples_per_client for every client')

    labels = np.array([item['category'] for item in all_data])
    categories = sorted(set(labels))

    for _ in range(max_attempts):
        idx_batch = [[] for _ in range(num_clients)]
        for category in categories:
            idx_k = rng.permutation(np.where(labels == category)[0])

            proportions = rng.dirichlet([alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            for i, s in enumerate(np.split(idx_k, split_points)):
                idx_batch[i].extend(s.tolist())

        if min(len(batch) for batch in idx_batch) >= min_samples:
            break
    else:
        raise ValueError(f'Could not create a Dirichlet split meeting the minimum size in {max_attempts} attempts')

    os.makedirs(f"{dir_path}/train", exist_ok=True)
    os.makedirs(f"{dir_path}/test", exist_ok=True)

    for j in range(num_clients):
        client_indices = idx_batch[j]
        rng.shuffle(client_indices)

        split_idx = max(1, min(len(client_indices) - 1, int(len(client_indices) * train_ratio)))
        train_idxs = client_indices[:split_idx]
        test_idxs = client_indices[split_idx:]

        # Keep the category so the client distribution can be audited later.
        train_data = [{k: v for k, v in all_data[idx].items()} for idx in train_idxs]
        test_data = [{k: v for k, v in all_data[idx].items()} for idx in test_idxs]
        
        save_file(train_data, f"{dir_path}/train/{j}.jsonl")
        save_file(test_data, f"{dir_path}/test/{j}.jsonl")


def save_file(data, pth):
    import os
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    with open(pth, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
