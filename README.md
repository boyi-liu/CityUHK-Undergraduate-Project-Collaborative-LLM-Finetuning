# FedLLM Factory

## Federated LoRA pipeline

`federated_lora` simulates separate clients on one CUDA GPU. It prepares disjoint,
task-skewed Alpaca partitions, runs synchronous rounds or asynchronous completion
events, and maintains a global LoRA A/B pair on the server. Completion times are
currently synthetic; real availability traces and session deadlines remain future
work. A device-usage trace is retained in `dataset/device_usage_traces.json` for
that work. The runner's current factor averaging and staleness interpolation establish
the pipeline; they are not yet the full set of baselines in the project brief.

The experiment settings are in `configs/federated-lora-smoke.yaml`,
`configs/federated-lora-pilot.yaml`, and `configs/federated-lora-full.yaml`.
The example limit, client count, Dirichlet alpha, model, LoRA settings, local
steps, selection count and mode are editable. Runs save the resolved
configuration and a client-partition manifest.

### Alpaca task groups

`dataset/alpaca_task_groups.jsonl` contains one task-group label for each of the
52,002 Stanford Alpaca rows, keyed by zero-based `source_index`. The nine groups
are `create_content`, `transform_content`, `explain_describe`, `find_fact`,
`suggest_design`, `classify`, `summarize`, `calculate`, and `other_unclear`. Labels
were assigned from the instruction text; they are model-generated annotations,
not Stanford's own per-row categories. The companion
`dataset/alpaca_task_groups.manifest.json` records the source SHA-256, label counts,
and classification provenance. The grouping file contains IDs, labels, and brief
reasons, not the Alpaca examples themselves.

Preparation checks the exact raw Alpaca file hash and label coverage before it
assigns every selected row to a training client with a Dirichlet draw. The
smoke and pilot configurations limit the rows for quick runs; `max_examples: null`
in the full configuration assigns all 52,002 rows. Client assignments
depend on the seed, client count, and Dirichlet alpha. Changing labels requires
preparing the data again. There is no Alpaca validation or test split. Assigning
all rows makes them available for training; client selection and local steps
still control which rows are actually visited in a run.

On a Linux CUDA server, create an isolated environment once:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
python -m pip install -r requirements-federated-lora.txt
```

From the repository root, prepare the official Stanford Alpaca data and run the
small smoke experiment. The preparer downloads the research-only Alpaca
dataset from `tatsu-lab/stanford_alpaca` into an ignored local directory; its
data license is CC BY-NC 4.0.

```bash
python -m federated_lora prepare --config configs/federated-lora-smoke.yaml
python -m federated_lora run --config configs/federated-lora-smoke.yaml --gpu 1
python -m federated_lora run --config configs/federated-lora-smoke.yaml --mode async --gpu 1
```

Check `nvidia-smi` and your server's GPU-use rules before choosing `--gpu`.
The `--gpu` value is the physical index reported by `nvidia-smi`; training runs
on one GPU at a time. Start with the smoke configuration, then prepare and run
`configs/federated-lora-pilot.yaml` once the smoke run succeeds. Use
`configs/federated-lora-full.yaml` to prepare all 52,002 rows for client
training. A stopped run can be continued using the same command with `--resume`.

Outputs are under `exp/federated-lora-smoke/{sync,async}` or the pilot equivalent:
`config.yaml`, `data_manifest.json`, `metrics.jsonl`, `checkpoint.pt`,
`adapter.pt` and `summary.json`. The reported client losses are training losses,
not evaluation scores. MMLU evaluation from the project brief is not implemented
yet. Generated data, checkpoints and raw examples are excluded from Git.
