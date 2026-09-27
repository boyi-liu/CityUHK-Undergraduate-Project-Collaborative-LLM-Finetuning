# FedLLM Factory

## Point 2: federated LoRA pipeline

`point2` is the reproducible multi-client pipeline for the second compulsory FYP
task. It simulates separate clients on one CUDA GPU, with disjoint, category-skewed
Alpaca partitions, synchronous rounds or asynchronous completion events, and a
server-maintained LoRA A/B pair. Completion times in this stage are synthetic;
real availability traces and session deadlines are part of task 3. This stage's
simple factor averaging and staleness interpolation establish the pipeline;
the four formal baseline implementations and comparison are task 4.

The experiment settings are in `configs/point2-smoke.yaml` and
`configs/point2-pilot.yaml`. The train/validation/test fractions, client count,
Dirichlet alpha, model, LoRA settings, local steps, selection count and mode are
editable. Runs save the resolved configuration and a data-split manifest.

On a Linux CUDA server, create an isolated environment once:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-point2.txt
```

From the repository root, prepare the official Stanford Alpaca data and run the
small correctness experiment. The preparer downloads the research-only Alpaca
dataset from `tatsu-lab/stanford_alpaca` into an ignored local directory; its
data license is CC BY-NC 4.0.

```bash
python -m point2 prepare --config configs/point2-smoke.yaml
python -m point2 run --config configs/point2-smoke.yaml --gpu 1
python -m point2 run --config configs/point2-smoke.yaml --mode async --gpu 1
```

Check `nvidia-smi` and your server's GPU-use rules before choosing `--gpu`.
The `--gpu` value is the physical index reported by `nvidia-smi`; training runs
on one GPU at a time. Start with the smoke configuration, then prepare and run
`configs/point2-pilot.yaml` once the smoke run succeeds. A stopped run can be
continued using the same command with `--resume`.

Outputs are under `exp/point2-smoke/{sync,async}` or the pilot equivalent:
`config.yaml`, `data_manifest.json`, `metrics.jsonl`, `checkpoint.pt`,
`adapter.pt` and `summary.json`. Validation loss is measured during training;
the held-out Alpaca test loss is computed once at the end. These are pipeline
checks, not the brief's final MMLU evaluation. Generated data, checkpoints and
raw examples are excluded from Git.

The older `main.py` experiments remain below for comparison with existing code.

## 1. Generate the dataset

Configure the dataset split in `dataset/config.yaml`, then run the matching
generator:

```bash
cd dataset
python3 generate_<dataset>.py
cd ..
```

## 2. Configure the experiment

Edit `config.yaml`. Command-line arguments override values from that file.

- Algorithms: `fedit`, `fedasync`, `ffalora`
- Standard mode: `prototype`
- Event-driven mode: `realistic`

## 3. Run training

Standard version:

```bash
python3 main.py --alg fedit --mode prototype --epoch 1
```

Event-driven version:

```bash
python3 main.py --alg fedit --mode realistic --epoch 1
```

To run the experiments configured in `run.sh`:

```bash
bash run.sh
```

## 4. Evaluate the saved adapter

Use the same experiment arguments that were used during training:

```bash
python3 eval.py --alg fedit --dataset sst2 --model roberta-base --epoch 1
```
