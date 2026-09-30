# FedLLM Factory

## Federated LoRA pipeline

`federated_lora` simulates separate clients on one CUDA GPU. It prepares disjoint,
task-skewed Alpaca partitions, runs synchronous rounds or asynchronous completion
events, and maintains a global LoRA A/B pair on the server. It supports synthetic
completion times or device-trace availability with session deadlines and MMLU
evaluation. The runner's current factor averaging and staleness interpolation establish
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
not evaluation scores. The trace-session runner below adds MMLU evaluation.
Generated data, checkpoints and raw examples are excluded from Git.

### Device availability and session deadlines

`configs/federated-lora-sessions-smoke.yaml` uses 120 Alpaca rows, ten clients,
two local optimizer steps per job, and the same fixed 12 MMLU questions at every
deadline. These scores only check execution. `configs/federated-lora-sessions.yaml`
uses all 52,002 Alpaca rows and the full 14,042-question MMLU test split.

Both configurations map clients to distinct trace indices
`[0, 1, 3, 4, 7, 8, 9, 11, 13, 14]` in `dataset/device_usage_traces.json`.
They cover three consecutive overnight sessions starting January 30, 2020,
22:00–07:00, in the timestamps' recorded local time. These are the first ten
records covering the whole experiment, not the ten most available devices.
The trace file was present in the original repository; its external collection
source and timezone are unverified. Missing coverage is an error, not an
always-available fallback. The global adapter continues across session deadlines.

The preserved simulation rules are:

- Availability requires `(screen_off OR screen_locked) AND charging AND wifi`.
  Unknown states are ineligible; battery percentage does not change eligibility.
- Original availability windows shorter than 60 seconds are excluded before
  clipping them to sessions. State persists between recorded changes, only
  within each record's observed coverage.
- A job is skipped if its training cannot fit the known remaining window.
  This uses future trace knowledge, not a learned availability predictor.
- Sync selects clients randomly from all clients and waits for every selected
  upload. Incomplete rounds are discarded at the deadline; there is no round
  timeout. Sparse availability can therefore produce zero global updates.
- Async accepts at most one successful upload per original availability window.
  An assigned job that cannot fit retains its slot until that window closes.
- Unfinished training or upload is discarded on window closure or session end.
  The next job starts from the current global adapter without local resumption.
- Simulated durations use the inherited Jetson step-time ranges and configured
  bandwidth (default 1 Mbps), with float32 adapter tensor bytes as upload size.
  They are assumptions, not measurements of these devices or the training GPU.
  Download time is not modeled. Actual GPU execution time is logged separately.
- An upload completing exactly at window closure or the session deadline counts
  before closure/evaluation. Later completions cannot change that snapshot.

Inspect the schedule or exercise both modes without a GPU or model downloads:

```bash
python -m federated_lora schedule --config configs/federated-lora-sessions-smoke.yaml
python -m federated_lora dry-run --config configs/federated-lora-sessions-smoke.yaml --mode sync
python -m federated_lora dry-run --config configs/federated-lora-sessions-smoke.yaml --mode async
```

Run actual training and deadline evaluation on a CUDA GPU:

```bash
python -m federated_lora prepare --config configs/federated-lora-sessions-smoke.yaml
python -m federated_lora run --config configs/federated-lora-sessions-smoke.yaml --mode sync --gpu 1
python -m federated_lora run --config configs/federated-lora-sessions-smoke.yaml --mode async --gpu 1
```

MMLU is loaded from a pinned revision of
[cais/mmlu](https://huggingface.co/datasets/cais/mmlu).
Following the [original evaluator](https://github.com/hendrycks/test/blob/master/evaluate.py),
prompts use up to five subject-specific development examples, reducing this count
when needed to fit context. Accuracy is weighted by question count. The scorer
compares conditional log probabilities of the complete A/B/C/D answer tokens,
including cases where a tokenizer splits an answer into multiple tokens.
The same questions are evaluated before training and at each session deadline;
evaluation does not advance simulated time. Do not tune settings on test scores.

Outputs under `exp/federated-lora-sessions-smoke/{sync,async}` include the resolved
configuration and data/evaluation identity in `run_manifest.json`,
`session_manifest.json`, `availability.svg`, `events.jsonl`, `checkpoint.pt`,
initial and per-session adapters/predictions/evaluation reports, and `summary.json`.
Scheduling-only outputs are under `dry-run/{sync,async}` and contain no accuracy.
Add `--resume` to continue an interrupted program with its saved pending events,
global model, and client jobs. This program recovery is separate from the policy
of discarding a simulated device's interrupted local job. Resume rejects changed
configuration, traces, client data, or evaluation protocol.
