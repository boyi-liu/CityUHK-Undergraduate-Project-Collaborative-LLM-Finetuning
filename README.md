# FedLLM Factory

## Grouped MMLU federated training

The active round-training setup uses MMLU in place of Alpaca-52k. It combines
the `dev` and `validation` source splits for training. The 1,816 source rows
produce 1,780 unique questions after removing 32 exact test matches and four
repeated questions. Both source splits supply training data; neither acts as
a validation holdout in this experiment.

The repository contains:

- `dataset/mmlu_train.jsonl`: complete questions, four choices, answer labels,
  original source indices, subjects, and major groups.
- `dataset/mmlu_subject_groups.json`: the mapping of 57 subjects into eight groups.
- `dataset/mmlu_train.manifest.json`: the pinned revision, file hashes, source
  counts, duplicate exclusions, and group counts.
- `dataset/MMLU-LICENSE.txt`: the source MIT license.

The eight groups are mathematics; natural sciences; computing and engineering;
medicine and health; business and economics; social sciences; history, law and
philosophy; and general knowledge. This is the project's grouping. The source
dataset does not supply these eight categories.

Preparation assigns questions to five clients using a Dirichlet distribution
with alpha 0.5 and seed 42. The full setup produces client sizes of 488, 304,
247, 232, and 509. Each client holds a different mix of the groups. Every
training question belongs to one client.

### Training settings

| Setting | Full configuration |
|---|---|
| Base model | Qwen2.5-0.5B, pinned revision |
| LoRA | Rank 8, alpha 32, dropout 0.1, `q_proj` and `v_proj` |
| Batch size | 8, without accumulation |
| Optimizer | Adam, learning rate 0.0002, no weight decay |
| Local work | One epoch per client per round; fresh optimizer state |
| Training limit | 128 tokens, right truncation, answer-only loss |
| Precision | Frozen FP32 base; FP16 CUDA autocast; loss scaling |
| Aggregation | Synchronous sample-weighted means of A and B |
| Clients | All five clients participate in each round |
| Evaluation | All 14,042 MMLU test questions, zero shot, full question context |
| Stop rule | First strict accuracy decrease, or a maximum of 10 rounds |

A round starts all clients from the same global adapter. Each client visits its
whole partition once. The server waits for all five clients, combines their
adapters, and evaluates the new global adapter before starting another round.

The 128-token limit leaves answer targets in 1,383 of the 1,780 training rows.
The other 397 rows remain assigned but add no answer loss. The output file
`token_coverage.json` reports this limit. Test questions retain their full context.

### Accuracy stopping and checkpoint selection

The initial model is round 0. After each completed round, compare its accuracy
with the preceding round on exactly the same questions. Stop if accuracy is
strictly lower. Equal accuracy allows another round. Retain the earliest adapter
with the highest observed accuracy, including round 0. Save the first lower
round too, so its result can be reviewed.

This rule uses the MMLU `test` split for checkpoint selection. It therefore acts
as selection data in this experiment. The selected score is **not an independent
final test score**. Reports and manifests record this role explicitly. A separate,
untouched evaluation set is needed for an independent final assessment.

### Verified stopping result

The full-data GPU check stopped after round 4 and retained round 3 at 48.90%.
See [the validation report](reports/mmlu-accuracy-stop.md) and its
[configuration and verification record](reports/mmlu-accuracy-stop.json).
These are checkpoint-selection scores, not independent final test scores.

### Run

Use Python 3.10 or newer on a CUDA server. Install the pinned runtime once:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-federated-lora.txt
```

Run the active full configuration from the repository root:

```bash
python -m federated_lora prepare
python -m federated_lora run --gpu 0
```

The default is `configs/federated-lora-full.yaml`. Training rows and groups are
already in the repository. The first model run loads the pinned model and MMLU
test data into local caches. The test examples and model weights stay outside Git.

For a quick execution check, use the smoke configuration. It uses 120 training
questions and 12 fixed selection questions; its scores are debug results.
The pilot uses the full data with a three-round cap.

```bash
python -m federated_lora prepare --config configs/federated-lora-smoke.yaml
python -m federated_lora run --config configs/federated-lora-smoke.yaml --gpu 0
python -m federated_lora run --gpu 0 --resume
```

Outputs include the resolved configuration, data and evaluation identities,
token coverage, client metrics, predictions and adapters for every completed
round, a resumable checkpoint, `best-adapter/`, and `summary.json`. A completed
resume does not train or evaluate the same rounds again. It checks configuration
and data identities before continuing an incomplete run.

Rebuild the committed pool from the pinned source with:

```bash
python -m federated_lora.mmlu_data --cache-dir dataset/mmlu_raw
```

Sources: [MMLU on Hugging Face](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe),
[original MMLU repository](https://github.com/hendrycks/test).

The active runner provides synchronous FedIT-style factor averaging. It does
not claim to implement all four research baselines or the proposed asynchronous
aggregation method. Earlier Alpaca assets and trace-session tools remain for
historical experiments; the commands above use grouped MMLU.


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
