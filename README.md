# FedLLM Factory

## Training charts for a selected algorithm

Use the current grouped MMLU data. Choose one algorithm and run both availability
conditions with the same settings:

```bash
python -m federated_lora compare --algorithm fedit --gpu 0
# Other choices: ffa-lora, flora, fedasync-lora
python -m federated_lora compare --algorithm fedit --gpu 0 --resume
python -m federated_lora plot --algorithm fedit
```

`--algorithm` selects that algorithm's full session configuration. For custom
settings or a small test, use `--config` instead. Do not supply both options:

```bash
python -m federated_lora compare --config configs/mmlu-trace-fedit-smoke.yaml --gpu 0
```

The comparison runs the chosen algorithm twice. First, every client is available
throughout each session. Next, the recorded traces control availability. Client
partitions, seeds, initial adapters, training settings, speed profiles, upload
speed, evaluation questions, and deadlines stay the same. FedIT, FFA-LoRA, and
FLoRA keep synchronous rounds. FedAsync-LoRA keeps asynchronous updates.

The full configurations evaluate the initial model, each hour within a session,
and each session deadline. Smoke and pilot configurations use a 10-minute
interval. Set `evaluation.interval_seconds` to another positive value, or to
`null` for deadline-only evaluation. More evaluations take more GPU time.
Accuracy does not stop these runs or select a model.

Each run updates its own `plots/` folder after each evaluation and when it
returns. The comparison also refreshes the combined figures at these times.
Use `plot` to refresh them without training or loading model weights. A single
`run` produces charts for its configured condition; the other accuracy panel
states that no measured data exist. Use `compare` to obtain both conditions.
The `--simulation-only` option runs schedules without training or accuracy.

For FedIT, the combined output folder is
`exp/mmlu-trace-fedit/comparison/plots/`. Other algorithms use their own output
folder. Figures use PNG and SVG formats:

| Figure | Content |
|---|---|
| `device-availability` | Recorded availability for each device and session |
| `accuracy-always` | Accuracy over time with all devices available |
| `accuracy-trace` | Accuracy over time with recorded availability |
| `training-loss-always`, `training-loss-trace` | One loss curve per client; each point is a completed local job |
| `evaluation-loss-always`, `evaluation-loss-trace` | MMLU answer loss at the same evaluation times |

`accuracy.csv` contains the measured points. `deadline-comparison.csv` contains
deadline scores, changes from the initial model, and gaps to the always-available
control in percentage points. `training-loss.csv` marks whether the server has
accepted each completed job. `batch-loss.csv` contains its batch losses. The
FedAsync records distinguish answer loss, proximal penalty, and total objective.
Loss curves show answer loss so the meaning stays the same across algorithms.
They do not represent one global training loss: clients hold different data.
Jobs that never finish in simulated time do not enter the loss plots, even if
the simulator computed their local results in advance.

The horizontal axis adds elapsed time within sessions. It excludes gaps between
sessions and GPU evaluation time. Dashed lines mark deadlines; diamonds mark
deadline scores. Lines connect measured points and do not add measurements.
Both accuracy panels use the same vertical limits. The labels identify small
test subsets and incomplete runs. Missing runs produce empty panels, not scores.
The plot loader rejects mismatched data, initial adapters, evaluation protocols,
and evaluation times. GPU execution seconds remain in the training records.

See [the chart verification report](reports/training-charts.md) for test scope.

## Trace-driven training and deadline evaluation

The default experiment follows recorded device availability and fixed session
deadlines. It uses `configs/mmlu-trace-fedasync-lora.yaml`. It keeps the current
grouped MMLU training pool, as requested. The global model continues from one
session to the next. Accuracy does not stop training or choose a checkpoint.

```bash
python -m federated_lora prepare
python -m federated_lora schedule
python -m federated_lora dry-run
python -m federated_lora run --gpu 0
python -m federated_lora run --gpu 0 --resume
```

The four trace configurations use their original update schedules:

| Configuration | Algorithm | Schedule |
|---|---|---|
| `configs/mmlu-trace-fedit.yaml` | FedIT | Synchronous rounds |
| `configs/mmlu-trace-ffa-lora.yaml` | FFA-LoRA | Synchronous rounds; fixed A |
| `configs/mmlu-trace-flora.yaml` | FLoRA | Synchronous rounds; stacked products |
| `configs/mmlu-trace-fedasync-lora.yaml` | FedAsync-LoRA | Immediate asynchronous mixing |

Each configuration also has `-smoke.yaml` and `-pilot.yaml` versions. The full
version covers three overnight sessions, 22:00-07:00, starting January 30, 2020
in recorded local time. Smoke and pilot versions use two short morning sessions,
06:30-06:50. Smoke checks use 120 training rows and 12 test questions. Pilots use
the full training pool and all 14,042 test questions. Their scores have different
scopes and must not be combined.

Five clients map to trace indices `[0, 1, 3, 4, 7]`. Availability requires
`(screen_off OR screen_locked) AND charging AND wifi`. Each completed upload must
arrive while that client remains available and no later than the session
deadline. Unfinished jobs and uploads are discarded. Incomplete synchronous
rounds are also discarded at the deadline. A session with no accepted update
still produces a deadline report. Missing trace coverage causes an error.

The runner measures zero-shot MMLU at the initial state, configured intervals,
and every deadline.
It saves each exact deadline adapter and the final adapter. It does not save a
test-selected `best-adapter`. The earlier test set has already been used in
other experiments, so these scores do not become independent final test scores.

The repository supplies the device traces and step-time estimates. Their
external collection source, recorded timezone, and measurement setup remain
unverified. Timing estimates scale with each client's epoch size. Network cost
currently includes uploads only. FFA-LoRA uploads B only. FLoRA upload sizes use
local ranks, rather than the growing rank of its portable checkpoint.

All four trace runs start with PEFT Kaiming A and zero B. This differs from the
earlier FedIT-style uniform A initialization. FedAsync uses proximal SGD; the
three synchronous methods retain Adam. These choices are recorded in manifests.

```bash
python -m federated_lora run --config configs/mmlu-trace-flora-smoke.yaml --gpu 0
python -m federated_lora.trace_audit --config configs/mmlu-trace-fedasync-lora.yaml
```

See [the trace protocol and verification report](reports/trace-baselines.md).
The standalone checks below retain their earlier accuracy stopping rule.

## Standalone grouped MMLU checks

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
python -m federated_lora prepare --config configs/federated-lora-full.yaml
python -m federated_lora run --config configs/federated-lora-full.yaml --gpu 0
```

This standalone example uses `configs/federated-lora-full.yaml`. Training rows and groups are
already in the repository. The first model run loads the pinned model and MMLU
test data into local caches. The test examples and model weights stay outside Git.

For a quick execution check, use the smoke configuration. It uses 120 training
questions and 12 fixed selection questions; its scores are debug results.
The pilot uses the full data with a three-round cap.

```bash
python -m federated_lora prepare --config configs/federated-lora-smoke.yaml
python -m federated_lora run --config configs/federated-lora-smoke.yaml --gpu 0
python -m federated_lora run --config configs/federated-lora-full.yaml --gpu 0 --resume
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

The active round runner supports three synchronous methods:

| Configuration | Method | Server rule |
|---|---|---|
| `configs/federated-lora-full.yaml` | FedIT-style | Average A and B separately, with sample weights |
| `configs/ffa-lora.yaml` | FFA-LoRA | Keep the shared A fixed; train and average B only |
| `configs/flora.yaml` | FLoRA | Stack weighted A factors and unweighted B factors; merge before fresh local training |

```bash
python -m federated_lora run --config configs/ffa-lora.yaml --gpu 0
python -m federated_lora run --config configs/flora.yaml --gpu 1
```

Each new method has a `-smoke.yaml` configuration. It uses 120 training rows and
12 selection questions. Use `python -m federated_lora.baseline_audit --config
configs/flora.yaml` after a run to check the products, saved model, and resume.

Both methods use the grouped MMLU pool, five clients, batch size 8, one local
epoch, rank 8, alpha 32, and the same accuracy stop rule. FFA-LoRA and FLoRA use
PEFT's Kaiming initialization for A and zero initialization for B. The older
FedIT-style run used uniform A values from -0.01 to 0.01. This difference is
recorded in each manifest. These experiments apply the reference algorithms
with the project's settings. They do not reproduce the papers' result tables.
FFA-LoRA runs without differential privacy; it makes no privacy guarantee.

FLoRA starts a new local adapter each round, on the preceding global model.
It keeps local rank 8. The rank of each five-client round stack is 40. Saved
checkpoints retain all completed products as one cumulative adapter on the
original base. Their rank grows by 40 per round. This form permits a standard
PEFT reload and exact resume without an SVD or rank truncation. It is a storage
form, not a rule that increases the local training rank. The runtime preserves
alpha/r when it changes the saved rank. See [the method notes and GPU checks](reports/lora-baselines.md).

For FLoRA only, `federation.client_ranks` can set one rank per client, for example
`[4, 8, 12, 8, 4]`. Each client uses the same alpha/r ratio as the configured
reference rank. Upload files contain local A/B factors. FFA-LoRA upload files
contain B only; full model exports also retain the shared A for later loading.

These configurations retain the standalone round checks. The new trace
configurations also run these methods with their synchronous schedules.
The proposed asynchronous method remains separate work. Earlier Alpaca assets and trace-session tools
remain for historical experiments; the commands above use grouped MMLU.

### FedAsync-LoRA standalone checks

FedAsync-LoRA uses the asynchronous rule from
[Asynchronous Federated Optimization](https://arxiv.org/abs/1903.03934v5).
The original paper trains full models. This project applies its rule to the
vector containing both LoRA factors. The base model stays frozen.

Each client downloads a global A/B pair and its server version. The client
trains with plain SGD on answer loss plus `rho/2 * (||A-A_download||^2 +
||B-B_download||^2)`. The penalty uses a sum, not a mean over parameters.
The anchor stays equal to the downloaded pair for the whole local job.

The server accepts one client model at a time. It computes staleness as the
number of accepted updates since that client downloaded its model. It sets
`beta = mixing_alpha * s(staleness)` and applies `(1-beta) * global + beta *
client` to both factors. It uses the same beta for A and B. It does not add
sample weights, average a round, or replace this rule with a product correction.
This baseline retains the product cross terms that the proposed method will
later address.

The new configurations use SGD without momentum, a learning rate of 0.0002,
rho 0.005, mixing alpha 0.6, and polynomial decay with exponent 0.5. Model,
data, batch size 8, rank 8, LoRA alpha 32, and one local epoch follow the current
project setup. The optimizer changes from Adam to SGD to follow Algorithm 1.
These settings do not reproduce the paper's CIFAR-10 or WikiText-2 experiments.

| Configuration | Simulation | Limit |
|---|---|---|
| `configs/fedasync-lora.yaml` | Uniform stale-version sampling, as in the paper | 50 accepted updates |
| `configs/fedasync-lora-pilot.yaml` | Full training data and full MMLU | 5 accepted updates |
| `configs/fedasync-lora-smoke.yaml` | 120 training rows, 12 selection questions | 10 accepted updates |
| `configs/fedasync-lora-arrivals-smoke.yaml` | Synthetic client completion events | 10 accepted updates |

The paper-style schedule samples a real saved version from the last zero to
four accepted updates. It trains from that exact saved model. The event schedule
starts one job on each client. Each arrival updates the server immediately and
lets that client start another job from the new model. Other jobs keep their
older downloads. Ties use job order. Neither schedule uses a round barrier.

The event times in these standalone configurations are simulated durations,
not measured GPU times or device traces. Their event example uses always-available
clients. The trace configurations above now connect this baseline to the recorded
windows. The decay can also use the paper's
`constant` or `hinge` function through `federation.staleness`.

Evaluate the initial model and every accepted update with zero-shot MMLU.
Stop at the first strict accuracy decrease, or at the update limit. Keep the
earliest highest score, including update zero. This remains test-based model
selection. The scores are not independent final test results.

```bash
python -m federated_lora run --config configs/fedasync-lora-smoke.yaml --gpu 0
python -m federated_lora run --config configs/fedasync-lora-pilot.yaml --gpu 0
python -m federated_lora run --config configs/fedasync-lora.yaml --gpu 0
python -m federated_lora run --config configs/fedasync-lora.yaml --gpu 0 --resume
python -m federated_lora.fedasync_audit --config configs/fedasync-lora-pilot.yaml
```

Each checkpoint saves the current and selected models, accepted update count,
pending downloads, event order, historical versions, and scheduler random state.
A resumed job uses the same downloaded model and shuffle seed. A completed run
starts no more jobs. See [the reference notes and checks](reports/fedasync-lora.md).


### Historical Alpaca availability tools

The older tools below use factor-averaging and interpolation prototypes. They
remain available for historical experiments. Use the `mmlu-trace-*` configurations
above for the four reference baseline rules and fixed deadline reporting.

The session experiment compares four runs: always-available sync/async and
trace-based sync/async. Both conditions use the same initial adapter seed, client
partitions, LoRA settings, device timing profiles, upload bandwidth, capacity,
sessions, and MMLU protocol. Only availability and federation mode differ.

`configs/federated-lora-sessions.yaml` assigns all 52,002 Alpaca rows and evaluates
full MMLU (14,042 test questions). `configs/federated-lora-sessions-smoke.yaml`
uses 120 Alpaca rows and a fixed 12-question MMLU subset for execution checks.
Both trace configurations use **one local epoch per job** and an effective
batch size of **32** (`micro_batch_size: 1`, `grad_accum: 32`). Each epoch visits
each usable local row once, in a shuffled order. The last batch can be smaller;
the trainer does not repeat rows to fill it. It weights accumulated losses by
valid response-token counts, including the final short batch. A client with N
usable rows needs `ceil(N / 32)` optimizer steps for one epoch. Larger clients
therefore need more steps. Each new job starts a new shuffled local epoch.
The smoke configuration keeps its smaller context and data limit; its scores
are debug checks. LoRA uses rank 8, alpha 16, dropout 0.05, and `all-linear`.
PEFT adds adapters to attention and feed-forward linear layers. It excludes the
output head and keeps the backbone frozen. For Qwen2.5-0.5B, these targets are
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj. They contain
4,399,104 trainable adapter parameters, or 17,596,416 bytes in float32.
Training settings differ from older runs. Configured output directories now end
in `-epoch`. Do not resume old checkpoints or reuse their scores.

Both conditions map ten clients to trace indices
`[0, 1, 3, 4, 7, 8, 9, 11, 13, 14]` and use three consecutive overnight sessions
starting January 30, 2020, 22:00-07:00, in the recorded local time. These are the
first ten records covering the whole experiment, not the ten most available
records. The original trace collection source and timezone remain unverified.
The always-available control retains the same mapping and metadata, but replaces
each client's windows with one full window per session. Both conditions stop
training outside sessions. The global adapter continues across deadlines.

#### Change from the original async code: repeated local training

The original async code allowed one successful upload per availability window.
The comparison now uses `sessions.participation: repeated` in **both availability
conditions**. Each job starts from the current global adapter, completes one
local epoch, then uploads its LoRA update. Async applies the update with the
existing staleness interpolation and the client can start another job while
eligible and capacity is available. A returning client joins behind clients
waiting for capacity, so the first five clients cannot permanently hold all five
slots in the always-available condition. This queue order is deterministic and
saved in checkpoints. A client has only one job or upload in progress at a time.

Sync retains its existing round rule: randomly select five clients from all ten,
wait for every selected upload, aggregate, then start another round from the
latest global adapter. A client that has uploaded waits for its round; it does
not submit multiple updates into the same round. There is no round timeout or
partial-round aggregation. Incomplete rounds are discarded at the deadline.
Sparse traces can therefore still produce zero sync global updates.

The original `once_per_window` option remains available for historical async
reproduction, but `compare` requires `repeated`. It preserves the configured
epoch and batch settings. Old smoke scores from the one-upload policy are not
reused for this comparison.

#### Preserved simulation rules

- Eligibility is `(screen_off OR screen_locked) AND charging AND wifi`.
  Unknown required states are ineligible. Battery percentage is ignored.
- Original windows shorter than 60 seconds are excluded before session clipping.
  State persists between changes only within each record's observed coverage.
  Invalid or missing trace coverage is an error, including for the matched control.
- A job is skipped if its training cannot fit the known remaining availability
  window. This uses future trace information, not a learned availability predictor.
  A skipped async assignment retains its slot until window closure, as before.
- Losing availability or reaching a deadline discards unfinished training or
  uploads. Local jobs do not resume in a later window. Already uploaded sync
  updates remain buffered until the round completes or the deadline discards it.
- Uploads completing exactly at window closure or the deadline count before
  closure and evaluation. Later uploads cannot change that snapshot.
- Simulated durations retain the inherited Jetson step-time ranges and 1 Mbps
  upload default, using float32 adapter tensor bytes. These are unverified device
  timing assumptions. Download time is not modeled. Actual GPU training time is
  logged separately; it is not the plotted time axis. Training duration uses each
  client's epoch step count, not a fixed 20 steps. Actual runs count usable encoded
  rows. Dry runs count prepared rows and need prepared client data first. The old
  per-step ranges have not been measured with batch size 32 and all-linear LoRA.
  Treat them as simulation assumptions until measured. They do not yet account
  for sequence length, accumulation cost, or the last batch size.

#### Periodic MMLU evaluation

`evaluation.interval_seconds: 3600` evaluates the global adapter every simulated
hour within a session, plus the initial model and every session deadline. Use a
positive interval in seconds or `null` for deadline-only evaluation. The deadline
is evaluated once even when it coincides with the interval. Quiet sessions still
produce evaluation points. Valid arrivals at a timestamp are aggregated before
its evaluation. Evaluation does not cancel pending client jobs or advance the
simulated device clock. Save/resume preserves pending jobs and evaluation points.

MMLU uses a pinned revision of [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
and the same fixed questions and prompts at every point. Up to five
subject-specific development demonstrations are used, with fewer when needed to
fit context. The scorer sums conditional log probabilities of complete A/B/C/D
answer tokens, including multi-token answers. Overall accuracy is weighted by
question count. Full MMLU is used for reported curves and deadline scores. Debug
subset curves are labeled explicitly. Do not tune settings on test scores.

#### Run the comparison

Install the CUDA environment and requirements as described above; Matplotlib is
included for headless plotting. Check server GPU-use rules before choosing a GPU.
First prepare the client data. Then check all four schedules without a GPU or model downloads:

```bash
python -m federated_lora prepare --config configs/federated-lora-sessions.yaml
python -m federated_lora compare --config configs/federated-lora-sessions.yaml --simulation-only
```

This produces scheduling results and an availability plot. The two accuracy
panels explicitly state that measured MMLU results are missing; no scores are
invented or taken from dry runs.

Run the debug comparison, then the full experiment:

```bash
python -m federated_lora prepare --config configs/federated-lora-sessions-smoke.yaml
python -m federated_lora compare --config configs/federated-lora-sessions-smoke.yaml --gpu 1
python -m federated_lora prepare --config configs/federated-lora-sessions.yaml
python -m federated_lora compare --config configs/federated-lora-sessions.yaml --gpu 1
```

`compare` runs always/sync, always/async, trace/sync, and trace/async sequentially
on one physical GPU. Each run starts from the same initial adapter and carries
its own adapter across the three sessions. Full hourly MMLU evaluation is costly:
each run has 28 evaluations (initial, 24 hourly interior points, 3 deadlines).

To resume an interrupted comparison, add `--resume` to its command. Completed
runs are checked and resumed without further training; missing runs start fresh.
Checkpoint format 2 and run fingerprints reject older or changed settings,
traces, client data, initial adapters, and evaluation protocols. Legacy output
folders are preserved. Use a new `output_dir` for a different experiment.

To run just one condition/mode, or regenerate figures:

```bash
python -m federated_lora run --config configs/federated-lora-sessions.yaml --availability trace --mode async --gpu 1
python -m federated_lora schedule --config configs/federated-lora-sessions.yaml --availability always
python -m federated_lora dry-run --config configs/federated-lora-sessions.yaml --availability always --mode async
python -m federated_lora plot --config configs/federated-lora-sessions.yaml
```

#### Outputs and the three plots

Measured runs are saved under `exp/federated-lora-sessions-epoch/{always,trace}/{sync,async}`.
Scheduling-only runs are under `dry-run/{always,trace}/{sync,async}` and contain
no accuracy. Schedules alone are under `schedule/{always,trace}`.

Each measured run saves the resolved configuration and source identities in
`run_manifest.json`, the source/session manifest, `availability.svg`, event log,
checkpoint, initial/periodic/deadline adapters, predictions, evaluation reports,
and `summary.json`. The summary contains every time-stamped evaluation and the
per-session training/upload/update counters. Periodic snapshot filenames include
exact microsecond timestamps. The initial adapter hash and MMLU identity must
match before the plotting code combines runs.

`plots/` contains three figures, each exported as PNG and SVG:

1. `device-availability`: one device timeline per overnight session, with clock
   offsets, unavailable intervals, and deadline markers.
2. `accuracy-always`: measured sync and async MMLU accuracy with all clients
   available throughout each session.
3. `accuracy-trace`: measured sync and async MMLU accuracy with recorded availability.

Both accuracy figures use the same y limits. Their x axis is cumulative simulated
session time, 0-27 hours, excluding the daytime gaps and physical GPU evaluation
time. Vertical lines mark deadlines at 9, 18, and 27 hours; diamonds mark their
measured scores. Lines connect evaluated points and do not imply continuous
measurement. `accuracy.csv` exports those points. `deadline-comparison.csv`
exports scores, changes from the initial model, gaps to the always-available
control in percentage points, and work counters. `plot_manifest.json` states
whether all four measured runs are present. Missing runs produce clearly labeled
empty panels; mismatched or incomplete saved results are rejected.

The aggregation rules remain preliminary factor averaging and staleness
interpolation. This comparison does not implement the four named baselines or
product-consistent aggregation from the later project tasks.

New MMLU runs use zero-shot evaluation. The session configurations set
`evaluation.n_shot: 0`. Keep historical scores with their original settings.
