# Recorded availability and session deadlines

This record describes the earlier deadline-only checks. The default configurations
now also report at fixed intervals and generate charts. See
[the chart workflow and checks](training-charts.md). The scores below are unchanged.

## Current protocol

The main experiment now uses the existing device traces to control client
availability. It keeps the user's current MMLU training pool and grouping.
The earlier standalone runs remain unchanged as historical experiments.

The default is `configs/mmlu-trace-fedasync-lora.yaml`. Separate configurations
run FedIT, FFA-LoRA, and FLoRA on the same trace map. Those three methods retain
synchronous rounds. FedAsync-LoRA retains its proximal local SGD and immediate
server mixing. This change connects the baseline rules to device availability;
it does not add asynchronous variants of the synchronous baselines.

The five clients use trace indices `[0, 1, 3, 4, 7]`. These are the first five
entries of the earlier ten-device map. The mapping does not select devices for
high availability. The full configuration covers three overnight sessions,
22:00-07:00, starting January 30, 2020. Smoke and pilot configurations cover
two short morning sessions, 06:30-06:50, starting on the same date. Times use
the timestamps as recorded; the external timezone remains unverified.

The source file is `dataset/device_usage_traces.json`. It has 20 device records
and 20,155 events. It was present in the original repository. Its external
collection source remains unverified. This implementation uses its recorded
events without claiming verified provenance for the collection.

## Work and update rules

- A device can train when `(screen_off OR screen_locked) AND charging AND wifi`.
- Short eligible windows below 60 seconds are removed, as in the earlier parser.
- Each client keeps its private, disjoint MMLU partition.
- Each local job visits that client's rows for one epoch, with batch size eight.
- The full and pilot configurations allow repeated jobs while a device remains
  available. The asynchronous smoke run uses the earlier once-per-window policy.
  Synchronous smoke runs retain their round barrier and round restart rule.
- The inherited scheduler checks whether training fits the known window. This
  uses future trace information. It is not a learned availability predictor.
- Training time uses the inherited AGX/Nano per-step estimates and the assigned
  epoch size. The estimate counts all batches, including any with no answer loss.
- The simulated network charges for uploads only. Download costs remain outside
  this protocol. Upload size comes from the actual local factor payload.
- FFA-LoRA sends B only. FLoRA sends its local factors at the local rank. Its
  cumulative portable checkpoint rank does not determine the upload size.
- An upload must finish during availability and no later than the deadline.
  Uploads exactly at the boundary count. Later uploads do not change the server.
- Interrupted work is discarded. The implementation does not resume a client's
  unfinished local work in a later availability window.

Synchronous methods select from all clients, including unavailable clients, as
in the existing engine. The default selects all five. They wait for all selected
uploads before applying a global update. An incomplete round does not change
the global model. It is discarded at the deadline. FedAsync-LoRA instead applies
each valid upload immediately. Its staleness is the number of accepted server
updates since the client downloaded its starting pair.

All four new trace runs use the same PEFT Kaiming A initialization and zero B.
The earlier standalone FedIT-style run used uniform A from -0.01 to 0.01.
FedIT, FFA-LoRA, and FLoRA use Adam at 0.0002. FedAsync-LoRA uses plain SGD at
0.0002 with proximal rho 0.005, mixing alpha 0.6, and polynomial exponent 0.5.
The model is pinned Qwen2.5-0.5B. Rank eight, alpha 32, dropout 0.1, q_proj/v_proj,
128 training tokens, and the existing eight MMLU groups remain in use.
The token limit leaves answer targets in 1,383 of the 1,780 assigned rows. The
remaining rows stay assigned but supply no answer loss, as in the earlier runs.

## Deadline evaluation and recovery

The run stops at the last configured session deadline. It does not stop when
accuracy decreases. Test accuracy does not choose a model. It reports the current
global model at each deadline and continues that model into the next session.
A quiet session still produces a report. Optional intermediate reports do not
change the clock or choose a checkpoint.

The runner saves an initial adapter, one adapter per deadline, and a final
adapter. There is no `best-adapter` export. Each report uses zero-shot MMLU.
The full and pilot configurations score all 14,042 complete test questions.
Smoke checks score 12 fixed questions and remain debug results. These test
questions have been used in earlier experiments, so the new scores are still
not independent final test scores.

The checkpoint preserves the event queue, pending results, current global
pair, accepted version, selected synchronous clients, buffered uploads, and
scheduler random state. Local GPU work is computed when the simulated job
starts. Its result stays pending until its simulated upload arrives. A later
window close or deadline can discard that result. This represents isolated
client jobs on one GPU; simulated time differs from GPU elapsed time.

After process recovery, pending results retain their original downloaded
versions. They do not restart from a later global model. This process recovery
does not change the rule that device interruptions discard unfinished work.
A completed resume performs no more training or evaluation.

## Local verification and availability findings

All 86 local tests passed. New tests exercise all four rules through the trace
callbacks. They check stale uploads, synchronous barriers, B-only uploads,
declining accuracy without early stopping, exact deadline snapshots, quiet
sessions, and recovery with pending jobs. Older boundary tests check uploads
at and after the deadline, window close, interruption, and trace coverage.

The CPU dry run for three full nights predicts:

| Algorithm | Night 1 updates | Night 2 updates | Night 3 updates |
|---|---:|---:|---:|
| FedIT | 0 | 0 | 0 |
| FFA-LoRA | 0 | 0 | 0 |
| FLoRA | 0 | 0 | 0 |
| FedAsync-LoRA | 17 | 26 | 50 |

These are schedule counts, not accuracy results. Some selected devices have
no usable overnight window. Since the synchronous configurations require all
five clients, their rounds remain incomplete. The simulator does not replace
missing availability or drop the barrier to make those methods progress.

All four baseline paths passed GPU smoke checks on two recorded morning
sessions. Those checks used 120 training rows and 12 fixed MMLU questions.
The synchronous methods produced no accepted round because some selected
clients were unavailable. They still trained eligible clients and produced
both deadline reports. FedAsync accepted four uploads. All smoke deadline
scores were 6/12. These are debug results, not full MMLU accuracy estimates.

The full FedAsync run assigned all 1,780 training rows across five clients.
It covered three recorded overnight sessions and scored all 14,042 MMLU test
questions at each deadline. Actual training depended on availability. It accepted
93 updates in total. The observed counts matched the CPU schedule exactly.
It continued through all three deadlines. There was no accuracy-based stop
and no selection of a highest-scoring model.

Clients 0, 2, 3, and 4 trained during this full run. Their partitions hold 1,476
assigned rows. Client 1 had no window long enough for a full local epoch. Its
data stayed private and was not reassigned to another client.

| Snapshot | Accepted updates in session | Correct / 14,042 | Accuracy |
|---|---:|---:|---:|
| Initial | 0 | 6,568 | 46.77% |
| Night 1 deadline | 17 | 6,553 | 46.67% |
| Night 2 deadline | 26 | 6,688 | 47.63% |
| Night 3 deadline | 50 | 6,555 | 46.68% |

Each deadline used the exact global state after the last valid upload at or
before that deadline. The audit replayed every accepted update and verified
its original download, availability window, upload size, completion time,
staleness, mixing coefficient, and resulting factors. The final PEFT export
reproduced every prediction and score from the last deadline after a fresh
load. Frozen-base checks passed. Completed resume repeated no events or scores.

These results verify the trace-driven FedAsync path. Full overnight GPU
benchmarks for the three synchronous methods were not run; their zero-update
entries above remain CPU scheduling predictions. Their baseline equations were
already checked in the earlier training work, and the new local tests exercise
their trace aggregation callbacks with controlled windows.

See [the structured verification record](trace-baselines.json) for settings,
counts, time assumptions, availability totals, source hashes, and audit results.
Raw predictions, checkpoints, and adapters remain outside Git.
