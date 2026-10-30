# Training charts and availability comparisons

Checked on October 30, 2026.

## Experiment workflow

The active workflow uses the current grouped MMLU data. It does not switch back
to Alpaca. Choose one baseline with `compare --algorithm`, or supply its YAML
file with `compare --config`. The comparison runs that algorithm with all
devices available, then with recorded availability. It changes only the
availability condition and output folder. Each baseline keeps its original
synchronous or asynchronous schedule.

Each run generates availability, accuracy, local training loss, and evaluation
loss figures. PNG and SVG exports accompany CSV records. The combined figures
refresh after each evaluation. A single `run` also generates figures, but it
does not invent the missing control experiment. Use `plot` to rebuild figures
without model weights or a CUDA device.

Full configurations report before training, at hourly intervals, and at every
session deadline. Smoke and pilot configurations use 10-minute intervals.
Users can change `evaluation.interval_seconds` or set it to `null` for deadline
reports only. Test accuracy never stops a session or selects a checkpoint.

The horizontal axis adds simulated time within sessions. It excludes gaps
between sessions and actual GPU evaluation time. Both accuracy panels share
vertical limits. Deadline markers show the exact measured global model.

Training loss plots show one token-weighted answer loss per completed local
job, with separate curves for each client. They do not show a global training
loss. The CSV marks jobs whose updates the server accepted. Results from jobs
that did not finish in simulated time stay out of these plots. Batch records
retain the FedAsync proximal penalty and objective separately from answer loss.

## Verification

The isolated staged source and the GPU server each passed 92 tests. The tests
cover the four schedules,
matching conditions, periodic and deadline times, score-to-CSV values, missing
results, small-subset labels, incomplete runs, live callbacks, and restart.
They reject comparisons with different initial adapters or evaluation protocols.
Image checks verify PNG and SVG exports. Visual review checked availability,
accuracy, and loss figures for readable labels and deadline markers.

Eight real GPU runs used Qwen2.5-0.5B on a V100: one pair per algorithm. Each
run assigned 120 MMLU training examples across five clients and evaluated the
same 12 zero-shot test questions. The test session covered 06:30-06:31 on
January 30, 2020, in recorded local time. Evaluation occurred before training,
at 30 seconds, and at the 60-second deadline. Repeated participation was
enabled. Other training settings came from the corresponding smoke file.

| Algorithm | Accepted rounds/updates: always available | Accepted rounds/updates: traces |
|---|---:|---:|
| FedIT | 1 | 0 |
| FFA-LoRA | 1 | 0 |
| FLoRA | 1 | 0 |
| FedAsync-LoRA | 5 | 2 |

All eight runs passed the update replay and saved-model checks. The audits
checked downloads, availability, upload timing, factor aggregation, deadline
states, frozen backbone weights, and final-model reload predictions. Completed
restart repeated no training or evaluation. CSV accuracy values matched the
saved evaluation records. These small runs verify code behavior; they do not
estimate full MMLU performance or rank the baselines.

The final source also passed a completed-restart check on the GPU FedIT pair.
It regenerated charts without changing or rewriting any prediction file.
All server source hashes matched the local files prepared for commit. See
[the structured check record](training-charts.json).

The recorded windows leave some selected clients unavailable. Therefore, the
three synchronous methods complete no round in the trace condition. Their
charts retain the unchanged global model and still report at the deadline.
The code does not replace missing availability or remove the round barrier.

The earlier full trace experiment remains in [its original report](trace-baselines.md).
The new paired GPU checks use small subsets. They do not replace a full paired
experiment on all 1,780 training examples and 14,042 test questions.
