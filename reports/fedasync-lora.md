# FedAsync-LoRA implementation and checks

Later trace integration is documented in [the deadline report](trace-baselines.md).
The results below retain the earlier standalone simulation and stopping policy.

## References and scope

The method follows Algorithm 1 and Section 5.2 of
[Asynchronous Federated Optimization, version 5](https://arxiv.org/abs/1903.03934v5)
by Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. The project brief calls this
the 2019 reference. Version 5 contains the later revision from 2020.
The check also used the authors' code at revision
[`912ecf33e0fb811ac8551c4f3cdab344cdedfb53`](https://github.com/congxie1108/async_fl/tree/912ecf33e0fb811ac8551c4f3cdab344cdedfb53).

The paper uses a regularized local objective, SGD, and immediate server mixing
after each received model. It includes constant, polynomial, and hinge weights.
Its experiments simulate asynchronous updates through uniformly sampled stale
versions. The authors' code also stores earlier model versions and trains from
the selected version. Its command-line rho is learning rate times the paper's
rho. This implementation exposes the objective coefficient rho directly.

The paper does not define a LoRA algorithm. This project defines FedAsync-LoRA
by setting x to the vector of all A and B entries. The pretrained base stays
frozen. This is an explicit adaptation. It does not claim the paper's reported
scores or convergence guarantees for a Qwen model with LoRA factors.

## Update rules

For one client job, let theta_download contain the A/B values at download.
Use plain SGD on:

`answer_loss(theta) + rho/2 * sum((theta - theta_download)^2)`

The anchor remains fixed during the job. The penalty sums all parameter entries.
It does not pull factors to zero or to a later server model. The client keeps
the same rank and trains both factors. It visits its partition once per local
epoch. Each job starts with a fresh optimizer. The project keeps shuffled local
epochs; it does not copy the paper's dataset size or local iteration count.

If the current server version is v and the downloaded version is tau, set age
to v-tau. This counts updates already accepted after the download. A fresh
download has age zero. Set beta to mixing_alpha times s(age):

- Constant: s(age) = 1.
- Polynomial: s(age) = (age+1)^(-a).
- Hinge: s(age) = 1 for age <= b; otherwise 1/(a*(age-b)+1).

Set each global factor to `(1-beta)*global_factor + beta*client_factor`.
The server uses the current global pair and the complete client pair. It does
not add a difference from the stale download. It uses one beta for A and B.
It does not multiply beta by a client's share of the dataset. These choices
follow FedAsync on the concatenated factor vector.

Factor mixing can introduce cross terms in BA. This baseline intentionally
retains them. It does not add separate factor decays, a product correction,
an SVD, or a rank change. Those would change the baseline or enter the proposed
research method. FedIT, FFA-LoRA, and FLoRA keep their synchronous schedules.

## Simulation and recovery

The default `uniform_staleness` schedule samples an age uniformly from zero to
min(v,4), then selects a client uniformly. It uses the exact saved adapter at
version v-age. It retains the last five versions. Startup never uses a future
or missing model. This matches the paper's experimental approach, rather than
claiming that one sequential GPU runs five simultaneous training processes.

The optional `arrivals` schedule represents five overlapping client jobs with
one GPU. Each job stores its downloaded adapter, version, seed index, and finish
time. The engine computes the local result when it processes that arrival.
This defers GPU work without changing the isolated client's starting model.
After one arrival, the server applies it immediately. That client can start
again while the other clients retain older downloads. Equal finish times use
job identifiers as a fixed order. A fast client can update more than once before
a slow client finishes. There is no round barrier.

The event durations are synthetic. They do not come from the device traces or
the measured GPU run times. Clients in this example remain available. Connecting
this baseline to trace sessions remains separate work, as for the two previously
added synchronous baselines.

The checkpoint saves accepted updates atomically. It includes pending adapter
downloads, event order, version history, and the scheduler random state. Each
local job uses seed 42 + 1000*(job_id+1) + client_id. Recovery reruns an unfinished
job from the same download with the same shuffle order. It does not copy that
job onto the latest model. A completed resume launches no further updates.

## Project settings

| Setting | Value |
|---|---|
| Model | Pinned Qwen2.5-0.5B base |
| Training data | 1,780 clean MMLU dev/validation questions |
| Partitions | Eight major groups; five clients; Dirichlet alpha 0.5; seed 42 |
| Client sizes | 488, 304, 247, 232, 509 |
| LoRA | Rank 8; alpha 32; dropout 0.1; q_proj and v_proj |
| Initialization | PEFT Kaiming A and zero B |
| Local optimizer | SGD; learning rate 0.0002; no momentum or weight decay |
| Proximal rho | 0.005, applied to the summed factor distance |
| Local work | One epoch; batch 8; no accumulation |
| Context | 128 training tokens; answer-only loss |
| Server mix | Alpha 0.6; polynomial exponent 0.5 |
| Paper-style delay | Uniform age from 0 to min(version,4) |
| Event example | Client jobs take 1, 1.5, 2, 2.5, and 3 simulated seconds |
| Evaluation | Zero shot; all 14,042 complete MMLU test questions |
| Pilot budget | Five accepted updates; a strict decrease can stop it earlier |
| Full configuration | Maximum of 50 accepted updates |

The 128-token limit leaves answer targets in 1,383 training rows. The other
397 rows remain assigned but supply no answer loss. Each accepted update comes
from one client. Five updates do not mean that each client has trained once.

The new method changes Adam to SGD to follow the paper. The model and data
settings otherwise follow the current project. The penalty and mixing values
are editable examples, not tuned optima. The paper uses different models,
datasets, learning rates, and training budgets.

The user requested accuracy-based stopping in the earlier training work. This
runner applies it after every accepted update. It retains the earliest highest
score, including update zero. Test data therefore serve as selection data.
The selected scores are not independent final test scores.

## Verification

All 80 local tests passed. Nine FedAsync tests also passed in the server runtime.
They check the three decay functions, hinge boundary, parameter interpolation,
proximal gradients, version bounds, stale downloads, deterministic event order,
no round barrier, interrupted recovery, and completed resume. The existing
synchronous baseline tests remain part of the local suite.

Both GPU pilots used one V100 32 GB each. Each pilot allowed at most five
accepted updates and used all 14,042 zero-shot MMLU questions after every update.
The stop rule ended both pilots before their budget.

| Update | Uniform stale versions | Client arrival events |
|---|---|---|
| 0 | 6,568 / 14,042 (46.77%) | 6,568 / 14,042 (46.77%) |
| 1 | 6,568 / 14,042 (46.77%) | 6,568 / 14,042 (46.77%) |
| 2 | 6,583 / 14,042 (46.88%) | 6,583 / 14,042 (46.88%) |
| 3 | 6,572 / 14,042 (46.80%) | 6,585 / 14,042 (46.90%) |
| 4 | Stopped | 6,579 / 14,042 (46.85%) |

The paper-style pilot stopped after update 3 and retained update 2: 46.88%.
The event pilot stopped after update 4 and retained update 3: 46.90%.
Both started at 46.77%. These are short execution and correctness checks.
The 50-update configuration has not been run.

The paper-style pilot trained clients 0 and 1. Client 1 trained twice. The
event pilot trained clients 0, 1, and 2. Client 0 trained twice. Neither pilot
visited every client before stopping. Do not compare these scores with earlier
synchronous runs as a controlled ranking of algorithms. Those runs used different
optimizers and amounts of local work.

Both 120-row smoke tests completed ten accepted updates. The paper-style smoke
test included ages zero through four. The event smoke test included ages up to
five. Every received model matched its recorded download. The server retained
the current model between arrivals and used the expected shared coefficient.

The audits independently replayed every update from saved global factors and
client uploads. The results matched exactly. Local jobs started with zero
proximal penalty, then produced positive penalties. No optimizer step was
skipped because of loss-scaler overflow. The base weights stayed unchanged.

Each saved best adapter reproduced all predictions and scores on all 14,042
questions after a fresh load. Completed resume launched no further updates.
The local tests also recovered pending stale downloads and random state after
a simulated job failure. An isolated copy passed 54 tests without the earlier
uncommitted trace changes.

See [the structured record](fedasync-lora.json) for update counts, losses,
staleness, coefficients, settings, source hashes, and audit results. Raw
predictions and selected adapters remain outside Git.
