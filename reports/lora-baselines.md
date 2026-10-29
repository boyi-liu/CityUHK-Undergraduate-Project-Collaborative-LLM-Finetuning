# Synchronous LoRA baselines

## Reference checks

FFA-LoRA follows Section 4 and Appendix A.8 of
[Improving LoRA in Privacy-preserving Federated Learning](https://arxiv.org/abs/2403.12313).
The method keeps one shared A fixed. It trains B and aggregates B with sample
weights. The main text describes a random Gaussian A. Appendix A.8 reports
Kaiming initialization as the usual setting and compares other initializations.
This implementation uses PEFT's Kaiming uniform A and zero B. It implements the
non-private setting. It adds neither gradient noise nor a privacy accountant.

FLoRA follows Section 3.2 of
[FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations](https://papers.nips.cc/paper_files/paper/2024/file/28312c9491d60ed0c77f7fff4ad86dd1-Paper-Conference.pdf).
The check also used the authors' code at revision
[`8efe72d1a7c77456f5ab342efd6394732bfabe05`](https://github.com/CASE-Lab-UMD/FLoRA/tree/8efe72d1a7c77456f5ab342efd6394732bfabe05).
Their aggregation code weights A once and joins B without a second weight.
Their main loop merges the global product and starts fresh local adapters in
the next round. The implementation follows these rules. It does not use SVD,
zero padding, factor averaging, or an asynchronous schedule for FLoRA.

## Rules and state

Let p_i be a client's share of the training rows. Let s = alpha/r.

- FFA-LoRA: A stays equal to A_0. The server sets B = sum_i p_i B_i.
  Thus s B A_0 = sum_i p_i s B_i A_0. Only B enters each upload.
- FLoRA: A_stack = concat_rows(p_i A_i). B_stack = concat_columns(B_i).
  Thus s B_stack A_stack = sum_i p_i s B_i A_i. There are no cross-client terms.

All clients finish before the next round starts. Each client uses a fresh Adam
optimizer. FLoRA also uses a fresh local A/B pair. It merges the preceding
global product into the frozen base for that local job. It does not optimize
the base weights. FFA-LoRA resumes the common B and never changes A.

For FLoRA, the runtime keeps original copies of the target base weights on the
CPU. It restores those weights before loading another global state. This
prevents a client's temporary merge from entering the next client's starting
model. A full hash check confirms that the runtime restores the original base.

Each FLoRA checkpoint joins the earlier global factors and the new round stack.
This stores the accumulated update on the original pretrained base. It permits
a standard PEFT export and resume without losing earlier rounds. Local ranks
remain fixed. The saved alpha changes with the cumulative rank so s stays
constant. For unequal local ranks, each local alpha also changes to keep s
constant. The default uses local rank 8, alpha 32, and s = 4.

This saved form does not model network traffic. Its increasing rank must not
be used as the size of an original FLoRA round upload. The run records the
actual local upload size separately. The trace-session tools do not yet run
these methods or model their full downlink costs.

## Project experiment

The configurations retain the existing Qwen2.5-0.5B model and pinned revisions.
They use the same 1,780 clean MMLU training rows, eight subject groups, five
Dirichlet clients, alpha 0.5, and seed 42. Client sizes are 488, 304, 247, 232,
and 509. The training settings are rank 8, LoRA alpha 32, dropout 0.1, q_proj and
v_proj, batch 8, Adam at 0.0002, no weight decay, and one local epoch. Training
uses 128 tokens and answer-only loss. This leaves answer targets in 1,383 rows.
The other 397 rows do not supply answer loss.

The initial model and every completed round use zero-shot evaluation on all
14,042 MMLU test questions. Complete questions fit within 4,096 tokens. The run
stops on the first strict accuracy decrease or after ten rounds. It keeps the
earliest highest score, including round zero.

The user requested this stop rule. It uses test accuracy to select a model.
The reported scores are selection scores, not independent final test scores.
The settings differ from the papers' datasets, models, and training settings.
These runs test the algorithms in this project; they do not reproduce paper
tables. New adapters use PEFT Kaiming A; the earlier FedIT-style experiment used
uniform A from -0.01 to 0.01. Keep that difference when comparing the results.

## Verification

The local tests check frozen A and base weights, B-only uploads, sample weights,
unequal local ranks, products over two rounds, fresh local adapters, constant
LoRA scale, saved PEFT reloads, interruption recovery, and completed resume.

All 71 local tests passed. Six baseline tests also passed in the server's
installed PEFT runtime. The GPU runs used one V100 32 GB per method.

Both full runs used identical client partitions and evaluation prompts. Their
initial adapters also matched. The table gives zero-shot MMLU selection scores.

| Round | FFA-LoRA | FLoRA |
|---|---|---|
| 0 | 6,568 / 14,042 (46.77%) | 6,568 / 14,042 (46.77%) |
| 1 | 6,695 / 14,042 (47.68%) | 6,733 / 14,042 (47.95%) |
| 2 | 6,751 / 14,042 (48.08%) | 6,800 / 14,042 (48.43%) |
| 3 | 6,791 / 14,042 (48.36%) | 6,820 / 14,042 (48.57%) |
| 4 | 6,791 / 14,042 (48.36%) | 6,746 / 14,042 (48.04%) |
| 5 | 6,766 / 14,042 (48.18%) | Stopped |

FFA-LoRA stopped after round 5. It retained round 3 at 48.36%.
FLoRA stopped after round 4. It retained round 3 at 48.57%.

For both methods, the saved best adapter reproduced every prediction and every
score on all 14,042 questions. Completed resume launched no further rounds.
The product audits passed for every completed round. FFA-LoRA kept A unchanged
and uploaded B only. FLoRA kept every local training rank at eight. Its separate
GPU check also passed with local ranks `[4, 8, 12, 8, 4]` over two rounds.

The runs visited every assigned row once per local epoch. Neither full run
skipped an optimizer step because of loss-scaler overflow. Base hash checks
passed. Both 120-row smoke runs also passed; their 12-question scores are debug
results and do not measure full MMLU performance.

The earlier FedIT-style run selected 48.90%. Its A initialization differs from
these runs, as described above. These results do not establish a controlled
ranking of the three methods. All reported best scores use test questions for
selection and therefore remain selection scores.

See [the structured verification record](lora-baselines.json) for round losses,
counts, settings, source hashes, upload sizes, and audit results. Raw predictions
and selected adapters remain outside Git.
