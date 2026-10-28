# MMLU accuracy stopping check

The full-data GPU check passed. The runner stopped after round 4 and retained round 3.

| Round | Correct / 14,042 | Selection accuracy |
|---|---:|---:|
| 0 | 6,568 | 46.77% |
| 1 | 6,634 | 47.24% |
| 2 | 6,768 | 48.20% |
| 3 (retained) | 6,867 | 48.90% |
| 4 | 6,848 | 48.77% |

Round 4 has lower accuracy than round 3. The runner saved this result and stopped before round 5.
It retained the highest observed adapter. Equal accuracy would allow another round; a tie retains the earliest highest adapter.
The MMLU test split controls this selection rule. These are selection scores, not independent final test scores.

The data contain 1,780 unique dev + validation questions, grouped into eight major subjects and split across five clients.
The client sizes are 488, 304, 247, 232, and 509. The 128-token limit leaves answer targets in 1,383 rows.
The configuration uses Qwen2.5-0.5B, rank 8, alpha 32, dropout 0.1, Q/V targets, batch 8, Adam at 0.0002, and one local epoch.
The isolated proposed commit passed 37 unit tests. The full workspace passed 63 tests.
The GPU check reproduced the first five saved adapters and all their predictions from the earlier fixed-round experiment exactly.
A fresh model load reproduced all 14,042 predictions for the retained adapter. The base weights stayed fixed.
All model weights and raw execution files stay in ignored local output directories. The JSON companion records configuration and verification identities.
