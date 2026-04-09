# MD External V2 Comparison Summary

## Overall

| Method | n | Sim | F1 | Lat(s) | Rubric Cov | Rubric Pass |
|---|---:|---:|---:|---:|---:|---:|
| weighted_merge | 100 | 0.6592 | 0.2719 | 4.263 | 0.1261 | 0.0000 |
| late_layer_injection | 100 | 0.6594 | 0.2715 | 3.890 | 0.1230 | 0.0000 |
| sequential_reverse | 100 | 0.6623 | 0.2734 | 4.605 | 0.1338 | 0.0000 |
| mistral | 100 | 0.6907 | 0.2917 | 10.654 | 0.1683 | 0.0100 |

## Key Read

- Best current Qwen by rubric coverage: `sequential_reverse`
- Fastest Qwen: `late_layer_injection`
- Mistral is stronger on soft and rubric metrics, but slower.
