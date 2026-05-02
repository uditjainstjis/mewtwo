# HumanEval n=164 — Statistical Analysis

## Confidence intervals (Wilson 95%)

| Mode | Pass@1 | 95% CI |
|---|---|---|
| Base Nemotron-30B | 92/164 = 56.1% | 48.4% – 63.5% |
| Format Guard | 120/164 = 73.2% | 65.9% – 79.4% |

**Delta: +17.1 pp**

## McNemar's test (paired, per-problem)

| Outcome | Count |
|---|---|
| Both pass | 81 |
| Both fail | 33 |
| Base only (regression) | 11 |
| FG only (improvement) | 39 |

**Net asymmetry favoring FG: +28 problems**
**McNemar's chi-square: 15.68 (p < 0.001)**

## Per-category breakdown

| Category | n | Base | FG | Delta |
|---|---|---|---|---|
| encoding | 1 | 0% | 0% | +0.0pp |
| list_ops | 14 | 64% | 57% | -7.1pp |
| math_arith | 12 | 33% | 67% | +33.3pp |
| ordering | 2 | 50% | 100% | +50.0pp |
| other | 76 | 61% | 80% | +19.7pp |
| strings | 59 | 54% | 69% | +15.3pp |

## Bottom line for the deck

The +17.1 pp delta between Format Guard and base is highly statistically significant (p < 0.001 via McNemar's paired test on n=164 problems). The 95% confidence interval for the FG pass rate is 66%–79%, non-overlapping with base's 48%–63%. This is publication-credible at NeurIPS standards.
