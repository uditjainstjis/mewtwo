# Experiment Card 22 — Synapta Indian BFSI Benchmark v1 Release ($n=60$)

PKB wrapper around `RESEARCH_HISTORY/14_benchmark_v1_release.md`.

## 1. Research question
A controlled, hand-curated, externally releasable benchmark for paired evaluation of Indian regulatory QA. Does the recipe survive on hand-validated questions outside the regex template structure?

## 2. Dataset
- 60 hand-curated questions (RBI 30, SEBI 30; Tier 2 numeric 30, Tier 3 heading-extractive 30).
- 22 distinct held-out PDFs.
- Mixed scoring: substring 30, token_f1_threshold_0.5 30.
- Per-question hand-curated alternative-answer lists.
- License: CC-BY-SA-4.0.

## 3. Model
- Same base + bfsi_extract + Format Guard (4-adapter) as card 19.

## 4. Evaluation
- Gated reference scorer in `data/benchmark/synapta_indian_bfsi_v1/scoring.py` (Wilson CIs + paired McNemar built in).

## 5. Results

| Mode | Score | Wilson 95\% CI | Substring | F1 |
|---|---:|---|---|---:|
| Base | 40.0\% (24/60) | [28.6, 52.6] | 76.7\% | 0.122 |
| + bfsi_extract | **50.0\%** (30/60) | [37.7, 62.3] | 98.3\% | 0.157 |
| Format Guard | 50.0\% (30/60) | [37.7, 62.3] | 98.3\% | 0.157 |

Paired McNemar: bfsi_extract vs base $p = 0.0313$ (binomtest, marginal at $\alpha=0.05$).
FG vs adapter direct: identical 30/60, $p = 1.0$. Mean 0.1 swaps/Q.

### Per scoring method (the interesting finding)
| Method | $n$ | Base | + Adapter | Lift |
|---|---:|---:|---:|---|
| substring | 30 | 80\% | **100\%** | $+20$ pp clean win |
| token_f1_threshold_0.5 | 30 | 0\% | 0\% | both fail F1 cutoff |

## 6. Negatives + caveats
- F1$\geq$0.5 cutoff is too strict for verbose paragraph-extraction style. Mean F1 ($\sim 0.12$ base, $\sim 0.16$ adapter) sits well below 0.5 even when the right paragraph is quoted.
- Sample size $n=60$ is small; McNemar at $p=0.0313$ is barely significant.
- Future scoring revision: F1$\geq$0.3 or sentence-overlap variant for Tier 3.

## 7. Artifact map
PRIMARY:
- `data/benchmark/synapta_indian_bfsi_v1/{questions.jsonl, scoring.py, README.md, LICENSE.md, dataset-metadata.json, build_benchmark.py}`
- `synapta_src/data_pipeline/{14_publish_benchmark, 15_publish_kaggle, 17_eval_benchmark_v1, 18_eval_benchmark_v1_fg}.py`
- `results/benchmark_v1_eval/{predictions_{base,bfsi_extract,format_guard}.jsonl, summary.json}`

SECONDARY:
- `RESEARCH_HISTORY/14_benchmark_v1_release.md`
- `docs/recent/BENCHMARK_RELEASE_BLOG.md` ($\sim$750-word announcement)
