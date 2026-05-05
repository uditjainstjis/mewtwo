# Synapta Indian BFSI Benchmark v1 — Release and Seed Baselines

**Date:** 2026-05-04
**Release license:** CC-BY-SA-4.0
**Source artefacts:**
- `data/benchmark/synapta_indian_bfsi_v1/` (release directory)
- `synapta_src/data_pipeline/14_publish_benchmark.py` (HF push, dry-run validated)
- `synapta_src/data_pipeline/15_publish_kaggle.py` (Kaggle push, dry-run validated)
- `synapta_src/data_pipeline/17_eval_benchmark_v1.py` (paired base+adapter eval)
- `synapta_src/data_pipeline/18_eval_benchmark_v1_fg.py` (FG mode eval)
- `results/benchmark_v1_eval/{predictions_*,summary.json}`

## Why this benchmark
Two motivations:
1. **External validation by community contributors.** Anyone can clone, run, and push baseline numbers via PR. Existing CC-BY-SA datasets in this domain are limited.
2. **A controlled paired eval surface** for the methodology — same gated `scoring.py`, paired McNemar built in, alternative-answer lists hand-curated.

## Composition

| Property | Value |
|---|---|
| Total questions | 60 |
| Regulator | RBI 30 / SEBI 30 |
| Tier | Tier 2 numeric 30 / Tier 3 heading-extractive 30 |
| Topic tags | governance 22, mutual_fund 12, foreign_exchange 9, banking_ops 6, reporting 4, fraud 3, derivatives 3, insurance 1 |
| Difficulty | medium 36, hard 14, easy 10 |
| Scoring methods | substring 30, token_f1_threshold_0.5 30 |
| Source-PDF coverage | 22 distinct held-out PDFs |
| Hand-curated alternative-answer lists | yes |

## Release contents
- `questions.jsonl` (60 rows × full metadata)
- `scoring.py` (gated reference scorer with paired McNemar + Wilson CIs)
- `README.md` (340 lines, full schema documentation)
- `LICENSE.md` (CC-BY-SA-4.0)
- `dataset-metadata.json` (Kaggle metadata)
- `build_benchmark.py` (deterministic builder; documentation of construction, not runnable from the published bundle without internal corpus)

## Seed baselines (n=60 paired, 3 modes)

| Mode | Score | Wilson 95\% CI | Substring | Token F1 |
|---|---|---|---|---|
| Base Nemotron-30B (4-bit) | 40.0\% (24/60) | [28.6, 52.6] | 76.7\% | 0.122 |
| + bfsi_extract LoRA | **50.0\%** (30/60) | [37.7, 62.3] | 98.3\% | 0.157 |
| Format Guard (4 adapters) | 50.0\% (30/60) | [37.7, 62.3] | 98.3\% | 0.157 |

Paired McNemar: bfsi_extract vs base $p = 0.0313$ (binomtest, marginal at $\alpha=0.05$ on $n=60$).

Format Guard vs bfsi_extract direct: identical 30/60 ($b_{10}=b_{01}=0$, $p = 1.0$). Mean adapter swaps per question under FG: 0.1 — the BFSI router stayed locked on bfsi_extract for nearly every question.

### Per scoring method

| Method | $n$ | Base | + Adapter | Lift |
|---|---|---|---|---|
| substring | 30 | 80.0\% | **100.0\%** | +20 pp clean win |
| token_f1_threshold_0.5 | 30 | 0.0\% | 0.0\% | both fail F1 cutoff |

The F1$\geq$0.5 cutoff is too strict for our model's verbose paragraph-extraction answer style on Tier 3 heading questions. Mean F1 ($\sim 0.12$ base, $\sim 0.16$ adapter) sits well below 0.5 even when the right paragraph is quoted. We disclose openly that a future scoring revision should consider F1$\geq$0.3 or sentence-overlap variants for that question class.

### Per regulator

| Regulator | $n$ | Base | Adapter | FG |
|---|---|---|---|---|
| RBI | 30 | 11/30 (36.7\%) | 15/30 (50.0\%) | 15/30 (50.0\%) |
| SEBI | 30 | 13/30 (43.3\%) | 15/30 (50.0\%) | 15/30 (50.0\%) |

## Open contribution path

The release-table baselines reserve external API rows (Anthropic Claude, OpenAI GPT-4o, Google Gemini) as `_pending_`. Community contributors are invited to add rows via PR with their evaluation harness, prompt, and date.

## Files
- `data/benchmark/synapta_indian_bfsi_v1/` (release directory, contents above)
- `results/benchmark_v1_eval/predictions_{base,bfsi_extract,format_guard}.jsonl`
- `results/benchmark_v1_eval/summary.json`
- `docs/recent/BENCHMARK_RELEASE_BLOG.md` (~750-word announcement)
