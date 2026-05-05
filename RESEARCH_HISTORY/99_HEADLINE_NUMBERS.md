# Headline Numbers — single source of truth

Every claim cited in the paper or pitch deck must trace to a row here. If a number is not in this table, it should not be cited.

## Format Guard / token-level routing

| Claim | Number | Source artefact |
|---|---|---|
| HumanEval $n=164$ Format Guard pass@1 | **73.2\%** [Wilson 65.9, 79.4] | `results/overnight/qa_pairs/humaneval_rescored_summary.json` |
| HumanEval $n=164$ base pass@1 | **56.1\%** [Wilson 48.4, 63.5] | same |
| Format Guard lift on HumanEval | **+17.1 pp** | computed; see paper §5.2 |
| McNemar $\chi^2$ HumanEval | **15.68** ($p < 10^{-3}$) | `docs/findings/humaneval_statistical_analysis.md` |
| Paired McNemar contingency | $b_{11}=81, b_{10}=11, b_{01}=39, b_{00}=33$ | same |
| Cold-swap latency | **315.9 ms** avg over 44 swaps NVMe SSD | `results/cold_swap_metrics.json` |
| Warm GPU swap | **$O(1)$** PEFT `set_adapter()` pointer flip | `synapta_src/...` source |

## Code Paradox (Nemotron-30B Phase 1)

| Adapter | ARC ($n=100$) | HumanEval ($n=100$) | MATH-500 ($n=200$) | MBPP ($n=100$) | Source |
|---|---|---|---|---|---|
| base | 20.0\% | 50.0\% | 41.5\% | 8.0\% | `results/nemotron/master_results.json` |
| math | 23.0\% | **60.0\%** | 50.5\% | 2.0\% | same |
| code | **31.0\%** | 27.0\% | **56.0\%** | 6.0\% | same |
| science | 21.0\% | 1.0\% | 55.0\% | 0.0\% | same |
| merged (DARE/TIES) | 19.0\% | 34.0\% | 56.0\% | 0.0\% | same |

## Code Paradox replication ($n=200$ Qwen-3.5-0.8B)
| Mode | Acc | Source |
|---|---|---|
| base | 15.0\% (30/200) | `results/overnight/qa_pairs/code_paradox_qwen_n200_summary.json` |
| math adapter | 16.0\% (32/200) | same |
| code adapter | 12.0\% (24/200) | same |

**Pattern preserved at smaller scale:** code-on-code regression survives base-model swap.

## Static composition (mixed-domain $n=45$ probe)
| Method | Score | Source |
|---|---|---|
| Base | 51.1\% | `results/nemotron/sprint_results.json` |
| Best single (routed) | 60.0\% | same |
| DARE merged | 60.0\% | same |
| TIES merged | 60.0\% | same |
| Linear merged | 60.0\% | same |
| **Composition gain** | **+0.0 pp** | computed |

## LoRI MoE (Qwen-2.5-1.5B, n=200)

| Phase | Benchmark | Mode | Score |
|---|---|---|---|
| Phase 1 base | gsm8k | zero-shot | 26.0\% |
| Phase 1 base | arc_challenge | zero-shot | 76.5\% |
| Phase 1 base | mmlu | zero-shot | 56.5\% |
| Phase 2 single | gsm8k | math adapter | 53.0\% |
| Phase 2 single | arc_challenge | legal adapter | 77.5\% |
| Phase 3 composite | gsm8k | routed top-1 | 4.0\% (regressed) |
| Phase 3 composite | arc_challenge | routed top-1 | 72.0\% |
| Phase 3 composite | mmlu | routed top-1 | 53.0\% |

Source: `results/lori_moe/all_results.json`. Verdict: composite routing on this base+benchmark mix did not exceed the single best expert; on GSM8K it regressed catastrophically.

## BFSI extract held-out ($n=664$ paired)

| Mode | Substring | Wilson 95\% CI | Token F1 |
|---|---|---|---|
| Base | 58.7\% | [54.95, 62.42] | 0.132 |
| + bfsi_extract | **89.6\%** | [87.06, 91.71] | 0.172 |
| Format Guard | 88.7\% | [86.07, 90.89] | 0.171 |

| McNemar comparison | $b_{10}$ | $b_{01}$ | Δ | $p$ |
|---|---|---|---|---|
| adapter vs base | 14 | 219 | +30.9 pp | **$1.66 \times 10^{-48}$** |
| FG vs base | 19 | 218 | +30.0 pp | $5.12 \times 10^{-44}$ |
| FG vs adapter | 6 | 0 | -0.9 pp | $0.031$ |

Source: `results/bfsi_eval/summary.json`.

## BFSI recall ($n=214$ paired F1)

| Mode | F1 mean | Source |
|---|---|---|
| Base | 0.158 | recomputed from `results/bfsi_recall_eval/eval_results.jsonl` |
| Adapter | 0.219 | same |
| FG | 0.219 | same |
| Lift | $+0.061$ ($+38.4\%$ rel.) | computed |
| Wilcoxon adapter > base | $p = 1.50 \times 10^{-16}$ | scipy.stats |
| Wilcoxon FG vs adapter | $p = 0.55$ | same |
| Adapter F1 > base F1 per question | 159 / 214 (74.3\%) | same |

## IndiaFinBench ($n=324$ test split)

| Task type | $n$ | Synapta | Token F1 | Gemini Flash baseline |
|---|---|---|---|---|
| regulatory_interpretation | 139 | 34.5\% | 0.347 | 93.1\% |
| temporal_reasoning | 62 | 16.1\% | 0.366 | 88.5\% |
| numerical_reasoning | 73 | 9.6\% | 0.297 | 84.8\% |
| contradiction_detection | 50 | 78.0\%* | 0.015 | 88.7\% |
| **Overall** | **324** | **32.1\%** [27.3, 37.4] | **0.288** | **89.7\%** |

\* contradiction_detection substring inflated by yes/no string-match coincidence; F1=0.015 confirms model is not actually performing the task.

Source: `results/indiafinbench_eval/summary.json`.

## Synapta Indian BFSI Benchmark v1 ($n=60$ paired, 3 modes)

| Mode | Primary score | Wilson 95\% CI | Substring | F1 |
|---|---|---|---|---|
| Base | 40.0\% (24/60) | [28.6, 52.6] | 76.7\% | 0.122 |
| + bfsi_extract | **50.0\%** (30/60) | [37.7, 62.3] | 98.3\% | 0.157 |
| Format Guard | 50.0\% (30/60) | [37.7, 62.3] | 98.3\% | 0.157 |

McNemar adapter vs base: **$p = 0.0313$** (binomtest). FG vs adapter: $p = 1.0$ (identical predictions).

Per scoring method:
| Method | $n$ | Base | Adapter | Lift |
|---|---|---|---|---|
| substring | 30 | 80.0\% | **100.0\%** | +20.0 pp clean win |
| token_f1_threshold_0.5 | 30 | 0.0\% | 0.0\% | both fail cutoff |

Source: `results/benchmark_v1_eval/summary.json`.

## Frontier comparison (Synapta vs Claude, $n=15$)

| Model | Substring | Token F1 |
|---|---|---|
| Synapta-Nemotron-30B + bfsi_extract | **87\%** | 0.38 |
| Anthropic Claude Opus | 7\% | **0.65** |
| Anthropic Claude Sonnet | 27\% | 0.49 |

Source: `results/frontier_comparison/subagent_results.jsonl`. Sample size is small (n=15) — directional only.

## Corpus stats

| Item | Number | Source |
|---|---|---|
| RBI Master Directions scraped | 80 | `data/rbi_corpus/pdfs/` (count) |
| SEBI Master Circulars scraped | 50 | `data/sebi_corpus/pdfs/` (count) |
| Total PDFs | 130 | sum |
| Total characters extracted | 8.06M | extraction logs |
| Sections detected | 7,329 | chunking logs |
| Smart chunks (median 384 tokens) | 4,185 | same |
| Raw QA pairs constructed | 4,477 | `04_build_qa_pairs.py` output |
| Train pairs (after v2 cleaner) | 2,931 | `data/rbi_corpus/qa/train_v2_clean.jsonl` |
| Eval pairs (held-out, doc-disjoint) | 664 | `data/rbi_corpus/qa/eval_clean.jsonl` |
| Held-out PDFs (entirely quarantined from training) | 26 (20\%) | `data/rbi_corpus/qa/split_manifest_v2.json` |
| 10-check validator pass rate | 98.45\% | `06_validate_qa.py` log |

## Training cost
| Metric | Value | Source |
|---|---|---|
| bfsi_extract training time | 3h 28min | `logs/train_bfsi_extract.log` |
| Trainable parameters | 434.6M / 32B = **1.36\%** | LoRA config |
| Adapter file size (bf16 safetensors) | 1.74 GB | `adapters/nemotron_30b/bfsi_extract/best/` |
| Peak VRAM | 17.81 GB | `logs/train_bfsi_extract.log` |
| Hardware | RTX 5090 (32 GB) | --- |

## Adapters released

| Adapter | Base | Rank | Domain | Path |
|---|---|---|---|---|
| math (Nemotron-30B) | Nemotron-Nano-30B-A3B | 16 | math | `adapters/nemotron_30b/math/best/` |
| code (Nemotron-30B) | same | 16 | code | `adapters/nemotron_30b/code/best/` |
| science (Nemotron-30B) | same | 16 | science | `adapters/nemotron_30b/science/best/` |
| bfsi_extract | same | 16 | RBI/SEBI extractive QA | `adapters/nemotron_30b/bfsi_extract/best/` |
| bfsi_recall | same | 16 | RBI/SEBI no-context recall | `adapters/nemotron_30b/bfsi_recall/best/` |
| 30 × Nemotron-Mini-4B zoo | Nemotron-Mini-4B-Instruct | {1,2,8,128,1024,3072} | math/code/science/merged | `adapters/small_models_zoo/from_hf_kaggle/` |
| 37 × Qwen-3.5-0.8B zoo | Qwen-3.5-0.8B | multi | math/code/science/legal/medical | same parent |

Total released adapters: **72** (including the merged DARE-sparsified ablations).
