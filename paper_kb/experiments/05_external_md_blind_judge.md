# Experiment Card 05 — External MD Benchmark and Blind Judge Comparison

## 1. Research question and hypothesis

Does any Qwen-1.5B Synapta variant beat Mistral-7B on a blindly judged, externally authored multi-domain benchmark?

The motivation: internal-similarity-based comparisons earlier ranked Qwen Synapta variants ahead of Mistral. The "Synapta beats Mistral" headline was a candidate paper claim. This experiment was designed to test whether that headline survives an external benchmark and a blind rubric judge.

## 2. Dataset and task definition

- **Dataset:** `data/multidomain_eval_claude_external_v2_100.json` — 100-question externally authored MD benchmark (the dataset name implies Claude-authored generation; gold answers are externally provided rubric responses).
- **Sub-experiment 1:** full 100-item Qwen and Mistral runs scored on semantic similarity + token F1 + latency.
- **Sub-experiment 2:** 30-item stratified blind pairwise rubric comparison (Qwen variant vs Mistral; judge does not know which is which).

## 3. Model and configuration

- Qwen base: Qwen2.5-1.5B-Instruct-4bit MLX, with 9 different injection / merge strategies (`08_injection_9technique_ablation.md`).
- Mistral baseline: Mistral-7B run on Apple Silicon (latency penalty noted).

## 4. Evaluation protocol

- **Sub-experiment 1 metrics:** semantic similarity (sentence-transformers), token F1, latency.
- **Sub-experiment 2 metrics:** Qwen wins / Mistral wins / Ties on a stratified blind 30-question subset.

## 5. Main quantitative results

### Sub-experiment 1 — full 100-item external MD
| System | Sim | F1 | Mean latency (s) |
|---|---:|---:|---:|
| `late_layer_injection` (Qwen) | 0.6594 | 0.2715 | 3.890 |
| `weighted_merge` (Qwen) | 0.6592 | 0.2719 | 4.263 |
| `sequential_reverse` (Qwen) | 0.6623 | 0.2734 | 4.605 |
| Mistral-7B | **0.6907** | **0.2917** | 10.654 |

Static Qwen methods are much faster; **none beat Mistral** on the external 100-item benchmark.

### Sub-experiment 2 — 30-item blind judge
| Qwen method | Qwen wins | Mistral wins | Ties |
|---|---:|---:|---:|
| `weighted_merge` | 6 | 23 | 1 |
| `late_layer_injection` | 4 | 26 | 0 |
| `sequential_reverse` | 4 | 25 | 1 |

## 6. Negative results, limitations, and bugs

- **The external blind judge does NOT support a "Qwen Synapta beats Mistral" headline.** All three Qwen methods lose blind comparisons by margins of $\sim 4-7$ wins for Qwen vs $\sim 23-26$ for Mistral.
- **Methodology lesson explicitly recorded:** the original internal benchmark over-ranked Qwen methods that the external rubric judge disagreed with. The conclusion in `docs/EXTERNAL_EVAL_PLAN.md` is to (a) retire semantic similarity as the headline metric, (b) treat the internal Qwen-authored benchmark as an ablation/dev set only, (c) never repeat "sequential routing is generally superior" as a claim.
- **TCAR collaborative pivot:** this finding motivated the move from weight-blending to a TCAR collaborative inference loop (`07_tcar_collaborative.md`).

## 7. Artifact map

PRIMARY:
- `results/md_external_v2_comparison_summary.json` (9K)
- `results/md_external_v2_soft_vs_blind_summary.json` (1K)
- `results/md_pairwise_latelayer_vs_mistral_v2_strat30_summary.json` (2K)
- `results/md_pairwise_merge_vs_mistral_v2_strat30_summary.json` (2K)
- `data/multidomain_eval_claude_external_v2_100.json` (external benchmark; cited by historical absolute path `/Users/uditjain/Desktop/adapter/data/...`)
- `results/md_head_to_head_v2_mistral_only_100.jsonl` (Mistral baseline raw)

SECONDARY:
- `docs/EXTERNAL_EVAL_PLAN.md`
- `docs/MASTER_EXPERIMENT_REPORTS.md` (Phase B narrative)

MISSING / off-device:
- some referenced JSONLs may have absolute paths from the older Apple Silicon machine; verify all referenced files in `results/` are present and that `data/multidomain_eval_claude_external_v2_100.json` is present locally.
