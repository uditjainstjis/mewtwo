# Experiment Card 08 — 9-Technique Injection Ablation

## 1. Research question and hypothesis

Across a fixed multi-domain benchmark, which adapter-injection technique (out of 9 candidates) maximises semantic similarity? Does the ranking from the internal benchmark survive on an externally authored benchmark?

## 2. Dataset and task definition

- **Internal benchmark:** earlier v2 internal MD benchmark (40 items + 9 methods).
- **External benchmark:** `data/multidomain_eval_v2.json` (40 questions externally constructed for this ablation).
- **Total inferences (external):** 360 (40 questions × 9 methods); plus earlier track A/B JSONLs at 440 each.

## 3. Model and configuration

- Base: Qwen2.5-1.5B-Instruct-4bit MLX.
- 9 techniques tested:
  1. `weighted_merge` — DARE/TIES-style weighted weight merge of two adapter LoRAs
  2. `late_layer_injection` — only inject adapter at later transformer layers
  3. `late_last_quarter` — inject only in last quarter of layers
  4. `early_third_only` — inject only in first third
  5. `sequential_token_segments` — alternate adapters per token segment
  6. `sequential_reverse` — sequential generation with reversed order
  7. `oracle_single_d1` — single oracle-selected adapter (domain 1)
  8. `oracle_single_d2` — single oracle-selected adapter (domain 2)
  9. `merge_high_clamp` — merge with high clamp constant
  10. `mistral` — external Mistral-7B baseline (10th comparator)

## 4. Evaluation protocol

- Sim, PPL, exact_match, token F1, latency (per row in JSONL).
- Aggregated to per-method means.

## 5. Main quantitative results (external benchmark)

From `results/injection_hypotheses_eval_full_20260408.jsonl` (n=360 rows = 9 methods × 40 Q):

Best Qwen variants (from `docs/MASTER_EXPERIMENT_REPORTS.md` external phase):
| Method | Sim | F1 | Latency (s) |
|---|---:|---:|---:|
| `sequential_reverse` | 0.6623 | 0.2734 | 4.605 |
| `weighted_merge` | 0.6592 | 0.2719 | 4.263 |
| `late_layer_injection` | 0.6594 | 0.2715 | 3.890 |
| `mistral` (baseline) | 0.6907 | 0.2917 | 10.654 |

Per-row examples (from JSONL inspection): the same item `md_01` (LEGAL_ANALYSIS + BEHAVIORAL_ECONOMICS) shows:
- `weighted_merge`: sim 0.8031, F1 0.1015
- `late_layer_injection`: sim 0.7908, F1 0.1031
- `sequential_token_segments`: sim 0.7855, F1 0.1015

## 6. Negative results, limitations, and bugs

- **Internal vs external disagreement:** the internal benchmark ranked `sequential_reverse` highest; on the external benchmark, `weighted_merge` and `late_layer_injection` are competitive but no Qwen technique beats Mistral.
- **The 9-technique ablation is the empirical evidence** that motivated the move from internal-similarity-based ranking to external blind judging (`05_external_md_blind_judge.md`).
- **No technique reaches a paper-safe "best" status.** The honest claim is "the choice of injection technique matters less than the underlying base-model + adapter geometry, on this 1.5B Qwen + 20-expert configuration."

## 7. Artifact map

PRIMARY:
- `results/injection_hypotheses_eval_full_20260408.jsonl` (n=360, 9 methods × 40 Q)
- `results/injection_hypotheses_eval.jsonl` (smaller earlier run, n=10)
- `results/injection_track_a.jsonl` (n=440)
- `results/injection_track_b.jsonl` (n=440)
- `data/multidomain_eval_v2.json` (external benchmark)

SECONDARY:
- `docs/MASTER_EXPERIMENT_REPORTS.md` (Phase A 9-technique narrative)
- `docs/EXTERNAL_MD_PLAN.md`
- `docs/EXTERNAL_EVAL_PLAN.md`

MISSING / off-device:
- consolidated per-method aggregate JSON (only raw JSONLs present; aggregates appear only in narrative form in MASTER_EXPERIMENT_REPORTS.md)
