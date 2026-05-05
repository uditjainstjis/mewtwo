# Experiment Card 06 — Router SFT + DPO on 5000 Synthetic Routing Examples

## 1. Research question and hypothesis

Can a trained classifier-style router (SFT on synthetic routing data) outperform CoT-generative routing? Does DPO further improve the router on a pairwise preference objective?

- **H6 (non-generative routing superiority):** trained classifier > CoT routing on routing accuracy.
- **H7 (gated dynamic K):** confidence-gated $K=1/K=2$ routing prevents noise injection on simple queries.
- DPO post-hoc: does pairwise preference DPO improve the router as a routing classifier?

## 2. Dataset and task definition

- **Train:** `data/router_synthetic_routing_5000.json` (5000 synthetic routing examples).
- **Validation holdout:** `data/router_synthetic_routing_5000_valid_holdout.json` ($n=100$).
- **DPO training:** `data/router_reasoning_dpo_5000.jsonl` (preference pairs).
- **Routing labels:** required-domain set per query (single or multi-label).

## 3. Model and configuration

- **Base:** the same Qwen-1.5B router stub used in earlier routing experiments.
- **SFT:** assistant-only loss masking on Apple Silicon MPS.
- **DPO:** response-only masked log-probs on MPS.
- **Adapter paths:**
  - `router_adapters/router_reasoning_sft_5000_mpsfix/`
  - `router_adapters/router_reasoning_dpo_5000_mpsfix/`

## 4. Evaluation protocol

- **Routing-classifier metrics:** exact-match accuracy, partial-overlap accuracy, mean overlap F1, mean router latency.
- **Downstream effect:** 10-item TCAR pilot (semantic sim, F1, latency) per router; final 100-item TCAR comparison vs Mistral baseline.

## 5. Main quantitative results

### Router accuracy (n=100 holdout)

| Router | Exact match | Partial overlap | Mean overlap F1 | Mean latency (s) |
|---|---:|---:|---:|---:|
| SFT | **0.85** | 1.00 | **0.945** | 1.079 |
| DPO | 0.42 | 0.75 | 0.6333 | 1.697 |

### Comparison to earlier routers (from `docs/MASTER_KNOWLEDGE_BASE.md` H6)

| Router | Exact match |
|---|---:|
| CoT (generative) | $\approx$48.7\% |
| Embedding (centroid) | 78.7\% |
| **SFT** | 85.0\% |
| DPO | 42.0\% |

### Downstream TCAR pilot (10 items)

| System | Sim | F1 | Latency (s) |
|---|---:|---:|---:|
| TCAR + SFT router | 0.6902 | 0.2874 | 16.845 |
| TCAR + DPO router | 0.7032 | **0.3046** | 19.642 |

### Final 100-item TCAR comparison (`07_tcar_collaborative.md` for full)

See card 07 for full results; key takeaway: TCAR+DPO matched Mistral on similarity ($-0.0007$), lost on F1.

## 6. Negative results, limitations, and bugs

- **DPO regressed the routing classifier sharply** (85 → 42 exact match) despite optimising the pairwise preference objective successfully. **Lesson: pairwise preference DPO objectives can degrade classification quality.**
- **Asymmetry:** the DPO router scored worse on routing classification but better on the 10-item downstream TCAR pilot (F1 0.30 vs 0.29). This justified running the full 100-item benchmark before rejecting DPO.
- **Synthetic training data:** routing labels are synthetically generated, not externally authored. Routing accuracy on real (externally constructed) multi-domain queries is not evaluated here.

## 7. Artifact map

PRIMARY:
- `results/router_accuracy_sft_5000_valid_holdout_mpsfix.json`
- `results/router_accuracy_dpo_5000_valid_holdout_mpsfix.json`
- `results/router_sft_mpsfix_results_2026_04_09.md`
- `results/tcar_collaborative_sft5000_mpsfix_pilot10.jsonl`
- `results/tcar_collaborative_dpo5000_mpsfix_pilot10.jsonl`
- `data/router_synthetic_routing_5000.json` (search local repo to confirm presence)
- `data/router_reasoning_dpo_5000.jsonl`

SECONDARY:
- `docs/ROUTER_SFT_DPO_PLAN_2026_04.md`
- `results/router_upgrade_execution_log_2026_04_08.md`

MISSING / off-device:
- the holdout JSONs are referenced; confirm both are present.
- DPO holdout file was at the time at `/Users/uditjain/Desktop/adapter/results/router_accuracy_dpo_5000_valid_holdout_mpsfix.json`; verify under current `results/` path.
