# Experiment Card 07 — TCAR Collaborative Inference (Synapta vs Mistral, n=100)

## 1. Research question and hypothesis

Does parallel branch generation by two domain adapters followed by a refiner pass (TCAR collaborative inference) outperform activation-space weight blending and approach Mistral-7B quality on an externally authored MD benchmark?

- **H8 (intelligence density):** a 1.5B model with TCAR can match or beat 7B Mistral on multi-domain semantic similarity at lower VRAM.
- **H9 (TCAR vs static):** parallel-branch + refiner > activation-space merging.

## 2. Dataset and task definition

- **Dataset:** `multidomain_eval_claude_external_v2_100.json` (100 items, externally authored).
- **Total inferences:** 100 (TCAR + DPO router) and 100 (Mistral) plus prior pilots.

## 3. Model and configuration

- **Qwen TCAR stack:**
  - Qwen2.5-1.5B-Instruct-4bit MLX
  - Two domain adapters per query (selected by SFT or DPO router)
  - Parallel branch generation: each branch produces a candidate
  - Refiner pass: a third forward pass synthesises from both candidates
  - Routers compared: SFT (`05_router_sft_dpo_5000.md`) and DPO
- **Mistral baseline:** Mistral-7B-Instruct on Apple Silicon (slower).

## 4. Evaluation protocol

- Semantic similarity (sentence-transformers), token F1, exact match, mean latency.
- Comparison strata: TCAR+SFT, TCAR+DPO, Mistral, plus best prior Qwen static methods (`sequential_reverse`, `late_layer_injection`).

## 5. Main quantitative results

### Final 100-item comparison

| System | Sim | F1 | Exact match | Mean latency (s) |
|---|---:|---:|---:|---:|
| TCAR + DPO router | 0.6900 | 0.2712 | 0.0000 | 24.198 |
| Mistral-7B baseline | **0.6907** | **0.2917** | 0.0000 | 10.654 |
| `sequential_reverse` (Qwen) | 0.6623 | 0.2734 | 0.0000 | 4.605 |
| `late_layer_injection` (Qwen) | 0.6594 | 0.2715 | 0.0000 | 3.890 |

### 10-item pilot (n=10 stratified)

| Method | Sim | F1 | Latency (s) |
|---|---:|---:|---:|
| late_layer_injection | 0.6804 | 0.2696 | 3.577 |
| sequential_reverse | 0.6583 | 0.2964 | 4.415 |
| tcar_collaborative | 0.6797 | 0.2682 | 18.859 |
| mistral | 0.7067 | 0.2971 | 10.718 |

## 6. Negative results, limitations, and bugs

- **TCAR matched but did NOT beat Mistral on semantic similarity** ($0.6900$ vs $0.6907$, $\Delta = -0.0007$).
- **TCAR LOSES on token F1:** $0.2712$ vs $0.2917$, $\Delta = -0.0205$. The "Synapta beats Mistral" claim is **NOT supported on F1** at $n=100$.
- **TCAR latency is $\sim 2.3\times$ Mistral** ($24.2$ s vs $10.7$ s). The "75\% less VRAM" advantage in older docs is real (1.1 GB Qwen vs 4.4 GB Mistral) but is paid for in latency.
- **Honest framing:** TCAR is a **near-match** to Mistral on similarity at $\sim$1/4 the VRAM, but does not dominate on quality. Any paper-claim must be careful: "approaches Mistral with a smaller base model, at higher inference latency."

## 7. Artifact map

PRIMARY:
- `results/tcar_dpo_final_100_report_2026_04_09.md` (canonical write-up)
- `results/tcar_collaborative_dpo5000_mpsfix_100.jsonl` (raw 100 rows)
- `results/tcar_collaborative_sft5000_mpsfix_pilot10.jsonl` (10-item SFT pilot)
- `results/tcar_collaborative_dpo5000_mpsfix_pilot10.jsonl` (10-item DPO pilot)
- `results/tcar_pilot_10_comparison.json`
- `results/tcar_pilot_10_summary.json`
- `results/tcar_pilot_10_report.md`
- `results/tcar_oracle_collaborative_pilot_10.jsonl`
- `results/tcar_verifier_pilot10_2026_04_09.md`
- `results/md_head_to_head_v2_mistral_only_100.jsonl`

SECONDARY:
- `docs/MASTER_EXPERIMENT_REPORTS.md` (Phase C TCAR)
