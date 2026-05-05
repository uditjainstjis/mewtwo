# Experiment Card 04 — Routing Gap: Oracle vs Real CoT Top-2 Router

## 1. Research question and hypothesis

How much of the oracle multi-adapter compositional gain is recoverable using a real heuristic top-2 CoT router?

- Define the routing gap: $\Delta_\text{SIM}(\text{Oracle} - \text{RealRouter})$.

## 2. Dataset and task definition

- **Dataset:** v2 MD split ($n=40$).
- **Total inferences:** 120 (40 questions × 3 methods).

## 3. Model and configuration

- Base: Qwen2.5-1.5B-Instruct-4bit MLX.
- Clamp: norm_ratio (the new variant from `03_clamp_ablation_norm_ratio.md`).
- Methods:
  - SingleAdapter (CoT, $K=1$)
  - **AC-v2-Norm-RealRouter** (CoT top-2, $K\leq 2$)
  - AC-v2-Norm-Oracle (oracle, $K=2$)

## 4. Evaluation protocol

- Sim, PPL, latency.
- Routing achieved $K$ recorded.

## 5. Main quantitative results

| Method | Routing | Avg Sim | Avg PPL | Avg Latency (s) | Avg $K$ |
|---|---|---:|---:|---:|---:|
| SingleAdapter | CoT ($K=1$) | 0.6296 | 12.8 | 4.178 | 1.00 |
| AC-v2-Norm-RealRouter | CoT (top-2) | 0.6350 | 12.7 | 4.167 | 1.75 |
| AC-v2-Norm-Oracle | Oracle ($K=2$) | **0.6502** | 12.6 | 4.211 | 2.00 |

Deltas:
- **Oracle headroom:** $+0.0206$ (Oracle vs SA)
- **Realised gain:** $+0.0054$ (RealRouter vs SA)
- **Routing gap:** $-0.0152$ (Oracle vs RealRouter)
- **Headroom recovered by router:** $\approx 26\%$

## 6. Negative results, limitations, and bugs

- **Oracle headroom is small.** Even with perfect routing, two-adapter composition yields only $+2.06\%$ similarity gain over single-adapter routing on MD.
- **Routing accuracy is a bottleneck but not the primary failure mode.** The realised $+0.0054$ gain is an order of magnitude below the v2 pre-registered $+0.05$ threshold, and even oracle routing ($+0.0206$) does not clear that bar.
- **Implication:** the 1.5B base model size or adapter orthogonality limits compositional efficacy more than the router itself. A better router cannot rescue the architecture.

## 7. Artifact map

PRIMARY:
- `results/v2_routing_gap_summary.md`
- `results/v2_md_routing_ablation.jsonl` (raw 120 inferences)

SECONDARY:
- `docs/tested_hypotheses_and_results.md` (H5/H6 narrative)
