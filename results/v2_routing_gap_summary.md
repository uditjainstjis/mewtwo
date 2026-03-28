# v2c Routing Gap Summary (Real Router vs Oracle)

**Date:** 2026-03-29
**Experiment:** 120 real inferences (40 MD questions × 3 methods)
**Log file:** `results/v2_md_routing_ablation.jsonl`

## Goal
To measure the "routing gap" — how much of the oracle multi-adapter compositional gain is recoverable using a real, heuristic top-2 CoT router. All methods in this phase utilized the new `norm_ratio` clamp.

## Aggregate Results (MD Split)

| Method | Routing | Avg Sim ↑ | Avg PPL ↓ | Avg Lat (s) | Avg K |
|--------|---------|-----------|-----------|-------------|-------|
| SingleAdapter | CoT (K=1) | 0.6296 | 12.8 | 4.178 | 1.00 |
| **AC-v2-Norm-RealRouter** | **CoT (Top-2)** | **0.6350** | **12.7** | **4.167** | **1.75** |
| AC-v2-Norm-Oracle | Oracle (K=2) | 0.6502 | 12.6 | 4.211 | 2.00 |

## Computed Deltas

- **Oracle Headroom:** Δ_SIM(Oracle − SA) = **+0.0206**
- **Realized Gain:** Δ_SIM(RealRouter − SA) = **+0.0054**
- **Routing Gap:** Δ_SIM(Oracle − RealRouter) = **-0.0152**

## Conclusion

1. **Oracle headroom is small.** Even with perfect knowledge of the required domains, composing two adapters yields only a +2.06% similarity gain over a single CoT-routed adapter baseline.
2. **The Real Router recovers ~26% of the available headroom.** The heuristic CoT router successfully extracts $K=2$ on 75% of questions (Avg $K=1.75$), yielding a +0.54% gain. 
3. **Routing accuracy is a bottleneck, but not the primary failure mode.** The $+0.0054$ realized gain is an order of magnitude below the original preregistered $+0.05$ threshold. Even if the router improved to oracle-level accuracy ($+0.0206$), the compositional benefit remains marginal. This strongly implies that the 1.5B parameter base model space, or adapter orthogonality, limits compositional efficacy more than the router itself.
