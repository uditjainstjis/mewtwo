# v2b Clamp Ablation Summary

**Date:** 2026-03-29
**Experiment:** 120 real inferences (40 MD questions × 3 methods)
**Log file:** `results/v2_md_clamp_ablation.jsonl`

## Goal
To determine if switching from the v2 per-adapter global weight cap to a true per-layer activation norm-ratio clamp (`γ = min(1, c * ||z|| / ||m||)`) improves compositional performance on the Multi-Domain (MD) split.

## Aggregate Results (MD Split)

| Method | Clamp Mode | Avg Sim ↑ | Avg PPL ↓ | Avg Lat (s) | Avg K |
|--------|------------|-----------|-----------|-------------|-------|
| SingleAdapter (c=0.5) | `weight_cap` | 0.6334 | 12.7 | 4.008 | 1.00 |
| AC-v2-WeightCap (c=0.5) | `weight_cap` | 0.6505 | 12.6 | 4.055 | 2.00 |
| **AC-v2-NormRatio** (c=0.5) | `norm_ratio` | **0.6502** | **12.6** | **4.221** | 2.00 |

## Computed Deltas

- **Δ_SIM(AC-v2-WeightCap − SA):** +0.0171 *(Matches original v2 run)*
- **Δ_SIM(AC-v2-NormRatio − SA):** +0.0168
- **Δ_SIM(AC-v2-NormRatio − AC-v2-WeightCap):** -0.0003

## Conclusion

The per-layer norm-ratio clamp is functionally **identical** to the simpler per-adapter weight cap on the current MD benchmark (Δ = -0.0003). For most questions, the un-clamped adapter activation vector `||m||` is already small relative to the base model activation `||z||`. Consequently, the norm-ratio scalar `γ` evaluates to 1.0 at almost all layers, producing identical outputs to the baseline geometric addition.

The clamp does not meaningfully alter the compositional dynamics of Qwen2.5-1.5B with these specific LoRA experts. This confirms that the v2 H5 failure (Clamped ≡ Unclamped) was a genuine property of the model/adapter representations, not merely an artifact of the `weight_cap` implementation.
