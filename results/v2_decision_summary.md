# v2 Pre-Registered Hypothesis Decision Summary

**Date:** 2026-03-29
**Experiment:** 560 real inferences (140 questions × 4 methods)
**Log file:** `results/v2_both_raw.jsonl`

## Methods

| Method | K | Clamp | Routing |
|--------|---|-------|---------|
| Baseline | 0 | 0.001 | None |
| SingleAdapter | 1 | 0.5 | CoT (real) |
| AdaptiveClamp-v2 | 2 | 0.5 | Oracle (required_adapters) |
| UnclampedMix-v2 | 2 | 999 | Oracle (required_adapters) |

## Aggregate Results

### Single-Domain Split (SD, 100 questions)

| Method | Avg Sim ↑ | Avg PPL ↓ | Avg Lat (s) |
|--------|-----------|-----------|-------------|
| Baseline | 0.6090 | 64.5 | 3.700 |
| SingleAdapter | 0.6064 | 60.9 | 3.571 |
| AdaptiveClamp-v2 | 0.6058 | 57.9 | 3.657 |
| UnclampedMix-v2 | 0.6041 | 52.3 | 3.623 |

### Multi-Domain Split (MD, 40 questions)

| Method | Avg Sim ↑ | Avg PPL ↓ | Avg Lat (s) |
|--------|-----------|-----------|-------------|
| Baseline | 0.6473 | 12.7 | 4.059 |
| SingleAdapter | 0.6334 | 12.7 | 4.057 |
| AdaptiveClamp-v2 | 0.6505 | 12.6 | 4.090 |
| UnclampedMix-v2 | 0.6505 | 12.6 | 4.100 |

## Hypothesis Verdicts

### H1: SD Non-Inferiority — **PASS** ✅
- **Metric:** Δ_SIM(AC-v2 − SA) = −0.0006
- **Threshold:** ≥ −0.005
- **Interpretation:** AdaptiveClamp-v2 is effectively identical to SingleAdapter on SD. On SD questions, the oracle routing gives K=1 (only one domain), so AC-v2 degenerates to SA. This confirms that the gated/oracle approach causes no harm on single-domain queries.

### H2: MD Compositional Gain — **FAIL** ❌
- **Metric:** Δ_SIM(AC-v2 − SA) = +0.0171
- **Threshold:** > +0.03
- **Interpretation:** AC-v2(K=2) does outperform SA(K=1) on multi-domain queries (+0.0171), showing a *directionally positive* composition effect, but the gain falls short of the pre-registered +0.03 threshold. This is a **partial positive**: composition helps, but not enough to declare statistical significance at the preregistered level.

### H3: PPL Preservation — **PASS** ✅ (both splits)
- **SD:** PPL(AC-v2) = 57.9 ≤ PPL(SA) = 60.9 ✅
- **MD:** PPL(AC-v2) = 12.6 ≤ PPL(SA) = 12.7 ✅
- **Interpretation:** Multi-adapter composition improves (lowers) perplexity on both splits.

### H4: Latency Bound — **PASS** ✅
- **Metric:** Δ_LAT = +1.9%
- **Threshold:** ≤ 15%
- **Interpretation:** Negligible latency overhead from K=2 oracle routing, well within acceptable bounds.

### H5: Clamp Necessity (MD) — **FAIL** ❌
- **Metric:** Δ_SIM(clamped − unclamped) = 0.0000
- **Threshold:** > 0 (clamped should be better)
- **Interpretation:** On MD questions with oracle routing, clamped (c=0.5) and unclamped (c=999) produce identical results. This suggests that with correct oracle routing and equal adapter weights (0.5 each), the adapter contributions are small enough relative to the base model that the clamp never activates. The per-adapter weight cap of min(weight, clamp) = min(0.5, 0.5) = 0.5 for clamped, and min(0.5, 999) = 0.5 for unclamped — they are identical! This is a methodological artifact of the per-adapter weight-cap mechanism, not the per-layer norm-ratio clamp.

## Key Observations

1. **Composition signal exists but is below threshold.** The +0.0171 improvement is directionally consistent with the composition hypothesis. A larger model, better adapters, or the true per-layer norm-ratio clamp (not the per-adapter weight cap) might push this over the threshold.

2. **H5 is an artifact of the weight-cap mechanism.** Because oracle routing assigns weight=0.5 per adapter, and the clamp is also 0.5, the weight cap never activates. To properly test H5, the backend would need the per-layer norm-ratio clamp γ_l = min(1, c·||z_l||/||m_l||), which operates on activation norms, not routing weights.

3. **Per-question variance is high.** Some MD questions show large gains (md_32: +0.303, md_19: +0.109, md_20: +0.083) while others show losses (md_09: −0.125, md_25: −0.061). This suggests composition effects are domain-pair-dependent.
