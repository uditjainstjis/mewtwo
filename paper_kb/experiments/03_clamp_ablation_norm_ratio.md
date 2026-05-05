# Experiment Card 03 — Clamp Ablation: Per-Layer Norm-Ratio vs Per-Adapter Weight-Cap

## 1. Research question and hypothesis

The H5 failure on the v2 MD benchmark (clamped $\equiv$ unclamped) could be either (a) a property of the model/adapter geometry, or (b) an artefact of the per-adapter weight-cap clamp implementation. Does switching to a true per-layer activation norm-ratio clamp recover the H5 effect?

Clamp formula tested:
\[
\gamma_\ell = \min\Big(1, \, c \cdot \frac{\|z_\ell\|_2}{\|m_\ell\|_2}\Big)
\]
where $z_\ell$ is the base-model activation and $m_\ell$ is the LoRA contribution at layer $\ell$.

- **H (norm-ratio recovers H5):** $\Delta_\text{SIM}(\text{NormRatio} - \text{WeightCap}) > 0$.

## 2. Dataset and task definition

- **Dataset:** v2 MD split ($n=40$).
- **Total inferences:** 120 (40 questions × 3 methods).

## 3. Model and configuration

- Base: Qwen2.5-1.5B-Instruct-4bit MLX.
- Adapters: same MD-required pool as v2.
- Methods:
  - SingleAdapter ($c=0.5$, weight_cap)
  - AC-v2-WeightCap ($c=0.5$, weight_cap, $K=2$)
  - **AC-v2-NormRatio** ($c=0.5$, norm_ratio, $K=2$)

## 4. Evaluation protocol

- Same metrics as v2 (sim, PPL, latency).
- Clamp triggered per-layer per-adapter rather than once per adapter.

## 5. Main quantitative results

| Method | Clamp | Avg Sim | Avg PPL | Avg Latency (s) | Avg $K$ |
|---|---|---:|---:|---:|---:|
| SingleAdapter | weight_cap | 0.6334 | 12.7 | 4.008 | 1.00 |
| AC-v2-WeightCap | weight_cap | 0.6505 | 12.6 | 4.055 | 2.00 |
| AC-v2-NormRatio | norm_ratio | **0.6502** | 12.6 | 4.221 | 2.00 |

Computed deltas:
- $\Delta_\text{SIM}(\text{WeightCap} - \text{SA}) = +0.0171$ (matches v2 run)
- $\Delta_\text{SIM}(\text{NormRatio} - \text{SA}) = +0.0168$
- $\Delta_\text{SIM}(\text{NormRatio} - \text{WeightCap}) = -0.0003$

## 6. Negative results, limitations, and bugs

- **Norm-ratio is functionally identical to weight-cap on this base+adapter set.** $\Delta = -0.0003$ is within sampling noise.
- **Why:** for most questions, the un-clamped adapter activation vector $\|m_\ell\|$ is already small relative to the base-model activation $\|z_\ell\|$. The norm-ratio scalar $\gamma_\ell$ evaluates to 1.0 at almost all layers, producing identical outputs to the simpler additive composition.
- **Implication:** the v2 H5 failure (clamped $\equiv$ unclamped) is a genuine property of the Qwen-1.5B + 20-expert geometry, not an implementation artefact.
- **Honest framing for paper:** clamp is unnecessary on this base+adapter set, but cannot be claimed to be unnecessary in general. Larger models or higher-rank adapters may produce non-trivial $\|m_\ell\|$ relative to $\|z_\ell\|$ and reactivate the clamp's role.

## 7. Artifact map

PRIMARY:
- `results/v2_clamp_ablation_summary.md`
- `results/v2_md_clamp_ablation.jsonl` (raw 120 inference rows)

SECONDARY:
- `docs/MASTER_KNOWLEDGE_BASE.md` (clamp discussion)
