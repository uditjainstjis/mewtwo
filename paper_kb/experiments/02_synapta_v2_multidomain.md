# Experiment Card 02 — Synapta v2: Multi-Domain Composition

## 1. Research question and hypothesis

Does prompt-level multi-adapter composition outperform single-adapter routing when queries genuinely require knowledge from multiple distinct domains?

- **H1 (SD non-inferiority):** $\Delta_\text{SIM}(\text{AC-v2} - \text{SA}) \geq -0.005$ on single-domain split.
- **H2 (MD compositional gain):** $\Delta_\text{SIM} > +0.03$ on multi-domain split.
- **H3 (PPL preservation):** PPL(AC-v2) $\leq$ PPL(SA) on both splits.
- **H4 (latency bound):** $\Delta_\text{LAT}(\text{AC-v2} - \text{SA}) / \text{SA} \leq 15\%$.
- **H5 (clamp necessity on MD):** clamped > unclamped strictly.

## 2. Dataset and task definition

- **Dataset:** v2 dual-split benchmark.
  - SD (single-domain): 100 questions, one required adapter each.
  - MD (multi-domain): 40 questions, two required adapters each (e.g., Sanskrit Linguistics + Ancient History).
- **Construction:** internally authored; required-adapter labels available for oracle routing.
- **Total inferences:** 560 (140 questions × 4 methods).

## 3. Model and configuration

- **Base:** Qwen2.5-1.5B-Instruct-4bit MLX.
- **Adapters:** subset of the 20-expert pool used in v1.
- **Methods:**

| Method | $K$ | Clamp $c$ | Routing |
|---|---:|---:|---|
| Baseline | 0 | 0.001 | none |
| SingleAdapter | 1 | 0.5 | CoT (real) |
| AdaptiveClamp-v2 | 2 | 0.5 | Oracle (required adapters) |
| UnclampedMix-v2 | 2 | 999 | Oracle |

## 4. Evaluation protocol

- **Metrics:** semantic similarity, PPL, latency (s).
- **Splits reported separately:** SD ($n=100$), MD ($n=40$).
- **Pre-registered thresholds:** as listed in §1.

## 5. Main quantitative results

### SD split ($n=100$)
| Method | Sim | PPL | Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6090 | 64.5 | 3.700 |
| SingleAdapter | 0.6064 | 60.9 | 3.571 |
| AdaptiveClamp-v2 | 0.6058 | 57.9 | 3.657 |
| UnclampedMix-v2 | 0.6041 | 52.3 | 3.623 |

### MD split ($n=40$)
| Method | Sim | PPL | Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6473 | 12.7 | 4.059 |
| SingleAdapter | 0.6334 | 12.7 | 4.057 |
| AdaptiveClamp-v2 | **0.6505** | 12.6 | 4.090 |
| UnclampedMix-v2 | 0.6505 | 12.6 | 4.100 |

### Hypothesis verdicts
| H | Measured | Threshold | Verdict |
|---|---|---|---|
| H1 (SD non-inferiority) | $-0.0006$ | $\geq -0.005$ | PASS |
| H2 (MD compositional gain) | $+0.0171$ | $> +0.03$ | **FAIL (sub-threshold positive)** |
| H3 (PPL preservation SD) | $57.9 < 60.9$ | $\leq$ | PASS |
| H3 (PPL preservation MD) | $12.6 < 12.7$ | $\leq$ | PASS |
| H4 (latency bound) | $+1.9\%$ | $\leq 15\%$ | PASS |
| H5 (clamp necessity MD) | $0.0000$ (clamped $\equiv$ unclamped) | $> 0$ strict | **FAIL** |

## 6. Negative results, limitations, and bugs

- **H2 sub-threshold:** $+0.0171$ is directionally positive but below the pre-registered $+0.03$ bar. The "Synapta wins" headline based on this result is **not paper-safe**.
- **H5 failure:** clamped and unclamped MD outputs are identical to 4 decimal places. This is investigated in `03_clamp_ablation_norm_ratio.md` — the per-layer activation norm of the adapter contribution is dominated by the base model norm, so the clamp is operationally inactive.
- **Sub-threshold positive interpretation:** "composition helps, but not at the magnitude originally claimed" — this is the honest framing recommended in `docs/MASTER_KNOWLEDGE_BASE.md`.

## 7. Artifact map

PRIMARY:
- `results/v2_decision_summary.md`
- `results/v2_both_raw.jsonl` (raw 560 inference rows)

SECONDARY:
- `docs/v2_prereg.md` (pre-registration document)
- `docs/MEWTWO_RESEARCH_PAPER_DRAFT.md`

MISSING / off-device:
- per-domain breakdown JSON (only aggregate is present)
