# Experiment Card 01 — Synapta v1: Prompt-Level Multi-Adapter Composition (Apple Silicon)

## 1. Research question and hypothesis

Does prompt-level multi-adapter composition with bounded mixing on a small base model outperform single-adapter routing on multi-domain queries?

- **H1 (compositional gain):** $\Delta_\text{SIM}(\text{AdaptiveClamp} - \text{SingleAdapter}) > +0.05$ on the v1 internal benchmark.
- **H2 (PPL minimization):** $\text{PPL}(\text{AC}) \leq \text{PPL}(\text{SA})$.
- **H3 (clamp necessity):** Unclamped activation mixing causes catastrophic representation collapse.

## 2. Dataset and task definition

- **Dataset:** v1 internal multi-domain benchmark, 100 questions, 20 distinct domains (mathematics, medical_diagnosis, philosophy, sanskrit_linguistics, maritime_law, etc.).
- **Construction:** synthetically templated single-domain questions; one domain per question.
- **No leakage safeguard claimed at v1**: questions were authored internally and overlap with training distribution is not document-disjoint.

## 3. Model and configuration

- **Base:** Qwen2.5-1.5B-Instruct-4bit MLX format (Apple Silicon UMA).
- **Adapters:** 20 domain-specific LoRA experts.
- **Composition methods:**

| Method | $K$ | Clamp $c$ | Routing |
|---|---:|---:|---|
| Baseline | 0 or 1 | 0.001 | none |
| SingleAdapter | 1 | 0.5 | CoT |
| AdaptiveClamp | 2 | 0.5 | CoT |
| UnclampedMix | 2 | 999 | CoT |

- **Hardware:** Apple M-series, MLX runtime.

## 4. Evaluation protocol

- **Primary metric:** semantic similarity via `sentence-transformers` against gold reference.
- **Secondary:** perplexity (PPL), latency (s).
- **Pre-registered thresholds:** $\Delta_\text{SIM} > +0.05$ (H1), $\text{PPL}(\text{AC}) \leq \text{PPL}(\text{SA})$ (H2), latency overhead $\leq 10\%$.

## 5. Main quantitative results

| Method | Avg semantic similarity | Avg PPL | Avg latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6196 | 64.5 | 2.803 |
| SingleAdapter | **0.6223** | 60.9 | 2.695 |
| AdaptiveClamp | 0.6106 | 58.0 | 2.672 |
| UnclampedMix | 0.5573 | 51.2 | 2.511 |

| Hypothesis | Measured | Threshold | Verdict |
|---|---|---|---|
| H1 ($\Delta_\text{SIM}$ AC − SA) | $-0.011$ | $> +0.05$ | **FAIL** |
| H1 ($\Delta_\text{SIM}$ AC − Baseline) | $-0.009$ | $> +0.05$ | **FAIL** |
| H2 (PPL AC vs SA) | $58.0 < 60.9$ | $\leq$ | PASS |
| H2 (PPL AC vs Baseline) | $58.0 < 64.5$ | $\leq$ | PASS |
| H3 (UnclampedMix collapse) | 8\% prompts $< 0.1$ similarity | qualitative | PASS |
| Latency overhead (AC vs SA) | $-0.7\%$ | $\leq 10\%$ | PASS |

Per-domain breakdown shows mixed signals: AC won by $+0.030$ on `MEDICAL_DIAGNOSIS`, $+0.044$ on `MATHEMATICS`; lost by $-0.145$ on `MARITIME_LAW`, $-0.035$ on `SANSKRIT_LINGUISTICS`.

## 6. Negative results, limitations, and bugs

- **H1 is the headline failure:** bounded $K=2$ prompt-level composition does NOT beat single-adapter routing on the v1 benchmark.
- **Per-domain noise:** the lift/drop pattern suggests that adding a redundant adapter on a single-domain question injects noise. The v2 multi-domain benchmark (`02_synapta_v2_multidomain.md`) was constructed in response.
- **Methodology critique acknowledged in `docs/MASTER_KNOWLEDGE_BASE.md`:** the v1 evaluation was on single-domain synthetic templates, so "compositional gain" was not meaningfully tested at v1.

## 7. Artifact map

PRIMARY:
- `results/decision_summary.md`
- `results/v2_decision_summary.md` (cross-references v1 in headers)
- `results/tested_hypotheses_and_results.md`

SECONDARY:
- `docs/MEWTWO_RESEARCH_PAPER_DRAFT.md` (narrative)
- `docs/MASTER_KNOWLEDGE_BASE.md` (Phase-1 narrative)
- `docs/tested_hypotheses_and_results.md` (synthesis)

MISSING / off-device:
- 20 expert LoRA safetensors for full Apple Silicon reproduction (see `missing_artifacts.md` item 2).
- Per-question raw output JSONL from v1 run (search for `results/v1_*` returned no files; the aggregate scores in `decision_summary.md` are the only surviving artifact).
