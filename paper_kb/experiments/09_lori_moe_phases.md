# Experiment Card 09 — LoRI-MoE: Composite Routing Across Phases

## 1. Research question and hypothesis

Does LoRI-MoE (frozen shared random projection + sparse trainable expert matrices + a learned router) on Qwen-2.5-1.5B outperform single-best-adapter routed baselines on standard benchmarks? Specifically, does a composite top-1 routed configuration exceed the best individual adapter on the same benchmark?

## 2. Dataset and task definition

- **Phase 1 baselines:** GSM8K ($n=200$), ARC-Challenge ($n=200$), MMLU ($n=200$).
- **Phase 2 single-adapter:** each domain (math, code, science, legal, medical) evaluated singly.
- **Phase 3 composite:** top-1 routed composite evaluated on GSM8K, ARC, MMLU.

## 3. Model and configuration

- **Base:** Qwen-2.5-1.5B-Instruct.
- **Adapters:** 5 LoRI experts: code, legal, math, medical, science.
- **Routing in Phase 3:** prompt-keyword classifier (early prototype, NOT token-level).
- **Adapter location:** `adapters/lori_moe/lori-qwen2.5-1.5b-{code,legal,math,medical,science}/`

## 4. Evaluation protocol

- **GSM8K:** exact-match.
- **ARC-Challenge, MMLU:** accuracy.

## 5. Main quantitative results

### Phase 1 — base zero-shot
| Benchmark | Score | $n$ |
|---|---:|---:|
| GSM8K | 26.0\% | 200 |
| ARC-Challenge | 76.5\% | 200 |
| MMLU | 56.5\% | 200 |

### Phase 2 — single adapter
| Adapter | GSM8K | ARC |
|---|---:|---:|
| math | **53.0\%** | --- |
| code | 3.5\% | --- |
| science | 1.0\% | 65.0\% |
| legal | 15.0\% | **77.5\%** |
| medical | 1.0\% | 71.5\% |

### Phase 3 — composite top-1 routed
| Benchmark | Composite | Best single | $\Delta$ |
|---|---:|---:|---:|
| GSM8K | **4.0\%** | 53.0\% | $-49$ pp catastrophic |
| ARC | 72.0\% | 77.5\% | $-5.5$ pp |
| MMLU | 53.0\% | --- | --- |

## 6. Negative results, limitations, and bugs

- **Composite top-1 routing did NOT exceed the single best expert** on any tested benchmark.
- **GSM8K catastrophic regression:** composite scored 4\% (presumably the router selected non-math adapters on math queries). This is the same "routing failure dominates" pattern as `04_routing_gap_oracle_vs_real.md` on the Apple Silicon side.
- **Phase 2 cross-domain damage:** the medical and science adapters destroy GSM8K capability ($-25$ pp from base). This is a Code-Paradox-adjacent finding for non-code domains.
- **Token-level routed LoRI-MoE is NOT yet evaluated end-to-end.** All Phase 3 numbers above are prompt-level routed. The token-level routed variant exists in code (`synapta_src/src/lori_moe/`) but lacks a benchmark suite output. **This is the largest open gap in cluster D and is recorded in `missing_artifacts.md` item 1.**
- **The "saved router accuracy" referenced in older docs reflects training-set classification, not held-out multi-domain routing.** Treat as Category 3 unless re-validated.

## 7. Artifact map

PRIMARY:
- `results/lori_moe/all_results.json`
- `results/lori_moe/phase1_baselines.json`
- `results/lori_moe/phase2_single_adapter.json`
- `results/lori_moe/phase3_composite.json`
- `results/lori_moe/phase4_interference.json`
- `adapters/lori_moe/lori-qwen2.5-1.5b-{code,legal,math,medical,science}/`

SECONDARY:
- `docs/MEWTWO_RESEARCH_PAPER_DRAFT.md` (LoRI-MoE narrative)

MISSING / off-device:
- token-level routed LoRI-MoE benchmark output (gates the strongest LoRI-MoE claim)
- per-question raw JSONL for Phase 3 (only aggregates present)
