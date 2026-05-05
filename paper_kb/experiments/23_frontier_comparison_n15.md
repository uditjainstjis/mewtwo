# Experiment Card 23 — Frontier Comparison: Synapta vs Anthropic Claude ($n=15$ subagent)

PKB wrapper around `RESEARCH_HISTORY/15_frontier_comparison.md`.

## 1. Research question
Directional comparison: does Synapta's bfsi_extract adapter dominate frontier models on the citation-faithful substring metric for Indian regulatory QA?

## 2. Dataset
- 15 hand-curated questions stratified across Tier 2/3 and RBI/SEBI.
- Synapta and Claude both receive identical context paragraph and prompt format.

## 3. Model
- Synapta: Nemotron-30B 4-bit + bfsi_extract.
- Frontier: Anthropic Claude Opus + Claude Sonnet, accessed via subagent harness (no API key required; user does not have API budget).

## 4. Evaluation
- Substring match + token F1.

## 5. Results

| Model | Substring | Token F1 |
|---|---:|---:|
| Synapta-Nemotron-30B + bfsi_extract | **87\%** | 0.38 |
| Anthropic Claude Opus | 7\% | **0.65** |
| Anthropic Claude Sonnet | 27\% | 0.49 |

## 6. Negatives + caveats
- $n=15$ is too small for statistical claims; **directional only**.
- Synapta wins substring by 60-80 pp (citation-faithful production metric); Claude wins F1 by 27 pp (semantic polish).
- Two metrics measure different deliverables; the paper-safe interpretation is "Synapta is configured for citation-faithful production output and dominates that metric; Claude is configured for semantic polish."
- Subagent latency may have affected generation parameters not fully controlled.

## 7. Artifact map
PRIMARY:
- `synapta_src/data_pipeline/11_frontier_comparison.py`
- `results/frontier_comparison/subagent_results.jsonl` (35 rows)
- `results/frontier_comparison/stratified_50.jsonl`

SECONDARY:
- `RESEARCH_HISTORY/15_frontier_comparison.md`
- `docs/recent/FRONTIER_COMPARISON_FINDINGS.md`
