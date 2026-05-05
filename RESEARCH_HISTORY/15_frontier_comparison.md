# Frontier Comparison — Synapta vs Anthropic Claude (n=15 paired)

**Date:** 2026-05-04
**Source artefacts:**
- `synapta_src/data_pipeline/11_frontier_comparison.py` (subagent harness)
- `results/frontier_comparison/subagent_results.jsonl` (35 rows)
- `results/frontier_comparison/stratified_50.jsonl`
- `docs/recent/FRONTIER_COMPARISON_FINDINGS.md`

## Why subagents (not API calls)
The user does not have OpenAI / Anthropic API keys for paid evaluation. Anthropic Claude Code's subagent feature was used as a free path: each subagent receives the full RBI/SEBI question + context and returns Claude's answer. 15 stratified questions × 3 models (Claude Opus, Claude Sonnet, repeat passes) = 35 result rows.

## Setup
- Sample: 15 hand-curated questions from the held-out BFSI eval set, balanced across Tier 2/3 and RBI/SEBI.
- Each question delivered to subagent with the same context paragraph and same prompt format used for Synapta evaluation.
- Outputs scored with the same substring + token F1 harness.

## Results

| Model | Substring | Token F1 |
|---|---|---|
| Synapta-Nemotron-30B + bfsi_extract | **87\%** | 0.38 |
| Anthropic Claude Opus | 7\% | **0.65** |
| Anthropic Claude Sonnet | 27\% | 0.49 |

## Reading

**Synapta wins substring by a wide margin (+60 to +80 pp).** This is the citation-faithful production metric for compliance use cases — a compliance officer must paste the regulator's exact words into a memo. Synapta's adapter has been trained to quote verbatim from the context paragraph; Claude has been trained for semantic polish.

**Claude wins token F1 by 20-30 pp.** This reflects Claude's strength at semantic paraphrase: it produces a correct, well-structured answer that conveys the same information without quoting verbatim. For chat / explanatory deliverables this is the right metric.

## Caveats
- $n=15$ is too small for confident statistical claims; this is **directional only**.
- The two metrics measure different deliverables; we are not claiming Claude is bad at this task. We are claiming Synapta is configured for a different deliverable (citation-faithful production output) and dominates on that specific metric.
- Subagent latency can affect generation parameters (temperature, length) compared to a calibrated API call. We did not control for these.

## Open path: full $n=60$ frontier comparison
Re-running on the released Synapta Benchmark v1 with proper API access ($\approx \$3$ at frontier API rates) would provide:
- Larger paired sample for proper McNemar.
- Controlled decoding settings.
- Direct comparison row for the public benchmark's seed baseline table.

Currently gated on API key availability.

## Files
- `synapta_src/data_pipeline/11_frontier_comparison.py`
- `results/frontier_comparison/subagent_results.jsonl`
- `results/frontier_comparison/stratified_50.jsonl`
- `docs/recent/FRONTIER_COMPARISON_FINDINGS.md`
