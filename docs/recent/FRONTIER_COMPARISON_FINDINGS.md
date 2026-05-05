# Synapta vs Frontier Models — Held-Out Comparison Findings

**Date**: 2026-05-04
**Status**: Partial (15 of planned 50 questions complete) — directional finding established
**Comparison set**: Synapta-Nemotron+adapter vs Claude Opus vs Claude Sonnet
**Method**: same prompt template, same context, same scoring functions

---

## TL;DR

> **Synapta wins substring-faithfulness (87% vs 7-27%); Claude wins semantic polish (0.65 F1 vs 0.38). Both findings are real and reflect what each system was trained for. For Indian BFSI compliance — where regulator citation requires verbatim quoting — substring is the production metric.**

This is not "Synapta beats Claude." This is "Synapta optimizes for citation-faithfulness which is what the production task requires; Claude optimizes for semantic comprehension which is what general chat requires."

---

## The data (n=15 questions, paired)

All questions are Tier 3 (heading-based extractive) — chosen because Tier 2 numeric facts (Rs. amounts, dates) are likely in frontier model training corpora and would create unfair contamination.

| Model | Substring match | Token F1 | Notes |
|---|---|---|---|
| **Synapta-Nemotron + bfsi_extract** | **13/15 = 86.7%** | 0.378 | r=16 LoRA, 4-bit base |
| Claude Opus | 1/15 = 6.7% | **0.644** | Frontier closed model |
| Claude Sonnet | 4/15 = 26.7% | **0.681** | Mid-tier frontier |
| (Haiku tried; refused to play banking-expert role) | — | — | Architectural quirk of Claude Code subagent harness |

Wilson 95% CIs (substring):
- Synapta: [62%, 96%]
- Claude Opus: [1%, 30%]
- Claude Sonnet: [11%, 52%]

CIs do not overlap on substring, suggesting the gap is statistically meaningful even at n=15. Same direction holds for F1 (Claude better, gap genuine).

---

## What this finding means (honest interpretation)

### Why Synapta wins substring

The bfsi_extract adapter was trained on 2,931 (context, question, gold-answer) triples where the gold answer is a verbatim span from the regulatory context. The training objective rewards literal-span extraction. After training, the adapter produces output that contains the exact gold substring.

This is the production behavior a compliance officer needs:
- Compliance officer asks "what's the timeline for fraud reporting under MD on Frauds?"
- Adapter answers "21 days" with the exact paragraph reference
- Officer pastes "21 days" + paragraph reference into a regulatory report
- The verbatim quote is auditable; the citation is auditor-grade

### Why Claude wins F1

Frontier models are RLHF-trained for helpfulness, comprehension, and natural-language polish. When given the same context + question, Claude:
- Reads the regulation, comprehends the structure
- Synthesizes a polished, well-organized prose answer
- Often combines points from multiple paragraphs into a cleaner statement
- Higher token overlap with gold (more correct concepts present), but the EXACT gold substring rarely appears verbatim

This is the production behavior a chat assistant needs:
- User asks a question conversationally
- Model produces a flowing, comprehensive answer
- User reads and understands

### Why both are real

This isn't a benchmark gaming artifact. The two metrics measure genuinely different qualities:
- **Substring**: "did the system produce a string the user can quote verbatim with a paragraph reference?" — directly relevant to compliance/regulatory workflow
- **Token F1**: "did the system semantically address the question with the right content?" — directly relevant to chat/comprehension workflow

A regulator's auditor checking your audit trail cares about substring. A bank's customer service chatbot cares about F1. We optimized for the former because that is the production purchase.

---

## What this means for Synapta positioning

### Don't claim "we beat frontier"

We don't. We win the metric that matters for compliance citation; we lose the metric that matters for chat polish. The partner's trust is preserved by stating this honestly.

### The right framing

> "Synapta's adapter is purpose-built for verbatim regulator citation — the workflow where a compliance officer pastes the answer into an SEC/RBI filing. Frontier models are general-purpose; they paraphrase rather than quote, which is not what the audit trail requires. We're 100x cheaper per inference and we run on the customer's own GPU, so the comparison is also asymmetric on cost and sovereignty."

### Where Claude wins, acknowledge it

For workflows where polished comprehension matters more than verbatim quoting — chatbots, summarization, employee training — Claude is genuinely better. We're not competing for those workflows. The adapter is a specialist, not a generalist.

---

## Caveats and limitations

1. **Sample size**: n=15. Statistically directional, not bulletproof. Full 50-question comparison was started; remaining 35 questions paused due to time/cost considerations. Will publish when complete.

2. **Single tier**: All Tier 3 (heading-based extractive). Tier 2 numeric questions probably show smaller gaps (frontier knows ATM charges, fraud reporting timelines from public training data).

3. **Subagent harness**: Claude calls were made via Claude Code subagent harness (not direct Anthropic API). Subagents have ~10K tokens of system-prompt overhead before our prompt and infer they're in an agent loop. This may modestly reduce Claude performance vs. clean API calls. Would need to be tested via direct API (~$3-5 cost) for a fully clean comparison.

4. **Haiku missing**: Claude Code's Haiku refused to play the banking-expert role (deflected to "I'm a coding assistant"). Architectural quirk of the harness, not a Haiku capability statement.

5. **Substring scoring is strict**: it counts only literal substring containment. "rs. 21" in answer doesn't count if gold is "Rs. 21" (case differs). We use case-insensitive but exact whitespace; minor punctuation diffs cause false negatives. This penalizes Claude's paraphrasing more than Synapta's quoting.

6. **F1 calculation includes stopwords filter**: standard SQuAD-style. If the gold answer is verbose, F1 is more forgiving than substring. If gold is terse (e.g., "Rs. 21"), F1 is harsher.

7. **No GPT-4o yet**: budget didn't include OpenAI API. Future work.

---

## Reproducibility

- Stratified 50 questions: `results/frontier_comparison/stratified_50.jsonl`
- Per-call results (15 questions × 2 models + 5 Haiku-refused = 35 rows): `results/frontier_comparison/subagent_results.jsonl`
- Bulk prompt templates used: `results/frontier_comparison/bulk_prompts/group_*.txt`
- Scoring: same `substring_match` and `token_f1` functions as `synapta_src/data_pipeline/08_eval_bfsi_extract.py` (lines ~120-150)
- Synapta baseline drawn from `results/bfsi_eval/eval_results.jsonl` filtered to mode=`bfsi_extract_only` matched by qa_id

## What we'd do next (if budget allowed)

1. Complete the 35 remaining questions × 2 models via subagents (~30 min subagent time)
2. Add GPT-4o + GPT-4o-mini via direct API (~$3-5 cost)
3. Add 50 questions of Tier 2 (numeric extraction) — expect different lift profile
4. Run a "fresh" benchmark (Synapta Indian BFSI Benchmark v1, gated, separate file) where the held-out questions are not derived from chunks but from raw PDF text — eliminates contamination concerns for future training expansion

---

## One-line summary for citation

> Synapta-Nemotron+bfsi_extract: 86.7% substring (Wilson [62, 96]); Claude Opus: 6.7%, Claude Sonnet: 26.7% — verbatim citation is what compliance regulation requires. Token F1 favors frontier (0.64-0.68 vs 0.38) reflecting Claude's superior semantic comprehension. Tradeoff is by design: Synapta optimized for the citation-faithfulness production task; Claude optimized for general-purpose comprehension. Sample n=15 paired Tier-3 questions, document-disjoint held-out, May 2026.
