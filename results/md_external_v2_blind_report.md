# External MD Evaluation Report

Date: 2026-04-08

## Scope

- External dataset: `data/multidomain_eval_claude_external_v2_100.json`
- Soft-metric run: 100 items
- Blind-judge subset: 30 stratified items, 3 per section
- Judge: `claude-4.6-sonnet` via Perplexity proxy

## 100-Item Soft Metrics

| System | Semantic Sim | Token F1 | Latency (s) | Rubric Coverage |
| --- | ---: | ---: | ---: | ---: |
| weighted_merge | 0.6592 | 0.2719 | 4.263 | 0.1261 |
| late_layer_injection | 0.6594 | 0.2715 | 3.890 | 0.1230 |
| sequential_reverse | 0.6623 | 0.2734 | 4.605 | 0.1338 |
| mistral | 0.6907 | 0.2917 | 10.654 | 0.1683 |

Soft-metric read:

- `sequential_reverse` looked like the best Qwen by semantic similarity and rubric coverage.
- `late_layer_injection` was the fastest Qwen.
- `mistral` led on quality metrics but was about 2.3x to 2.7x slower.

## 30-Item Blind Pairwise Results vs Mistral

| Qwen Method | Qwen Wins | Mistral Wins | Ties | Avg Qwen Score | Avg Mistral Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| weighted_merge | 6 | 23 | 1 | 3.767 | 5.300 |
| late_layer_injection | 4 | 26 | 0 | 3.500 | 5.333 |
| sequential_reverse | 4 | 25 | 1 | 3.533 | 5.300 |

Blind-judge read:

- All three Qwen variants lost clearly to `mistral`.
- `weighted_merge` was the least-bad Qwen method under blind judging.
- The soft-metric leader, `sequential_reverse`, did not translate into blind-judge wins.

## Main Inference

The current external evaluation does not support a "Qwen beats Mistral on answer quality" claim.

What it does support:

- The routed Qwen system is materially faster.
- Blind judging is more skeptical than embedding-based metrics.
- Method ranking changes when evaluation moves from soft similarity to blind correctness-focused judgment.
- `weighted_merge` is the strongest current Qwen baseline under blind judgment, even though `sequential_reverse` looked best on soft metrics.

## Honest Claim Boundary

Research-safe:

- "On a 100-item externally authored MD benchmark, Qwen routing methods are substantially faster than Mistral, but blind pairwise judging on a stratified 30-item subset still favors Mistral on answer quality."
- "Semantic similarity and token-overlap metrics overstated the strength of sequential routing; blind judging changed the Qwen method ranking."

Startup-safe:

- "Our small routed system offers materially better latency and controllable expert composition, but it does not yet beat a larger Mistral baseline on externally judged answer quality."
- "The current edge is efficiency and architecture control, not superior final-answer correctness."

Not supported:

- "We beat Mistral overall."
- "Sequential routing is the best method."
- Any general-quality superiority claim from the current blind evidence.

## Section-Level Note

Across all three Qwen methods, `Cross Domain Synthesis` was the only section with repeated Qwen wins against Mistral. That is interesting, but it is only `n=3` per section in the blind subset, so it is a hypothesis, not a claim.

## Artifacts

- `results/md_external_v2_comparison_summary.json`
- `results/md_external_v2_comparison_summary.md`
- `results/md_pairwise_seqrev_vs_mistral_v2_strat30_summary.json`
- `results/md_pairwise_latelayer_vs_mistral_v2_strat30_summary.json`
- `results/md_pairwise_merge_vs_mistral_v2_strat30_summary.json`
- `results/md_external_v2_soft_vs_blind_summary.json`

## Recommended Next Step

If the goal is product positioning, improve Qwen answer quality before making superiority claims.

If the goal is research, keep this result and use it to motivate:

- better judgeable benchmark construction
- blind evaluation over embeddings
- further ablations on routing and answer truncation
