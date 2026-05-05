# BFSI Recall Adapter — No-Context Recall Task

**Date:** 2026-05-04
**Source artefacts:**
- `synapta_src/data_pipeline/10_build_recall_dataset.py`
- `synapta_src/data_pipeline/12_train_bfsi_recall.py`
- `synapta_src/data_pipeline/13_eval_bfsi_recall.py`
- `adapters/nemotron_30b/bfsi_recall/best/`
- `results/bfsi_recall_eval/eval_results.jsonl`
- `results/bfsi_recall_eval/summary.json`

## Goal
Test whether the same recipe (same base, same hyperparameters, same evaluation protocol) transfers to a structurally different task: **no-context recall** instead of context-grounded extraction.

This is the second-data-point check that a single result is not a fluke.

## Task
Questions are presented WITHOUT a context paragraph. The model must answer from parametric knowledge (recall what the regulation said). Substring match is uninformative on this task — gold answers like "Rs. 21" rarely appear verbatim in verbose recall outputs. Token F1 is the primary metric.

## Setup
- Same base: Nemotron-30B 4-bit NF4.
- Same LoRA: $r=16$, $\alpha=32$, attn-only target modules (smaller surface than extract).
- Same optimiser, same lr, same epoch count.
- MAX_LEN=384 (shorter, since no context).
- Training data: recall-formatted Q&A pairs derived from the same train documents.

## Evaluation — paired F1

$n=214$ paired held-out questions.

| Mode | Mean F1 | Lift | Wilcoxon vs base |
|---|---|---|---|
| Base | 0.158 | --- | --- |
| + bfsi_recall | 0.219 | $+0.061$ ($+38.4\%$ rel.) | $\mathbf{p = 1.50 \times 10^{-16}}$ |
| Format Guard (4 adapters) | 0.219 | $+0.061$ | $p = 0.55$ vs adapter direct |

74.3\% of questions show adapter F1 > base F1.

## Substring is the wrong metric here

Gold answers tend to be short numeric/textual values that do not appear verbatim in the model's recall output. Substring match is essentially zero across all three modes:

| Mode | Substring rate |
|---|---|
| Base | 0.0\% |
| Adapter | 0.5\% |
| Format Guard | 0.5\% |

Source: `results/bfsi_recall_eval/summary.json`. F1 (a soft-overlap metric) is the right tool for this task.

## Format Guard zero-overhead replicates again

FG vs adapter direct: Wilcoxon $p = 0.55$, virtually identical F1 means (0.219 vs 0.219). The routing layer's empirically-zero overhead from `04_format_guard.md` (HumanEval $n=164$) and `11_bfsi_extract_eval.md` ($n=664$) replicates a third time at $n=214$ on a different task type.

## Files
- `synapta_src/data_pipeline/{10,12,13}_*.py`
- `adapters/nemotron_30b/bfsi_recall/best/`
- `results/bfsi_recall_eval/{eval_results.jsonl, summary.json}`
- `data/rbi_corpus/qa/bfsi_recall/`
