# Experiment Card 20 — BFSI Recall Adapter ($n=214$ paired Wilcoxon)

PKB wrapper around `RESEARCH_HISTORY/12_bfsi_recall_eval.md`.

## 1. Research question
Does the same recipe (base, hyperparameters, eval protocol) generalise to a structurally different task: **no-context recall** (questions without a context paragraph)?

## 2. Dataset
- $n=214$ paired no-context recall questions derived from the same train documents.

## 3. Model
- Same base Nemotron-30B 4-bit. LoRA same hyperparameters but attn-only target modules. MAX_LEN=384.

## 4. Evaluation
- Substring match is uninformative (gold answers like "Rs.\ 21" rarely appear verbatim in verbose recall outputs).
- Token F1 is primary. Wilcoxon signed-rank for paired F1.

## 5. Results

| Mode | F1 mean | Lift | Wilcoxon vs base |
|---|---:|---|---|
| Base | 0.158 | --- | --- |
| + bfsi_recall | 0.219 | $+0.061$ ($+38.4\%$ rel.) | $p = 1.50 \times 10^{-16}$ |
| Format Guard (4 adapters) | 0.219 | $+0.061$ | vs adapter direct: $p = 0.55$ |

74.3\% of paired questions show adapter F1 > base F1.

Substring rate: base 0\%, adapter 0.5\%, FG 0.5\% (substring is the wrong metric here, as expected).

## 6. Negatives + caveats
- Substring scoring is essentially uninformative on the recall task; F1 is the right tool.
- The summary.json `mcnemar_f1` field is empty; F1-paired Wilcoxon is computed by re-reading `eval_results.jsonl` (script: standalone scipy call). Numbers above are recomputed from raw rows.

## 7. Artifact map
PRIMARY:
- `synapta_src/data_pipeline/{10_build_recall_dataset, 12_train_bfsi_recall, 13_eval_bfsi_recall}.py`
- `adapters/nemotron_30b/bfsi_recall/best/`
- `results/bfsi_recall_eval/{eval_results.jsonl, summary.json}`
- `data/rbi_corpus/qa/bfsi_recall/`

SECONDARY: `RESEARCH_HISTORY/12_bfsi_recall_eval.md`
