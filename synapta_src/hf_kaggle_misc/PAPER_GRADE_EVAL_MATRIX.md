# Paper-Grade Eval Matrix

## Objective

Upgrade the current benchmark pipeline from a pilot screen into a static-eval package that can support serious claims without weakening benchmark quality.

## Comparator Matrix

### Required stages

- `base`: one run per model, stored as `rank=0`
- `math_sft`
- `science_sft`
- `code_sft`
- `merged_dare`
- `dpo`

### Required ranks

- headline LoRA regime: `1, 2, 8, 128`
- ablation or boundary regime: `1024, 3072`

### Required models

- `qwen_0.8b`
- `nemotron_4b`
- `nemotron_30b` when the matched adapters are available

## Static benchmark modes

### `autonomous`

Purpose:

- overnight smoke test
- answer capture
- debugging and fast ranking

Samples:

- `gsm8k`: 40
- `math500`: 40
- `arc`: 40
- `mmlu`: 40
- `mbpp`: 20
- `humaneval`: 20

Claims supported:

- none beyond pilot observations

### `quick`

Purpose:

- rapid iteration after code or prompt changes

Samples:

- `gsm8k`: 100
- `math500`: 100
- `arc`: 100
- `mmlu`: 100
- `mbpp`: 50
- `humaneval`: 50

Claims supported:

- weak directional comparisons only

### `paper`

Purpose:

- minimum serious static benchmark pass

Samples:

- `gsm8k`: 250
- `math500`: 250
- `arc`: 500
- `mmlu`: 500
- `mbpp`: 200
- `humaneval`: 164

Claims supported:

- rank-by-stage comparisons on the Tier 1 benchmark suite
- DPO delta tables
- cross-stage regression analysis
- uncertainty-aware comparisons via confidence intervals

### `research`

Purpose:

- no-compromise static pass for the strongest internal paper version

Samples:

- `gsm8k`: 500
- `math500`: 500
- `arc`: 1000
- `mmlu`: 1000
- `mbpp`: 500
- `humaneval`: 164

Claims supported:

- strongest static claims available from the Tier 1 benchmark set
- more stable comparisons for low-effect-size rank differences

## Tier 2 benchmarks to add next

Supported Tier 2 additions in the evaluator:

- `TruthfulQA-MC`
- `MMLU-Pro`
- `GPQA`

Still required next for the strongest external paper story:

- `IFEval`
- `TruthfulQA` generation or upgraded official TruthfulQA protocol
- `LiveCodeBench`

Reason:

- Tier 1 is a strong internal profile, but not enough alone for a world-class paper that wants broad generalization claims.
- `IFEval` and `LiveCodeBench` should be added only with their official or benchmark-native evaluators, not with improvised heuristics.

## Statistical policy

For static deterministic benchmarks:

- report point estimate
- report `95%` Wilson confidence interval
- do not pretend repeated-seed variance is meaningful when generation is deterministic

For agentic benchmarks:

- repeated runs are mandatory
- report `pass@k`
- report trajectory variance
- report failure and recovery rates

## Claim discipline

### Claims the current static suite can support

- whether DPO improves or hurts specific benchmark families
- whether low-rank vs high-rank behavior differs materially
- whether `1024/3072` act like unstable or weak-return boundary conditions
- whether cross-domain SFT adapters transfer beyond their source domain

### Claims the static suite cannot support by itself

- strong agentic reliability conclusions
- scaffold sensitivity conclusions
- tool-use policy conclusions
- geometry-to-failure conclusions

Those require:

- `PlanBench`
- `tau-bench`
- `ToolSandbox` or `BFCL`
- repeated-run reliability overlays

## Practical reporting guidance

- headline comparisons should prioritize `1, 2, 8, 128`
- treat `1024, 3072` as ablations or near-full-rank controls, especially on `qwen_0.8b`
- use `base -> SFT -> merged_dare -> dpo` deltas, not only raw final scores
- keep all generation JSONL files for qualitative appendix and error analysis
