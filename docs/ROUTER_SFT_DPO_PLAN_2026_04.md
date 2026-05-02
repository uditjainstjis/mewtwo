# Router SFT + DPO Plan (April 2026)

## Why We Are Doing This

The first TCAR pilot changed the bottleneck.

The collaborative branch-and-refine architecture showed a better quality ceiling than weight blending, but the real-router version failed to realize that gain because routing quality was too poor.

From the TCAR pilot:

- `tcar_collaborative`: `Sim 0.6797`, `F1 0.2682`, `Lat 18.86s`
- `tcar_oracle_collaborative`: `Sim 0.6939`, `F1 0.2921`, `Lat 23.11s`
- `mistral`: `Sim 0.7067`, `F1 0.2971`, `Lat 10.72s`

Most important diagnosis:

- real TCAR router exact expert match on the 10-item pilot: `1/10`
- partial overlap: `4/10`

So the current problem is no longer “can collaborative reasoning help?”

It can.

The current problem is:

**the router is too inaccurate to unlock that ceiling.**

## New Hypothesis

Instead of asking the base model to improvise routing from scratch at inference time, train a dedicated router LoRA to imitate stronger routing/planning behavior.

Then sharpen it with preference optimization so it learns to avoid:

- one-expert collapse
- hallucinated unrelated experts
- generic fallback to familiar domains like `LEGAL_ANALYSIS`

## New Pipeline

### Step 1: Synthetic Routing Supervision

Generate a large routing corpus with:

- realistic user questions
- short `<thinking>` style rationale
- exact gold expert tags

This is implemented in:

- `src/router/generate_synthetic_routing_dataset.py`

Default target:

- `5000` items
- `70%` two-expert prompts
- `30%` single-expert prompts
- balanced coverage over the 20 domain experts
- structured workflow and difficulty fields

### Step 2: Router SFT

Train a dedicated router LoRA to map:

- user question

to:

- short planning rationale
- exact expert tags

This is implemented in:

- `src/router/prepare_router_sft_dataset.py`
- `src/router/train_router_sft.py`
- `src/router/run_router_upgrade_pipeline.py`

Current training stack:

- default base model: `Qwen/Qwen2.5-0.5B-Instruct`
- `transformers` + `peft` + `trl`
- LoRA over attention projection modules
- validated on local MPS training with smoke SFT and DPO runs

### Step 3: Router DPO

Build preference pairs where:

- `chosen` = gold routing trajectory
- `rejected` = collapsed/hallucinated routing pattern

Empirical negative patterns are seeded from:

- `results/tcar_collaborative_pilot_10.jsonl`

This is implemented in:

- `src/router/build_router_dpo_dataset.py`
- `src/router/train_router_dpo.py`
- `src/router/eval_router_accuracy.py`

This does not just teach the model what is correct.

It teaches the router what *failure modes to avoid*.

## What The Scripts Do

### `generate_synthetic_routing_dataset.py`

- builds a coverage schedule over experts, arity, workflow type, and difficulty
- uses the Perplexity frontier proxy to generate high-quality routing examples
- writes raw per-batch outputs and a merged dataset file incrementally
- recursively splits failed batches for robustness

### `prepare_router_sft_dataset.py`

- converts the synthetic routing JSON into chat-format JSONL
- writes:
  - `train.jsonl`
  - `valid.jsonl`
  - `metadata.json`

### `build_router_dpo_dataset.py`

- converts the synthetic set into DPO `prompt/chosen/rejected` triples
- injects real router failure patterns from the TCAR pilot
- adds synthetic negatives for broader coverage

### `train_router_sft.py`

- runs supervised fine-tuning for the router LoRA
- saves the adapter and tokenizer in `adapters/routers/router_reasoning_sft`

### `train_router_dpo.py`

- loads the SFT adapter
- runs preference optimization on the DPO set
- saves the improved adapter in `adapters/routers/router_reasoning_dpo`

### `run_router_upgrade_pipeline.py`

- waits for the full routing dataset
- prepares the SFT and DPO derivatives
- runs SFT followed by DPO in one controlled path

### `eval_router_accuracy.py`

- loads the trained router adapter
- scores expert-tag exact match and overlap on any routing-style dataset
- gives a clean diagnostic before rerunning full TCAR generation

## What We Already Validated

The full router-improvement stack is no longer speculative.

Smoke artifacts completed successfully:

- synthetic routing generation:
  - `data/router_synthetic_routing_smoke_2.json`
  - `data/router_synthetic_routing_smoke_4.json`
  - `data/router_synthetic_routing_smoke_8.json`
  - `data/router_synthetic_routing_smoke_10.json`
  - `data/router_synthetic_routing_smoke_12.json`
  - `data/router_synthetic_routing_smoke_20b.json`
  - `data/router_synthetic_routing_smoke_40.json`
  - `data/router_synthetic_routing_smoke_100.json`
  - `data/router_synthetic_routing_smoke_200.json`
- SFT smoke training:
  - `adapters/routers/router_reasoning_sft_smoke200`
- DPO smoke training:
  - `adapters/routers/router_reasoning_dpo_smoke200`

Smoke training outcomes:

- SFT smoke:
  - `train_runtime`: `6.29s`
  - `train_loss`: `3.971`
  - `mean_token_accuracy`: `0.418`
- DPO smoke:
  - `train_runtime`: `39.81s`
  - `train_loss`: `0.650`
  - `rewards/accuracies`: `0.80`
  - `rewards/margins`: `0.0915`

Practical generation finding:

- `batch-size 100` is the stable full-run setting
- `batch-size 500` proved too fragile for long frontier generations

Current long-running artifact:

- full generation target: `data/router_synthetic_routing_5000.json`
- raw batches: `results/router_generation_5000_raw`
- launched on April 8, 2026 with `claude-4.6-sonnet` through the Perplexity proxy
- stable observed throughput on the current run: about `100` items per `~2 minutes`

## What We Expect To Learn

There are three possible outcomes:

1. Router exact-match rises sharply and real TCAR approaches oracle TCAR.
   This is the success case.

2. Router improves, but real TCAR still does not move much.
   Then the refiner or expert-branch prompting is still the real bottleneck.

3. Router remains weak even after SFT/DPO.
   Then the small base model is not expressive enough for this routing formulation.

## Important Constraint: KV-Cache Sharing

The current TCAR implementation validates:

- collaborative routing
- independent expert branches
- refining aggregation

But it does **not yet** implement shared prompt KV-cache across all expert branches.

We inspected the local `mlx_lm` package and confirmed prompt-cache support exists.
The missing work is integration into our branch execution path in a way that preserves:

- expert-specific adapter activation
- branch isolation
- low branch-start overhead

## Router Integration Status

The trained router is now wired into the collaborative path.

Implementation status:

- `backend/collaborative_reasoning.py` supports an optional HF+LoRA router for the routing phase
- `src/eval/run_md_head_to_head.py` can pass:
  - `--tcar-router-model`
  - `--tcar-router-adapter`
- `backend/main.py` can load the same router through:
  - `TCAR_ROUTER_MODEL`
  - `TCAR_ROUTER_ADAPTER`

This means the end state is no longer:

- “train a router checkpoint and inspect it separately”

It is:

- “train a router checkpoint and plug it directly into the TCAR benchmark path”

That is the next latency optimization target after the router is improved.

## Repo Artifacts Added In This Phase

- `src/router/generate_synthetic_routing_dataset.py`
- `src/router/prepare_router_sft_dataset.py`
- `src/router/build_router_dpo_dataset.py`
- `src/router/train_router_sft.py`
- `src/router/train_router_dpo.py`

## Current Position

This is the right move scientifically.

The TCAR pilot already showed:

- collaborative reasoning has a better ceiling than blended routing
- oracle experts matter a lot
- router quality is now the dominant bottleneck

So the new experimental question is not whether collaborative reasoning is valid.

It is:

**Can a dedicated router LoRA, trained via SFT plus DPO, raise routing quality enough to unlock the oracle-TCAR ceiling in the real pipeline?**
