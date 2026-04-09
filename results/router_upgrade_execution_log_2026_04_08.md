# Router Upgrade Execution Log (April 8, 2026)

## Goal

Move from weight-space routing ablations to a trained collaborative router:

1. Generate a large synthetic routing dataset with frontier-model supervision.
2. Train a dedicated router LoRA with SFT.
3. Apply DPO against observed router failure modes.
4. Plug the trained router back into TCAR and evaluate routing accuracy before full answer-quality reruns.

## Why This Phase Exists

The TCAR pilot showed that collaborative execution has a better ceiling than direct adapter blending, but the real router was too weak to unlock it.

Pilot results on the 10-item stratified external slice:

| Method | Semantic Sim | Token F1 | Latency (s) |
| --- | ---: | ---: | ---: |
| `tcar_collaborative` | `0.6797` | `0.2682` | `18.86` |
| `tcar_oracle_collaborative` | `0.6939` | `0.2921` | `23.11` |
| `mistral` | `0.7067` | `0.2971` | `10.72` |

Router diagnosis from the same pilot:

- exact expert match: `1/10`
- partial overlap with gold experts: `4/10`

So the bottleneck moved from collaborative architecture quality to routing accuracy.

## Implemented In This Phase

New or updated code paths:

- `src/router/generate_synthetic_routing_dataset.py`
- `src/router/prepare_router_sft_dataset.py`
- `src/router/build_router_dpo_dataset.py`
- `src/router/train_router_sft.py`
- `src/router/train_router_dpo.py`
- `src/router/run_router_upgrade_pipeline.py`
- `src/router/eval_router_accuracy.py`
- `backend/collaborative_reasoning.py`
- `src/eval/run_md_head_to_head.py`
- `backend/main.py`

## Smoke Validation Completed

Synthetic routing dataset generation succeeded at multiple scales:

- `2`
- `4`
- `8`
- `10`
- `12`
- `20`
- `40`
- `100`
- `200`

Training stack smoke results:

| Stage | Artifact | Key Result |
| --- | --- | --- |
| SFT | `router_adapters/router_reasoning_sft_smoke200` | `mean_token_accuracy = 0.418` |
| DPO | `router_adapters/router_reasoning_dpo_smoke200` | `rewards/accuracies = 0.80` |

Standalone router-eval smoke validation:

| Adapter | Eval Slice | Exact Match | Partial Overlap | Mean Overlap F1 | Mean Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| `router_reasoning_sft_smoke200` | external `n=5` | `0.00` | `1.00` | `0.6667` | `6.49s` |
| `router_reasoning_dpo_smoke200` | external `n=5` | `0.00` | `1.00` | `0.6667` | `6.45s` |

Important correction:

- local `models/Qwen3.5-0.8B` was not a drop-in text-only causal LM path for this trainer
- the working default base for router training is `Qwen/Qwen2.5-0.5B-Instruct`

## Current Long Run

Active frontier-generation target:

- output: `data/router_synthetic_routing_5000.json`
- raw batches: `results/router_generation_5000_raw`
- source model: `claude-4.6-sonnet` via local Perplexity proxy
- stable batch setting: `100`

Observed behavior on the current run:

- `batch-size 100` is stable
- `batch-size 500` was too fragile
- early observed throughput is about `100` items per `~2 minutes`

Progress snapshot while writing this log:

- merged rows: `600 / 5000`

Later progress updates during the same run:

- merged rows reached `3200 / 5000`
- raw batches through `batch_0031.json` were saved cleanly
- no generator corruption or batch-loss was observed in the stable `batch-size 100` regime
- the 5k routing corpus later completed successfully at `5000 / 5000`

## Integration Completed Before Training Finished

The trained router is no longer isolated from the benchmark path.

Integration work already done:

- TCAR now supports an optional HF+LoRA router during the routing phase
- benchmark runner now accepts:
  - `--tcar-router-model`
  - `--tcar-router-adapter`
- API path can load the same router with:
  - `TCAR_ROUTER_MODEL`
  - `TCAR_ROUTER_ADAPTER`

This means the next eval loop is direct:

1. finish large routing generation
2. run SFT
3. run DPO
4. score router exact-match / overlap
5. rerun TCAR with the trained router in place

## Honest Caveats

- full 5k generation was still in progress when this log entry was written
- full SFT and DPO over the 5k corpus had not finished yet
- KV-cache sharing across expert branches is still not integrated
- collaborative answer quality has improved ceiling evidence, not a proven production win yet
- the router-eval path initially failed twice for infrastructure reasons:
  - importing the full MLX collaborative stack into the standalone HF router evaluator
  - `transformers` touching Hugging Face metadata endpoints despite cached weights
- both infra failures were fixed in this phase before the full 5k pipeline reached them

## Training Runtime Correction

The original `trl` / `Trainer` path did not survive the full 5k run.

What happened:

- data prep succeeded
- model load succeeded
- tokenization succeeded
- the trainer crashed on the first optimizer step with an Apple MLX / Metal runtime abort

Important interpretation:

- this was not a bad-data failure
- this was not a router-model logic failure
- it was a runtime-stack instability in the training path

Recovery action taken:

- replaced the SFT and DPO trainer path with manual PyTorch training loops
- validated both manual loops with 1-step probes on the full 5k-derived datasets
- relaunched the full router pipeline on the manual stack with `--force-cpu`

Manual full-run status when this log was last updated:

- SFT is live on the 5k corpus
- observed training progress reached:
  - `step 10 / 1226`
  - `step 20 / 1226`
- observed elapsed time at `step 20`: about `70s`
