# TCAR Pilot Report

Date: 2026-04-08

## Goal

Test a new inference-time-scaling architecture instead of weight blending:

1. natural-language reasoning router
2. independent expert branches
3. refining aggregator over branch outputs

This was implemented as `tcar_collaborative` and tested on a 10-item stratified external MD slice.

An oracle variant, `tcar_oracle_collaborative`, was also tested to isolate router error from collaborative execution quality.

## Implementation

Files added or updated:

- `backend/collaborative_reasoning.py`
- `backend/main.py`
- `src/eval/run_md_head_to_head.py`

New execution modes:

- `tcar_collaborative`
- `tcar_oracle_collaborative`
- API mode: `collaborative_reasoning`

## 10-Item Stratified Pilot Summary

Dataset:

- `data/multidomain_eval_claude_external_v2_10_stratified.json`

Comparison artifact:

- `results/tcar_pilot_10_comparison.json`

### Average Metrics

| Method | Semantic Sim | Token F1 | Latency |
| --- | ---: | ---: | ---: |
| weighted_merge | 0.6311 | 0.2603 | 4.0086s |
| late_layer_injection | 0.6804 | 0.2696 | 3.5773s |
| sequential_reverse | 0.6583 | 0.2964 | 4.4147s |
| mistral | 0.7067 | 0.2971 | 10.7177s |
| tcar_collaborative | 0.6797 | 0.2682 | 18.8588s |
| tcar_oracle_collaborative | 0.6939 | 0.2921 | 23.1086s |

## Main Read

### Real-router TCAR

- materially better than `weighted_merge` on similarity: `0.6797` vs `0.6311`
- roughly tied with `late_layer_injection` on similarity: `0.6797` vs `0.6804`
- below `mistral` on both similarity and F1
- much slower than every existing Qwen method

So the real-router TCAR pipeline does **not** yet justify its latency cost.

### Oracle TCAR

- best Qwen-family average on this slice
- improves over real-router TCAR:
  - semantic sim: `+0.0142`
  - token F1: `+0.0239`
- narrows the gap to `mistral`:
  - semantic sim gap: `0.6939` vs `0.7067`
  - token F1 gap: `0.2921` vs `0.2971`

This means the collaborative execution plus refiner idea is directionally promising.

The current bottleneck is primarily routing quality, not only the collaborative execution loop.

## Router Diagnosis

Real TCAR router on the 10-item pilot:

- exact expert-match: `1 / 10`
- partial overlap with gold experts: `4 / 10`

Examples of wrong routing:

- `ext_md_006`
  gold: `MEDICAL_DIAGNOSIS`, `MATHEMATICS`
  predicted: `QUANTUM_CHEMISTRY`, `CLIMATE_SCIENCE`

- `ext_md_016`
  gold: `MATHEMATICS`, `ASTROPHYSICS`
  predicted: `PYTHON_LOGIC`, `MLX_KERNELS`

- `ext_md_031`
  gold: `ANCIENT_HISTORY`, `PHILOSOPHY`
  predicted: `LEGAL_ANALYSIS`, `MEDICAL_DIAGNOSIS`

So the router is currently too weak for this architecture to realize its potential.

## Item-Level Leader Count

By semantic similarity leader on the 10 pilot items:

- `tcar_oracle_collaborative`: `3`
- `mistral`: `2`
- `late_layer_injection`: `2`
- `tcar_collaborative`: `1`
- `sequential_reverse`: `1`
- `weighted_merge`: `1`

This is a useful signal:

- the collaborative architecture can win items
- the oracle version wins more often than the real-router version
- the ceiling is better than the current real-router averages suggest

## Honest Conclusion

The new hypothesis is **partly validated**.

What worked:

- moving from weight blending to branch-and-refine did recover answer quality
- with oracle experts, the collaborative pipeline got close to `mistral` on this pilot slice

What did not work yet:

- the natural-language router is too inaccurate
- the latency cost is too high in the current implementation
- the real-router version is not yet better than the best old Qwen method

## Practical Interpretation

This does **not** prove TCAR is the new winner.

It does show something important:

- weight blending was probably not the right path for multi-step synthesis
- inference-time scaling has a better quality ceiling
- routing quality now dominates the error budget

## Next Recommended Steps

1. Replace or strengthen the TCAR router before judging the architecture.
2. Add a router-eval benchmark specifically for expert selection quality on the 100-item MD set.
3. Cut refiner latency with shorter expert prompts and tighter synthesis prompts.
4. Run blind judging on the oracle TCAR slice before claiming quality gains.
5. If oracle TCAR holds under blind judging, then invest in a much better router.

## Artifacts

- `results/tcar_collaborative_pilot_10.jsonl`
- `results/tcar_oracle_collaborative_pilot_10.jsonl`
- `results/tcar_pilot_10_comparison.json`
- `results/tcar_parallel_smoke.jsonl`
