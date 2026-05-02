# Priority Queue

## P1 — Demo Robustness (YC interview survival)
- [x] Bootstrap directories
- [x] Read reference script (token_router_eval.py)
- [ ] Write 20-prompt test suite (math/code/science/mixed)
- [ ] Write smoke test runner that loads model + 3 adapters + tests 4 modes (base / single-best / token-routing / format-guard)
- [ ] Launch smoke test as background job with heartbeats
- [ ] Inspect results, identify failure modes
- [ ] Write findings/demo_diagnosis.md
- [ ] Patch most critical failure (write to demo_artifacts/)

## P2 — Benchmark Defensibility
- [ ] Write full HumanEval (n=164) runner using known-good cache pattern
- [ ] Run base Nemotron-30B on full HumanEval
- [ ] Run Format Guard variant on full HumanEval
- [ ] Compute pass@1 for both, save to qa_pairs/humaneval_full.jsonl
- [ ] Write findings/humaneval_n164.md

## P3 — Code Paradox Replication
- [ ] Locate Qwen-3.5-0.8B and Nemotron-4B adapters in hf_kaggle_opensource/
- [ ] Run math vs code adapter on MATH-500 sample on each smaller model
- [ ] Test if code-adapter > math-adapter on math holds across families
- [ ] Write findings/code_paradox_replication.md

## P4 — Horizon
(only if P1-P3 clean)
