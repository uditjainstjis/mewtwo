# Experiment Card 12 — Static Adapter Composition Failure (Nemotron-30B)

PKB wrapper around `RESEARCH_HISTORY/03_static_composition_failure.md`.

## 1. Research question
Does weight-merging (DARE/TIES/uniform) of 4 task-specialised adapters on Nemotron-30B yield emergent composition gain over the best single expert?

## 2. Dataset
- 4-benchmark grid (ARC/HumanEval/MATH-500/MBPP), $n=100/100/200/100$.
- Phase-2 mixed-domain probe ($n=45$ author-curated multi-domain queries).

## 3. Model
- Same 4 adapters as `11_phase1_single_adapter_30b.md`. Three merge strategies: DARE, TIES, uniform linear, all at $w_i = 1/N$.

## 4. Evaluation
- Same per-benchmark metrics. $n=45$ probe scored on aggregate.

## 5. Results

| Method | ARC | HumanEval (v1) | MATH-500 | MBPP |
|---|---:|---:|---:|---:|
| Base | 20\% | 50\% | 41.5\% | 8\% |
| Best single | 31\% (code) | 60\% (math) | 56\% (code) | 6\% (code) |
| Static merge | 19\% | 34\% | 56\% | 0\% |

Mixed-domain probe ($n=45$): Base 51.1\%, Best single (routed) 60.0\%, all 3 merges 60.0\%, **$\Delta = +0.0$**.

## 6. Negatives + caveats
- Static merging never exceeds best-single; degrades on out-of-home tasks.
- Tested at $r=16$ uniform weights only. Learned-weight composition (LoRAHub-style) at 30B is not tested.

## 7. Artifact map
PRIMARY: `results/nemotron/master_results.json` (last row), `results/nemotron/sprint_results.json`
SECONDARY: `RESEARCH_HISTORY/03_static_composition_failure.md`
