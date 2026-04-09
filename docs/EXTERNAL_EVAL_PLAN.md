# External MD Evaluation Plan

Updated: 2026-04-08

## Goal

Move the project away from closed-loop, similarity-driven claims and toward a defensible external evaluation protocol that is acceptable for both:

- startup positioning
- research reporting

The guiding constraint is: optimize for honesty first, then speed, then scale.

## What We Learned From The Pilot

The 4-item externally authored pilot changed the story materially:

- the old internal ranking did not transfer cleanly
- `sequential_reverse` underperformed under blind rubric judging
- `weighted_merge` was the strongest Qwen method on the pilot
- Mistral was much slower, but not obviously worse on correctness

This implies:

1. We should not use semantic similarity as the headline metric.
2. We should not claim that sequential routing is generally superior.
3. We should treat the internal Qwen-authored benchmark as an ablation/dev set only.

## Why This Protocol, Not Something Else

### Why not semantic similarity?

Because it is too easy for cosine similarity to reward topical overlap while missing factual or procedural errors.

The pilot already showed ranking instability:

- internal dev set favored `sequential_reverse`
- external rubric judging did not

That is enough evidence to demote semantic similarity from primary metric to supporting metric.

### Why blind rubric judging?

Because the benchmark items can carry:

- `required_facts`
- `critical_errors`
- `grading_rubric`

This gives a stronger evaluator than free-form “which answer sounds better?” judging.

### Why not use all 20 domains equally in the next large run?

Because some domain styles produce noisy, hard-to-grade tasks:

- archaic register imitation
- pure stylistic translation
- open-ended humanities interpretation

These are not bad domains, but they increase evaluation variance and answer truncation.

For the next scale-up, we should favor domains whose outputs are:

- compact
- structurally checkable
- less sensitive to stylistic preference

This is not cherry-picking. It is reducing measurement noise.

## Recommended Domain Set For The Next Large Run

Prefer:

- `LEGAL_ANALYSIS`
- `MEDICAL_DIAGNOSIS`
- `PYTHON_LOGIC`
- `MATHEMATICS`
- `MLX_KERNELS`
- `QUANTUM_CHEMISTRY`
- `ORGANIC_SYNTHESIS`
- `ASTROPHYSICS`
- `MARITIME_LAW`
- `CRYPTOGRAPHY`
- `ROBOTICS`
- `CLIMATE_SCIENCE`
- `BEHAVIORAL_ECONOMICS`

Defer for the first large judged run:

- `ARCHAIC_ENGLISH`
- `SANSKRIT_LINGUISTICS`
- `ANCIENT_HISTORY`
- `RENAISSANCE_ART`
- `MUSIC_THEORY`
- `PHILOSOPHY`

Those deferred domains can be reintroduced later as a separate “stretch” split.

## Recommended Dataset Shape

### Phase 1: Fast external benchmark

- 24 items
- 2-domain only
- compact prompts
- short, judgeable reference answers
- strong grading metadata

### Phase 2: Stronger benchmark

- 40 items
- same schema
- balanced pair coverage over the preferred domain set
- human spot-audit on at least 10 items

### Phase 3: Stress split

- 12-20 items
- reintroduce stylistic/humanities domains
- report separately

## Systems To Compare

For speed and interpretability, do not run all 9 methods first.

Run these first:

- `weighted_merge`
- `late_layer_injection`
- `mistral`

Optional fourth method after the first pass:

- `sequential_reverse`

Rationale:

- `weighted_merge` was strongest on the external pilot
- `late_layer_injection` is architecturally plausible and relatively cheap
- `mistral` is the baseline users care about
- `sequential_reverse` should now be treated as a challenged hypothesis, not the default winner

## Metrics To Report

Primary:

- blind pairwise win rate
- average rubric score
- critical-error count
- required-fact coverage

Secondary:

- latency
- semantic similarity
- token F1

Semantic similarity stays in the appendix or as a secondary table only.

## Claims That Are Safe

If the larger run matches the pilot:

- “A small routed Qwen system can be competitive with Mistral on a rubric-judged external multi-domain benchmark while running substantially faster.”
- “Method ranking changes under external evaluation, so internal semantic-similarity ablations should not be overinterpreted.”
- “Simple weighted composition may outperform more elaborate temporal scheduling on externally authored tasks.”

## Claims That Are Not Safe

- “We beat Mistral overall.”
- “Sequential routing is the best method.”
- “Semantic similarity proves accuracy.”
- “The external benchmark is human gold-standard,” unless the references are actually audited by humans.

## Immediate Next Run

When network approval is available again:

1. Generate `24` items with the filtered domain set and `batch-size=1`.
2. Run `weighted_merge`, `late_layer_injection`, and `mistral`.
3. Blind-judge all 24 items.
4. Add `sequential_reverse` only if the first three runs complete cleanly.
5. Publish startup-safe and research-safe claims separately.
