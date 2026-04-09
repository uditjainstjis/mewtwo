# External MD Dataset And Evaluation Plan

## Goal

Build an externally authored multi-domain benchmark that is usable for:

- startup positioning around controllable domain specialization, latency, and deployment efficiency
- research claims about adapter composition, routing, and interference

The current internal MD set is useful for ablations, but it is not strong enough for headline claims because it is too close to the system that was optimized against it.

## What Changes

We split the work into two tracks:

- dataset generation: create a new externally authored MD benchmark through the local Perplexity proxy using `claude-4.6-sonnet-thinking`
- accuracy evaluation: score answers with structured checks rather than relying on full-answer embedding similarity

## Recommended Dataset Size

Use a staged rollout rather than one giant run.

1. Pilot: 24 items
2. Main benchmark: 72 items
3. Expansion: 120 items if variance is still high or you want stronger investor-grade confidence

Rationale:

- 24 items is enough to validate the prompt format and rubric quality
- 72 items is large enough for a paper ablation and startup deck table without making reruns too slow
- 120 items is better only after the scoring rubric is stable

## Composition Rules

Each item should require exactly two domains from `backend/expert_registry.json`.

Balance the benchmark across answer types:

- 40% explanatory synthesis
- 25% math or quantitative derivation
- 20% code or algorithm design
- 15% translation, formatting, or structured reasoning

Each item should include:

- `id`
- `domains`
- `required_adapters`
- `question`
- `reference_answer`
- `rubric`
- `provenance`

## Rubric Design

Each item rubric should prefer objective checks.

Required fields:

- `answer_type`
- `must_include_all`
- `must_include_any`
- `must_not_include`
- `numeric_targets`
- `regex_targets`
- `judge_focus`

Interpretation:

- `must_include_all`: critical concepts that must appear
- `must_include_any`: synonym groups, where matching any one phrase satisfies that slot
- `must_not_include`: known hallucinations or false claims
- `numeric_targets`: exact or tolerance-based checks for math and quantitative items
- `regex_targets`: patterns for formulas, code shapes, citations, or legal article references

## Evaluation Stack

Primary metrics:

- rubric pass rate
- required coverage rate
- critical error rate
- numeric check pass rate
- regex check pass rate
- pairwise win rate on correctness and usefulness

Secondary metrics:

- latency
- semantic similarity
- token F1

Semantic similarity should remain a secondary descriptive metric only.

## Startup-Safe Claims

Prefer claims like:

- smaller routed system with controllable domain specialization
- lower latency or lower memory footprint at near-competitive quality
- structured adapter scheduling beats naive merging
- transparent, auditable routing and expert composition

Avoid claims like:

- universally beats Mistral
- best model overall
- external validation complete

until the new benchmark is generated and scored.

## Research-Safe Claims

Use the current results to support:

- adapter composition is order-sensitive
- depth-aware and time-aware injection outperform naive merge
- scaling adapter magnitude alone does not solve interference
- prompt-level global routing is likely too coarse for multi-domain generation

Use the new benchmark to test:

- whether those findings survive an externally authored distribution
- whether pairwise sequential scheduling retains its benefit on objective checks
- whether single-adapter oracle wins imply many prompts are effectively one dominant domain

## Immediate Run Order

1. Generate 24 pilot items with Claude through the local proxy
2. Review rubric quality manually on 5 items
3. Expand to 72 items
4. Run Qwen methods and Mistral on the new set
5. Score with rubric-based metrics
6. Use semantic similarity only as a secondary diagnostic
