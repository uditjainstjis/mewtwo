You are authoring an external multi-domain benchmark for a routed small-language-model system with domain adapters.

Think carefully and deeply before answering, but do not reveal chain-of-thought. Return only the final JSON array.

Your job is to create benchmark items that are useful for:
- objective comparison between a small routed Qwen system and Mistral-7B
- routing and composition ablations
- startup-facing demonstrations of domain specialization
- research-facing evaluation with explicit caveats and provenance

Hard requirements:
- Output valid JSON only. No markdown. No commentary.
- Output exactly `1` items.
- Each item must preserve the provided `id`, `domains`, and `required_adapters`.
- Each question must require integrated reasoning across both listed domains. It must not be answerable well by only one domain.
- Avoid trivia, pure recall, and shallow "mention both topics" prompts.
- Favor tasks with checkable structure: derivation, algorithm design, legal-technical reasoning, scientific explanation with concrete steps, translation with interpretation constraints, formal comparison, code-plus-theory reasoning.
- Prefer `analysis` and `mixed` items. Use `code` only when the core reasoning can be judged from a short algorithmic answer rather than a long implementation.
- Avoid tasks whose best answer would require a full program, many numbered subparts, or long stylistic imitation.
- Avoid purely stylistic or literary tasks unless the correctness conditions remain concrete and judgeable.
- The reference answer must be concise but information-dense, typically 4-8 sentences unless the item is naturally code-oriented.
- Keep the question compact. Target <= 140 words and avoid numbered subparts unless truly necessary.
- Keep the reference answer compact. Target <= 160 words. Do not include long code blocks; if code is relevant, describe the algorithm and expected checks instead.
- Prefer tasks whose best answer can fit comfortably within 120-180 generated words.
- Avoid full-program synthesis, large derivations, or tasks that require reproducing long quotations.
- Prefer "analyze / compare / diagnose / design at a high but concrete level" over "write complete code" unless the code can be described briefly.
- Prefer domain pairs where both domains are semantically load-bearing, but the final answer can still be graded from 3-5 required facts.
- Avoid tasks that require many independent steps or many separate deliverables.
- Treat each item spec as a different test target. Vary challenge style across depth, breadth, diagnosis, tradeoffs, counterfactuals, edge cases, formalization, translation constraints, prioritization, and mechanism comparison.
- Include atomic grading fields so later evaluation is not forced to rely on semantic similarity.
- Do not write citations, URLs, or bibliography fields in the answer text.
- Do not mention Claude, Perplexity, Qwen, Mistral, adapters, or benchmark construction in the item text.

Dataset quality goals:
- Diverse task formats and difficulty.
- Clear, non-ambiguous questions.
- Realistic expert-style prompts, not synthetic theorem slogans.
- Enough specificity that a judge can identify factual omissions and critical errors.
- Avoid answer leakage by not repeating the question phrasing too closely in the reference answer.

For each item, produce an object with exactly these keys:
- `id`
- `domains`
- `required_adapters`
- `question`
- `reference_answer`
- `answer_type`
- `difficulty`
- `required_facts`
- `critical_errors`
- `grading_rubric`
- `provenance`

Field schema:
- `answer_type`: one of `analysis`, `math`, `code`, `mixed`, `translation`
- `difficulty`: one of `medium`, `hard`, `very_hard`
- `required_facts`: array of exactly 4 short atomic facts that a correct answer should cover
- `critical_errors`: array of exactly 3 short error conditions that should count strongly against correctness
- `grading_rubric`: object with:
  - `must_cover`: array of exactly 3 short strings
  - `nice_to_have`: array of exactly 2 short strings
  - `automatic_checks`: array of 0-4 short strings
- `provenance`: object with:
  - `question_author`
  - `reference_author`
  - `generation_model`
  - `quality_notes`

Use this exact provenance content:
- `question_author`: `claude-sonnet-via-perplexity-proxy`
- `reference_author`: `claude-sonnet-via-perplexity-proxy`
- `generation_model`: `claude-4.6-sonnet-thinking`
- `quality_notes`: short note on why the item genuinely requires both domains

Target dataset mix:
- Mostly `analysis` and `mixed`
- Some `math` and `translation`
- Very little `code`, and only when the expected answer is concise

Strongly prefer these task forms:
- identify-and-explain
- compare-two-mechanisms
- diagnose-a-bug-or-design-choice
- choose-between-two-options with explicit reasoning
- interpret-a-short archaic/technical passage and map it to a modern concept
- explain-why-one-approach-fails-on-an-edge-case
- connect-a-formal-rule-to-an-applied-design-choice

Strongly avoid these task forms:
- write a full implementation
- produce a long multi-part procedure
- dump a long formal proof
- give many examples or edge cases

Here are the required item specs you must fill:
[
  {
    "id": "ext_claude_010",
    "domains": [
      "PHILOSOPHY",
      "QUANTUM_CHEMISTRY"
    ],
    "required_adapters": [
      "PHILOSOPHY",
      "QUANTUM_CHEMISTRY"
    ],
    "testing_focus": "mechanism_comparison",
    "task_style": "summarize_with_constraints",
    "difficulty_bias": "hard"
  }
]

Each item spec may include helper planning fields such as `testing_focus`, `task_style`, or `difficulty_bias`.
Use those helper fields to shape the item, but do not add them as output keys unless they fit inside the required schema above.
