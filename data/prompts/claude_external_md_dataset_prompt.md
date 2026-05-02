You are generating a new external benchmark for a multi-domain adapter-routing system.

Think carefully and internally about benchmark quality, leakage risk, answerability, and evaluation reliability.
Do not output chain-of-thought.
Output only valid JSON.

## Objective

Create benchmark items that require synthesis across exactly two domains. The benchmark must be useful for comparing:

- a small Qwen-based routed adapter system
- a Mistral baseline
- different adapter attachment strategies such as merge, layer-gating, and sequential scheduling

The benchmark must avoid copying or lightly paraphrasing the existing internal MD dataset. Prefer new question constructions, new pairings, new framing, and new reference wording.

## Domain List

{{DOMAIN_LIST}}

## Requested Domain Pairs

{{DOMAIN_PAIRS}}

## Number Of Items

Generate exactly {{NUM_ITEMS}} items.

## Hard Requirements

For each item:

- require exactly two domains from the provided list
- use domain pairings from the requested pairs only
- produce a question that cannot be answered well by only one domain
- make the reference answer concise but information-dense
- include a structured rubric with objective checks whenever possible
- avoid mention of "benchmark", "adapter", "LoRA", "Qwen", "Mistral", or the internal project
- avoid synthetic nonsense facts or fake citations
- avoid near-duplicate phrasing across items

## Answer-Type Mix

Target this mix across the full batch:

- explanatory synthesis
- quantitative or mathematical derivation
- code or algorithm design
- structured translation or formatting

## JSON Schema

Return a JSON array. Each item must have this structure:

[
  {
    "id": "ext_md_001",
    "domains": ["DOMAIN_A", "DOMAIN_B"],
    "required_adapters": ["DOMAIN_A", "DOMAIN_B"],
    "question": "string",
    "reference_answer": "string",
    "rubric": {
      "answer_type": "explanation | math | code | translation | structured_reasoning",
      "must_include_all": ["string", "string"],
      "must_include_any": [["synonym a", "synonym b"], ["synonym c", "synonym d"]],
      "must_not_include": ["string"],
      "numeric_targets": [
        {
          "label": "string",
          "value": "string",
          "tolerance": 0.0
        }
      ],
      "regex_targets": ["string"],
      "judge_focus": ["correctness", "coverage", "hallucination", "usefulness"]
    },
    "provenance": {
      "question_author": "claude-4.6-sonnet-thinking-via-perplexity",
      "reference_author": "claude-4.6-sonnet-thinking-via-perplexity",
      "created_utc": "{{CREATED_UTC}}"
    }
  }
]

## Quality Rules For Rubrics

- `must_include_all` should contain only critical facts, not stylistic preferences
- `must_include_any` should be used for synonyms or alternate phrasings
- `must_not_include` should capture common but serious mistakes
- `numeric_targets` should be non-empty for quantitative tasks
- `regex_targets` should be used for formulas, code signatures, legal article references, or structured outputs

## Quality Filters

Before finalizing, internally check:

- Would a domain expert agree this really requires both domains?
- Is the reference answer factually plausible and concise?
- Is the rubric strong enough to score accuracy better than whole-answer semantic similarity?
- Does this item avoid obvious overlap with the old internal MD set?

Return only the final JSON array.
