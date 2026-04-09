You are generating one section of an external benchmark for a multi-domain adapter-routing system.

Think carefully and internally about benchmark quality, leakage risk, answerability, industry usefulness, and evaluation reliability.
Do not output chain-of-thought.
Output only valid JSON.

## Objective

Create benchmark items that require synthesis across exactly two domains. This section is part of a larger 100-item benchmark intended for:

- research-grade external evaluation
- startup-grade product positioning
- comparison between a small routed Qwen system and a Mistral baseline

The benchmark must avoid copying or lightly paraphrasing the internal MD dataset.

## Section

- Name: Edge Cases
- Goal: Stress-test a method or rule on a failure mode, boundary case, or hidden assumption.
- Cognitive demand: deep
- Workflow type: edge_case_analysis

## Domain List

- ANCIENT_HISTORY
- ARCHAIC_ENGLISH
- ASTROPHYSICS
- BEHAVIORAL_ECONOMICS
- CLIMATE_SCIENCE
- CRYPTOGRAPHY
- LATEX_FORMATTING
- LEGAL_ANALYSIS
- MARITIME_LAW
- MATHEMATICS
- MEDICAL_DIAGNOSIS
- MLX_KERNELS
- MUSIC_THEORY
- ORGANIC_SYNTHESIS
- PHILOSOPHY
- PYTHON_LOGIC
- QUANTUM_CHEMISTRY
- RENAISSANCE_ART
- ROBOTICS
- SANSKRIT_LINGUISTICS

## Item Specs

Generate exactly the following items and preserve each item's `id`, `domains`, and `required_adapters`.

[
  {
    "id": "ext_md_086",
    "domains": [
      "PYTHON_LOGIC",
      "ROBOTICS"
    ],
    "required_adapters": [
      "PYTHON_LOGIC",
      "ROBOTICS"
    ],
    "section_name": "Edge Cases",
    "workflow_type": "edge_case_analysis",
    "cognitive_demand": "deep",
    "testing_focus": "edge_case_handling",
    "task_style": "map_formal_rule_to_application"
  },
  {
    "id": "ext_md_087",
    "domains": [
      "QUANTUM_CHEMISTRY",
      "CLIMATE_SCIENCE"
    ],
    "required_adapters": [
      "QUANTUM_CHEMISTRY",
      "CLIMATE_SCIENCE"
    ],
    "section_name": "Edge Cases",
    "workflow_type": "edge_case_analysis",
    "cognitive_demand": "deep",
    "testing_focus": "formalization",
    "task_style": "justify_decision"
  },
  {
    "id": "ext_md_088",
    "domains": [
      "RENAISSANCE_ART",
      "MUSIC_THEORY"
    ],
    "required_adapters": [
      "RENAISSANCE_ART",
      "MUSIC_THEORY"
    ],
    "section_name": "Edge Cases",
    "workflow_type": "edge_case_analysis",
    "cognitive_demand": "deep",
    "testing_focus": "translation_constraints",
    "task_style": "analyze_buggy_reasoning"
  },
  {
    "id": "ext_md_089",
    "domains": [
      "SANSKRIT_LINGUISTICS",
      "ANCIENT_HISTORY"
    ],
    "required_adapters": [
      "SANSKRIT_LINGUISTICS",
      "ANCIENT_HISTORY"
    ],
    "section_name": "Edge Cases",
    "workflow_type": "edge_case_analysis",
    "cognitive_demand": "deep",
    "testing_focus": "stakeholder_usefulness",
    "task_style": "summarize_with_constraints"
  },
  {
    "id": "ext_md_090",
    "domains": [
      "QUANTUM_CHEMISTRY",
      "ORGANIC_SYNTHESIS"
    ],
    "required_adapters": [
      "QUANTUM_CHEMISTRY",
      "ORGANIC_SYNTHESIS"
    ],
    "section_name": "Edge Cases",
    "workflow_type": "edge_case_analysis",
    "cognitive_demand": "deep",
    "testing_focus": "mechanism_comparison",
    "task_style": "prioritize_under_constraints"
  }
]

## Hard Requirements

For each item:

- require exactly two domains from the provided list
- use the item's given domain pair exactly
- make both domains genuinely load-bearing
- optimize for realistic industry-style tasks rather than academic ornament
- keep the question compact, ideally under 120 words
- keep the reference answer compact, ideally under 160 words
- avoid long code blocks, long proofs, and multi-part deliverables
- avoid mention of "benchmark", "adapter", "LoRA", "Qwen", "Mistral", or the internal project
- avoid synthetic nonsense facts or fake citations
- avoid near-duplicate phrasing across items

## Preferred Task Types

Prefer:

- diagnosis/debugging
- decision support
- design review
- implementation planning
- risk/compliance review
- explanation for a stakeholder with concrete constraints
- edge-case analysis
- compare-and-choose reasoning

Avoid:

- full program generation
- long theorem proofs
- purely stylistic imitation
- vague open-ended essays

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
      "created_utc": "2026-04-08T06:55:09.475489+00:00",
      "section": "Edge Cases"
    }
  }
]

## Rubric Rules

- `must_include_all` should contain only critical facts
- `must_include_any` should be used for alternate acceptable phrasing
- `must_not_include` should capture serious mistakes
- `numeric_targets` should be used when quantitative checks are central
- `regex_targets` should be used for formulas, signatures, legal article references, or structured outputs
- keep rubrics compact and judgeable

## Quality Filters

Before finalizing, internally check:

- Would an industry user actually ask something like this?
- Is this measuring useful competence rather than verbosity?
- Is the answer short enough to avoid trivial truncation?
- Is the rubric strong enough to score accuracy better than semantic similarity?
- Does the item fit the section goal and the per-item testing focus?

Return only the final JSON array.
