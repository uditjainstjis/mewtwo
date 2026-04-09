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

- Name: Risk And Compliance
- Goal: Evaluate legal, safety, ethical, or operational risk under concrete factual constraints.
- Cognitive demand: deep
- Workflow type: risk_compliance

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
    "id": "ext_md_071",
    "domains": [
      "MEDICAL_DIAGNOSIS",
      "ROBOTICS"
    ],
    "required_adapters": [
      "MEDICAL_DIAGNOSIS",
      "ROBOTICS"
    ],
    "section_name": "Risk And Compliance",
    "workflow_type": "risk_compliance",
    "cognitive_demand": "deep",
    "testing_focus": "depth_reasoning",
    "task_style": "identify_and_explain"
  },
  {
    "id": "ext_md_072",
    "domains": [
      "MLX_KERNELS",
      "CRYPTOGRAPHY"
    ],
    "required_adapters": [
      "MLX_KERNELS",
      "CRYPTOGRAPHY"
    ],
    "section_name": "Risk And Compliance",
    "workflow_type": "risk_compliance",
    "cognitive_demand": "deep",
    "testing_focus": "breadth_synthesis",
    "task_style": "compare_two_options"
  },
  {
    "id": "ext_md_073",
    "domains": [
      "ORGANIC_SYNTHESIS",
      "LEGAL_ANALYSIS"
    ],
    "required_adapters": [
      "ORGANIC_SYNTHESIS",
      "LEGAL_ANALYSIS"
    ],
    "section_name": "Risk And Compliance",
    "workflow_type": "risk_compliance",
    "cognitive_demand": "deep",
    "testing_focus": "error_diagnosis",
    "task_style": "diagnose_and_fix"
  },
  {
    "id": "ext_md_074",
    "domains": [
      "PHILOSOPHY",
      "BEHAVIORAL_ECONOMICS"
    ],
    "required_adapters": [
      "PHILOSOPHY",
      "BEHAVIORAL_ECONOMICS"
    ],
    "section_name": "Risk And Compliance",
    "workflow_type": "risk_compliance",
    "cognitive_demand": "deep",
    "testing_focus": "tradeoff_analysis",
    "task_style": "choose_best_design"
  },
  {
    "id": "ext_md_075",
    "domains": [
      "PYTHON_LOGIC",
      "LATEX_FORMATTING"
    ],
    "required_adapters": [
      "PYTHON_LOGIC",
      "LATEX_FORMATTING"
    ],
    "section_name": "Risk And Compliance",
    "workflow_type": "risk_compliance",
    "cognitive_demand": "deep",
    "testing_focus": "counterfactual_reasoning",
    "task_style": "explain_failure_mode"
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
      "created_utc": "2026-04-08T06:52:42.846963+00:00",
      "section": "Risk And Compliance"
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
