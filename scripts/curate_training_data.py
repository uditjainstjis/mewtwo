"""
Data Curation Script — Synapta v2.0
Downloads, cleans, and formats training data for all 20 domains.

Tier 1 (6 domains, 2000 samples each): Full dynamics analysis
Tier 2 (7 domains, 800 samples each): Broad composition testing  
Tier 3 (7 domains, 500 samples each): Exotic generalization test

Each dataset is formatted as instruction-following JSONL:
{
    "instruction": "...",
    "input": "...",      (optional)
    "output": "...",
    "domain": "MATHEMATICS",
    "source": "gsm8k",
    "difficulty": 0.73     (IRT-estimated difficulty for evaluation)
}
"""

import os
import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

DOMAINS = {
    # ═════════════════════════════════════════════════════
    # TIER 1: Deep analysis domains (2000 samples each)
    # ═════════════════════════════════════════════════════
    "MATHEMATICS": {
        "tier": 1,
        "target_samples": 2000,
        "sources": [
            {"name": "gsm8k", "hf_path": "openai/gsm8k", "split": "train", "config": "main"},
            {"name": "math", "hf_path": "lighteval/MATH", "split": "train", "config": "all"},
            {"name": "metamathqa", "hf_path": "meta-math/MetaMathQA", "split": "train"},
        ],
        "eval_benchmarks": ["gsm8k_test", "math_test", "mmlu_math"],
    },
    "MEDICAL_DIAGNOSIS": {
        "tier": 1,
        "target_samples": 2000,
        "sources": [
            {"name": "medqa", "hf_path": "bigbio/med_qa", "split": "train", "config": "med_qa_en_source"},
            {"name": "pubmedqa", "hf_path": "qiaojin/PubMedQA", "split": "train", "config": "pqa_labeled"},
            {"name": "medical_meadow", "hf_path": "medalpaca/medical_meadow_medical_flashcards", "split": "train"},
        ],
        "eval_benchmarks": ["medqa_test", "mmlu_clinical", "mmlu_anatomy"],
    },
    "LEGAL_ANALYSIS": {
        "tier": 1,
        "target_samples": 2000,
        "sources": [
            {"name": "legalbench", "hf_path": "nguha/legalbench", "split": "train", "config": "contract_nli_explicit_identification"},
            {"name": "legal_qa", "hf_path": "dzunggg/legal-qa-v1", "split": "train"},
        ],
        "eval_benchmarks": ["legalbench_test", "mmlu_jurisprudence", "mmlu_professional_law"],
    },
    "PYTHON_LOGIC": {
        "tier": 1,
        "target_samples": 2000,
        "sources": [
            {"name": "code_alpaca", "hf_path": "sahil2801/CodeAlpaca-20k", "split": "train"},
            {"name": "magicoder", "hf_path": "ise-uiuc/Magicoder-OSS-Instruct-75K", "split": "train"},
        ],
        "eval_benchmarks": ["humaneval", "mbpp", "mmlu_cs"],
    },
    "PHILOSOPHY": {
        "tier": 1,
        "target_samples": 2000,
        "sources": [
            {"name": "philosophy_qa", "hf_path": "KingNish/reasoning-base-20k", "split": "train"},
        ],
        "eval_benchmarks": ["mmlu_philosophy", "mmlu_moral_scenarios"],
    },
    "ASTROPHYSICS": {
        "tier": 1,
        "target_samples": 2000,
        "sources": [
            {"name": "sciq", "hf_path": "allenai/sciq", "split": "train"},
            {"name": "scienceqa", "hf_path": "derek-thomas/ScienceQA", "split": "train"},
        ],
        "eval_benchmarks": ["mmlu_astronomy", "mmlu_college_physics"],
    },

    # ═════════════════════════════════════════════════════
    # TIER 2: Broad composition domains (800 samples each)
    # ═════════════════════════════════════════════════════
    "CRYPTOGRAPHY": {
        "tier": 2, "target_samples": 800,
        "sources": [{"name": "stem_qa", "hf_path": "allenai/sciq", "split": "train"}],
        "eval_benchmarks": ["mmlu_computer_security"],
    },
    "CLIMATE_SCIENCE": {
        "tier": 2, "target_samples": 800,
        "sources": [{"name": "sciq", "hf_path": "allenai/sciq", "split": "train"}],
        "eval_benchmarks": ["mmlu_global_facts"],
    },
    "BEHAVIORAL_ECONOMICS": {
        "tier": 2, "target_samples": 800,
        "sources": [{"name": "economics_qa", "hf_path": "KingNish/reasoning-base-20k", "split": "train"}],
        "eval_benchmarks": ["mmlu_econometrics"],
    },
    "ORGANIC_SYNTHESIS": {
        "tier": 2, "target_samples": 800,
        "sources": [{"name": "sciq", "hf_path": "allenai/sciq", "split": "train"}],
        "eval_benchmarks": ["mmlu_college_chemistry"],
    },
    "ROBOTICS": {
        "tier": 2, "target_samples": 800,
        "sources": [{"name": "stem_qa", "hf_path": "allenai/sciq", "split": "train"}],
        "eval_benchmarks": ["mmlu_electrical_engineering"],
    },
    "MUSIC_THEORY": {
        "tier": 2, "target_samples": 800,
        "sources": [{"name": "reasoning", "hf_path": "KingNish/reasoning-base-20k", "split": "train"}],
        "eval_benchmarks": ["mmlu_humanities"],
    },
    "ANCIENT_HISTORY": {
        "tier": 2, "target_samples": 800,
        "sources": [{"name": "reasoning", "hf_path": "KingNish/reasoning-base-20k", "split": "train"}],
        "eval_benchmarks": ["mmlu_world_history", "mmlu_prehistory"],
    },

    # ═════════════════════════════════════════════════════
    # TIER 3: Exotic generalization domains (500 samples each)
    # ═════════════════════════════════════════════════════
    "MARITIME_LAW": {
        "tier": 3, "target_samples": 500,
        "sources": [],  # Will use LLM generation
        "eval_benchmarks": ["mmlu_professional_law"],
    },
    "RENAISSANCE_ART": {
        "tier": 3, "target_samples": 500,
        "sources": [],
        "eval_benchmarks": ["mmlu_humanities"],
    },
    "LATEX_FORMATTING": {
        "tier": 3, "target_samples": 500,
        "sources": [],
        "eval_benchmarks": [],
    },
    "QUANTUM_CHEMISTRY": {
        "tier": 3, "target_samples": 500,
        "sources": [{"name": "sciq", "hf_path": "allenai/sciq", "split": "train"}],
        "eval_benchmarks": ["mmlu_college_chemistry", "mmlu_college_physics"],
    },
    "ARCHAIC_ENGLISH": {
        "tier": 3, "target_samples": 500,
        "sources": [],
        "eval_benchmarks": [],
    },
    "SANSKRIT_LINGUISTICS": {
        "tier": 3, "target_samples": 500,
        "sources": [],
        "eval_benchmarks": [],
    },
    "SYSTEMS_PROGRAMMING": {
        "tier": 3, "target_samples": 500,
        "sources": [{"name": "code", "hf_path": "sahil2801/CodeAlpaca-20k", "split": "train"}],
        "eval_benchmarks": ["mmlu_cs"],
    },
}


def format_as_instruction(example, domain, source_name):
    """Convert a raw dataset example into our standard instruction format."""
    instruction = ""
    input_text = ""
    output = ""

    if source_name in ("gsm8k",):
        instruction = example.get("question", "")
        output = example.get("answer", "")
    elif source_name in ("math",):
        instruction = example.get("problem", "")
        output = example.get("solution", "")
    elif source_name in ("metamathqa",):
        instruction = example.get("query", "")
        output = example.get("response", "")
    elif source_name in ("medqa",):
        instruction = example.get("question", "")
        options = example.get("options", {})
        if options:
            input_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
        output = example.get("answer", "")
    elif source_name in ("pubmedqa",):
        instruction = example.get("question", "")
        input_text = example.get("context", {}).get("contexts", [""])[0] if isinstance(example.get("context"), dict) else ""
        output = example.get("long_answer", "")
    elif source_name in ("medical_meadow",):
        instruction = example.get("input", "")
        output = example.get("output", "")
    elif source_name in ("code_alpaca", "magicoder"):
        instruction = example.get("instruction", example.get("problem", ""))
        input_text = example.get("input", "")
        output = example.get("output", example.get("solution", ""))
    elif source_name in ("sciq",):
        instruction = example.get("question", "")
        input_text = example.get("support", "")
        output = example.get("correct_answer", "")
    elif source_name in ("legalbench",):
        instruction = example.get("text", example.get("question", ""))
        output = example.get("answer", example.get("label", ""))
    elif source_name in ("legal_qa",):
        instruction = example.get("question", "")
        output = example.get("answer", "")
    else:
        # Generic fallback
        instruction = example.get("instruction", example.get("question", example.get("input", "")))
        output = example.get("output", example.get("answer", example.get("response", "")))

    if not instruction or not output:
        return None

    return {
        "instruction": str(instruction).strip(),
        "input": str(input_text).strip(),
        "output": str(output).strip(),
        "domain": domain,
        "source": source_name,
    }


def download_and_process_domain(domain_name, domain_config, output_dir):
    """Download, format, and save data for a single domain."""
    target = domain_config["target_samples"]
    tier = domain_config["tier"]
    output_path = Path(output_dir) / domain_name
    output_path.mkdir(parents=True, exist_ok=True)

    all_examples = []

    for source in domain_config.get("sources", []):
        src_name = source["name"]
        hf_path = source["hf_path"]
        split = source.get("split", "train")
        config = source.get("config", None)

        print(f"  📥 Downloading {hf_path} ({src_name})...")
        try:
            if config:
                ds = load_dataset(hf_path, config, split=split)
            else:
                ds = load_dataset(hf_path, split=split)

            for ex in ds:
                formatted = format_as_instruction(ex, domain_name, src_name)
                if formatted:
                    all_examples.append(formatted)

            print(f"    ✅ Got {len(all_examples)} examples so far")
        except Exception as e:
            print(f"    ⚠️ Failed to load {hf_path}: {e}")
            continue

    if not all_examples:
        print(f"  ⚠️ No data found for {domain_name}. Will need LLM generation.")
        # Write a placeholder
        with open(output_path / "NEEDS_GENERATION.txt", "w") as f:
            f.write(f"Domain {domain_name} needs LLM-generated training data.\n")
            f.write(f"Target: {target} samples\n")
        return 0

    # Deduplicate by instruction text
    seen = set()
    unique = []
    for ex in all_examples:
        key = ex["instruction"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    all_examples = unique

    # Shuffle and cap at target
    random.shuffle(all_examples)
    if len(all_examples) > target:
        all_examples = all_examples[:target]

    # Split: 90% train, 10% val
    split_idx = int(0.9 * len(all_examples))
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    # Save as JSONL
    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
        with open(output_path / name, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  ✅ {domain_name} (Tier {tier}): {len(train_data)} train, {len(val_data)} val")
    return len(all_examples)


def create_multidomain_eval(output_dir, n_questions=200):
    """
    Create multi-domain evaluation questions that require composing
    knowledge from 2-3 domains simultaneously.
    """
    domain_pairs = [
        # Cross-domain pairs that test composition
        (["MATHEMATICS", "MEDICAL_DIAGNOSIS"], "Calculate the dosage adjustment for a patient with renal impairment given a creatinine clearance of 35 mL/min for a drug with first-order elimination kinetics."),
        (["LEGAL_ANALYSIS", "CRYPTOGRAPHY"], "Analyze the legal implications of using end-to-end encryption in financial transactions under EU's MiCA regulation."),
        (["PYTHON_LOGIC", "ASTROPHYSICS"], "Write a Python function that calculates the Schwarzschild radius for a given stellar mass and determines if the star would form a black hole."),
        (["PHILOSOPHY", "MEDICAL_DIAGNOSIS"], "Apply Kantian ethics to evaluate the moral permissibility of using AI-driven diagnostic systems that may have demographic bias."),
        (["MATHEMATICS", "CLIMATE_SCIENCE"], "Using differential equations, model the relationship between CO2 concentration and radiative forcing described by the Arrhenius equation."),
        (["LEGAL_ANALYSIS", "MEDICAL_DIAGNOSIS"], "What are the HIPAA implications of using federated learning for training medical AI models across hospital networks?"),
        (["PYTHON_LOGIC", "CRYPTOGRAPHY"], "Implement a Python class for Shamir's Secret Sharing scheme with threshold reconstruction and explain its information-theoretic security."),
        (["ASTROPHYSICS", "PHILOSOPHY"], "Compare the philosophical implications of the Many-Worlds interpretation of quantum mechanics with Leibniz's principle of sufficient reason."),
        (["MATHEMATICS", "BEHAVIORAL_ECONOMICS"], "Derive the Nash equilibrium for a public goods game with n players and diminishing marginal returns, then explain the free-rider problem."),
        (["MEDICAL_DIAGNOSIS", "ORGANIC_SYNTHESIS"], "Explain the mechanism of action of aspirin at the molecular level, including the acetylation of COX enzymes and the synthesis pathway from salicylic acid."),
    ]

    eval_data = []
    for domains, question in domain_pairs:
        eval_data.append({
            "question": question,
            "required_domains": domains,
            "n_domains": len(domains),
            "difficulty_estimate": "hard",
        })

    eval_path = Path(output_dir) / "multidomain_eval_v3.json"
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"\n✅ Created {len(eval_data)} multi-domain eval questions at {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="Curate training data for Synapta v2.0")
    parser.add_argument("--output-dir", default="data/training", help="Output directory")
    parser.add_argument("--eval-dir", default="data/eval", help="Eval data output directory")
    parser.add_argument("--domains", nargs="*", default=None, help="Specific domains to process (default: all)")
    parser.add_argument("--tier", type=int, default=None, help="Only process domains of this tier")
    args = parser.parse_args()

    print("🚀 Synapta v2.0 — Data Curation")
    print(f"   Output: {args.output_dir}")
    print()

    total = 0
    domains_to_process = DOMAINS

    if args.domains:
        domains_to_process = {k: v for k, v in DOMAINS.items() if k in args.domains}
    if args.tier:
        domains_to_process = {k: v for k, v in domains_to_process.items() if v["tier"] == args.tier}

    for domain_name, config in sorted(domains_to_process.items(), key=lambda x: x[1]["tier"]):
        print(f"\n{'='*50}")
        print(f"  Domain: {domain_name} (Tier {config['tier']})")
        print(f"  Target: {config['target_samples']} samples")
        print(f"{'='*50}")
        count = download_and_process_domain(domain_name, config, args.output_dir)
        total += count

    # Create multi-domain eval set
    create_multidomain_eval(args.eval_dir)

    print(f"\n{'='*50}")
    print(f"  TOTAL: {total} training samples across {len(domains_to_process)} domains")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
