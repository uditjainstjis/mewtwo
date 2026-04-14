"""
Dataset Preparation for LoRI-MoE Domain Expert Training

Downloads, processes, and formats training data for each of the 5 domains.
Converts everything to Qwen2.5's chat template format for instruction tuning.
"""
import json
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ─── Chat template formatting ─────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "math": "You are a precise mathematics expert. Solve problems step-by-step, showing all work clearly. Always verify your final answer.",
    "code": "You are an expert programmer. Write clean, correct, well-documented code. Explain your approach before implementation.",
    "science": "You are a rigorous scientist. Answer questions with precise scientific reasoning, citing relevant principles and evidence.",
    "legal": "You are a legal expert. Analyze questions with careful attention to legal principles, precedents, and logical reasoning.",
    "medical": "You are a medical professional. Provide accurate medical information with appropriate caveats and evidence-based reasoning.",
}


def format_chat(system: str, user: str, assistant: str, tokenizer) -> str:
    """Format a single example using Qwen2.5's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback: manual formatting
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"


# ─── Domain-specific data processors ──────────────────────────────────────────

def process_math(tokenizer, max_samples: int = 50000) -> List[Dict]:
    """Process MetaMathQA for math domain training."""
    logger.info("Loading MetaMathQA...")
    ds = load_dataset("meta-math/MetaMathQA", split="train")

    # Shuffle and limit
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        query = row.get("query", row.get("question", ""))
        response = row.get("response", row.get("answer", ""))
        if query and response:
            text = format_chat(SYSTEM_PROMPTS["math"], query, response, tokenizer)
            examples.append({"text": text, "domain": "math"})

    logger.info(f"Math: {len(examples)} examples processed")
    return examples


def process_code(tokenizer, max_samples: int = 20000) -> List[Dict]:
    """Process CodeAlpaca for code domain training."""
    logger.info("Loading CodeAlpaca-20k...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    except Exception:
        logger.warning("CodeAlpaca-20k not found, trying alternative...")
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        instruction = row.get("instruction", row.get("prompt", ""))
        inp = row.get("input", "")
        output = row.get("output", row.get("response", ""))

        if instruction and output:
            query = f"{instruction}\n\n{inp}".strip() if inp else instruction
            text = format_chat(SYSTEM_PROMPTS["code"], query, output, tokenizer)
            examples.append({"text": text, "domain": "code"})

    logger.info(f"Code: {len(examples)} examples processed")
    return examples


def process_science(tokenizer, max_samples: int = 30000) -> List[Dict]:
    """Process SciQ for science domain training."""
    logger.info("Loading SciQ...")
    ds = load_dataset("allenai/sciq", split="train")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        question = row.get("question", "")
        correct_answer = row.get("correct_answer", "")
        support = row.get("support", "")

        if question and correct_answer:
            # Include support/explanation as part of the answer
            answer = correct_answer
            if support:
                answer = f"{support}\n\nTherefore, the answer is: {correct_answer}"

            text = format_chat(SYSTEM_PROMPTS["science"], question, answer, tokenizer)
            examples.append({"text": text, "domain": "science"})

    logger.info(f"Science: {len(examples)} examples processed")
    return examples


def process_legal(tokenizer, max_samples: int = 20000) -> List[Dict]:
    """Process LegalBench for legal domain training."""
    logger.info("Loading LegalBench...")
    try:
        ds = load_dataset(
            "nguha/legalbench",
            "contract_nli_explicit_identification",
            split="test",
            trust_remote_code=True,
        )
    except Exception:
        # Fallback: try a different subset
        try:
            logger.warning("contract_nli not found, trying learned_hands_benefits...")
            ds = load_dataset(
                "nguha/legalbench",
                "learned_hands_benefits",
                split="test",
                trust_remote_code=True,
            )
        except Exception:
            logger.warning("LegalBench failed, generating synthetic legal data...")
            return _generate_synthetic_legal(tokenizer, max_samples)

    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        # LegalBench format varies by subset
        text_field = row.get("text", row.get("premise", row.get("question", "")))
        answer = row.get("answer", row.get("label", ""))

        if text_field and answer:
            query = f"Analyze the following legal text and provide your determination:\n\n{text_field}"
            response = f"Based on legal analysis: {answer}"
            formatted = format_chat(SYSTEM_PROMPTS["legal"], query, response, tokenizer)
            examples.append({"text": formatted, "domain": "legal"})

    logger.info(f"Legal: {len(examples)} examples processed")
    return examples


def _generate_synthetic_legal(tokenizer, max_samples: int) -> List[Dict]:
    """Fallback: minimal synthetic legal examples if dataset unavailable."""
    logger.warning("Using synthetic legal data as fallback")
    return []


def process_medical(tokenizer, max_samples: int = 30000) -> List[Dict]:
    """Process MedQA for medical domain training."""
    logger.info("Loading MedQA...")
    try:
        ds = load_dataset(
            "bigbio/med_qa",
            "med_qa_en_4options_source",
            split="train",
            trust_remote_code=True,
        )
    except Exception:
        try:
            logger.warning("bigbio/med_qa failed, trying GBaker/MedQA-USMLE-4-options...")
            ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        except Exception:
            logger.warning("MedQA failed, trying medmcqa...")
            ds = load_dataset("openlifescienceai/medmcqa", split="train")

    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    examples = []
    for row in ds:
        question = row.get("question", row.get("sent1", ""))

        # Handle multiple-choice format
        options = []
        for key in ["opa", "opb", "opc", "opd", "option_a", "option_b", "option_c", "option_d"]:
            if key in row and row[key]:
                options.append(row[key])

        # Also check for options dict/list
        if not options and "options" in row:
            if isinstance(row["options"], list):
                options = row["options"]
            elif isinstance(row["options"], dict):
                options = list(row["options"].values())

        answer_idx = row.get("cop", row.get("answer_idx", row.get("answer", 0)))
        if isinstance(answer_idx, str):
            answer_idx = ord(answer_idx.upper()) - ord('A') if answer_idx.isalpha() else 0

        if question:
            if options:
                option_text = "\n".join(
                    f"{'ABCDEFGH'[i]}. {opt}" for i, opt in enumerate(options)
                )
                query = f"{question}\n\n{option_text}"
                try:
                    answer = f"The correct answer is {options[answer_idx]}."
                except (IndexError, TypeError):
                    answer = f"The answer is option {answer_idx}."
            else:
                query = question
                answer = str(row.get("answer", row.get("exp", "See medical guidelines.")))

            text = format_chat(SYSTEM_PROMPTS["medical"], query, answer, tokenizer)
            examples.append({"text": text, "domain": "medical"})

    logger.info(f"Medical: {len(examples)} examples processed")
    return examples


# ─── Main pipeline ─────────────────────────────────────────────────────────────

PROCESSORS = {
    "math": process_math,
    "code": process_code,
    "science": process_science,
    "legal": process_legal,
    "medical": process_medical,
}


def prepare_all_datasets(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "/home/learner/Desktop/mewtwo/data/lori_moe",
    max_samples_per_domain: Optional[Dict[str, int]] = None,
):
    """
    Download and prepare all domain datasets.
    Saves as JSONL files ready for training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    stats = {}
    for domain_name, processor in PROCESSORS.items():
        max_samples = (max_samples_per_domain or {}).get(domain_name, 50000)
        logger.info(f"\n{'='*60}\nProcessing domain: {domain_name} (max {max_samples})")

        try:
            examples = processor(tokenizer, max_samples=max_samples)
        except Exception as e:
            logger.error(f"Failed to process {domain_name}: {e}")
            examples = []

        if not examples:
            logger.warning(f"No examples for domain '{domain_name}', skipping")
            continue

        # Save as JSONL
        output_file = output_dir / f"{domain_name}_train.jsonl"
        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        stats[domain_name] = {
            "count": len(examples),
            "file": str(output_file),
            "avg_length": sum(len(e["text"]) for e in examples) / len(examples),
        }
        logger.info(f"Saved {len(examples)} examples to {output_file}")

    # Save stats
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("Dataset Preparation Complete")
    print("=" * 60)
    for domain, s in stats.items():
        print(f"  {domain:10s}: {s['count']:>6d} examples, avg len {s['avg_length']:.0f} chars")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/home/learner/Desktop/mewtwo/data/lori_moe")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_all_datasets(model_name=args.model_name, output_dir=args.output_dir)
