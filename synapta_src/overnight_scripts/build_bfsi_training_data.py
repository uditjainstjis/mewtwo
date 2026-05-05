#!/usr/bin/env python3
"""Build BFSI compliance training dataset from hand-curated questions.

Strategy: paraphrase augmentation (no external LLM call) — each question
expanded into multiple variants:
  - 3 question rephrasings
  - 2 system-prompt variants
  - With-context AND without-context variants

Total target: ~400-500 training examples from 55 seed questions.
Output: data/rbi_circulars/bfsi_train.jsonl (HuggingFace SFT format)
"""
import json, sys, random
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT / "data" / "rbi_circulars"))

from questions import QUESTIONS as Q1
from questions_v2_no_context import QUESTIONS_V2 as Q2

OUT = PROJECT / "data" / "rbi_circulars" / "bfsi_train.jsonl"

# Question rephrasing templates — apply transformations to add variation
REPHRASE_PREFIXES = [
    "Under RBI rules, ",
    "According to RBI guidelines, ",
    "Per the RBI Master Direction, ",
    "What does RBI say about: ",
    "From a banking compliance perspective, ",
    "For an Indian bank, ",
]

SYSTEM_PROMPTS = [
    "You are a senior banking and financial regulation expert in India. Answer precisely with the specific number, term, or rule. Be concise.",
    "You are a compliance officer at an Indian bank. Cite the relevant RBI/SEBI/IRDAI rule precisely.",
    "You are an expert on Indian financial regulations. Provide direct, specific answers grounded in the actual regulations.",
]


def make_with_context_examples(q):
    """For v1 questions (have context), build training examples that teach
    extraction from regulatory text."""
    examples = []
    context = q.get("context", "")
    if not context:
        return examples
    for sys_msg in SYSTEM_PROMPTS[:2]:
        # canonical phrasing
        user = f"REGULATION CONTEXT:\n{context}\n\nQUESTION: {q['question']}\n\nANSWER:"
        assistant = q['gold_answer']
        # Make assistant more verbose and natural
        if q.get("scoring") == "multi_term":
            terms = q.get("alternatives", [])[:5]
            assistant = f"Per the regulation context, the answer involves: {', '.join(terms)}."
        else:
            assistant = f"Per the regulation: {q['gold_answer']}."
        examples.append({
            "system": sys_msg,
            "user": user,
            "assistant": assistant,
            "circular": q.get("circular", "unknown"),
            "type": "with_context",
            "id": q["id"],
        })
        # Rephrased version
        for prefix in REPHRASE_PREFIXES[:2]:
            new_q = prefix + q['question'].lstrip("Under RBI rules, ").lstrip("According to RBI guidelines, ").lstrip("What does ").lower()
            new_q = new_q[0].upper() + new_q[1:] if new_q else new_q
            examples.append({
                "system": sys_msg,
                "user": f"REGULATION CONTEXT:\n{context}\n\nQUESTION: {new_q}\n\nANSWER:",
                "assistant": assistant,
                "circular": q.get("circular", "unknown"),
                "type": "with_context_rephrased",
                "id": q["id"],
            })
    return examples


def make_no_context_examples(q):
    """For v2 questions (no context), build training examples that teach
    recall of BFSI knowledge."""
    examples = []
    gold = q.get("gold_answer", "")
    alts = q.get("alternatives", [])
    # Build a more verbose canonical answer using alternatives as supporting facts
    answer_parts = [gold]
    if q.get("scoring") == "multi_term" and alts:
        answer_parts = [", ".join(alts[:5])]
    canonical_answer = answer_parts[0]
    source = q.get("source", "")
    if source:
        canonical_answer += f" (per {source})"

    for sys_msg in SYSTEM_PROMPTS:
        # canonical
        examples.append({
            "system": sys_msg,
            "user": f"QUESTION: {q['question']}\n\nANSWER:",
            "assistant": canonical_answer,
            "type": "no_context",
            "id": q["id"],
            "source": source,
        })
        # rephrased
        for prefix in REPHRASE_PREFIXES[:3]:
            base_q = q['question']
            for old in ["Under RBI rules, ", "According to RBI guidelines, ", "Per RBI, ", "What"]:
                if base_q.startswith(old):
                    base_q = base_q[len(old):]
            new_q = prefix + base_q.lower()
            new_q = new_q[0].upper() + new_q[1:] if new_q else new_q
            examples.append({
                "system": sys_msg,
                "user": f"QUESTION: {new_q}\n\nANSWER:",
                "assistant": canonical_answer,
                "type": "no_context_rephrased",
                "id": q["id"],
                "source": source,
            })
    return examples


def make_chain_of_thought_examples(q):
    """For numeric questions, build examples that show step-by-step reasoning."""
    if q.get("scoring") != "contains":
        return []
    if not q.get("gold_answer"):
        return []
    examples = []
    cot_answer = (
        f"Looking at the regulation: the specific rule is found in {q.get('source', 'the relevant RBI Master Direction')}. "
        f"The answer is: {q['gold_answer']}."
    )
    sys_msg = "You are a senior banking compliance expert. Reason step-by-step and provide the specific regulatory answer."
    examples.append({
        "system": sys_msg,
        "user": f"QUESTION: {q['question']}\n\nReason briefly, then give the specific answer.\n\nANSWER:",
        "assistant": cot_answer,
        "type": "cot",
        "id": q["id"],
        "source": q.get("source", ""),
    })
    return examples


def main():
    random.seed(42)
    all_examples = []

    print(f"V1 questions (with context): {len(Q1)}")
    for q in Q1:
        all_examples.extend(make_with_context_examples(q))
        all_examples.extend(make_chain_of_thought_examples(q))

    print(f"V2 questions (no context): {len(Q2)}")
    for q in Q2:
        all_examples.extend(make_no_context_examples(q))
        all_examples.extend(make_chain_of_thought_examples(q))

    # Shuffle
    random.shuffle(all_examples)

    print(f"\nTotal training examples: {len(all_examples)}")

    type_counts = {}
    for ex in all_examples:
        t = ex.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print("Type breakdown:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote: {OUT}")
    print(f"File size: {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
