#!/usr/bin/env python3
"""Phase D: build extractive QA pairs from RBI/SEBI regulatory chunks.

Reads data/rbi_corpus/chunks/all_chunks.jsonl and emits train/eval JSONL into
data/rbi_corpus/qa/ via three deterministic tiers (no LLM):
  T1: native FAQ Q/A patterns (rare in RBI MDs)
  T2: numeric-claim regex extraction (currency / time / percent / refs / thresholds)
  T3: heading-based section QA (numbered headings only)

Doc-disjoint train/eval split (10 PDFs held out, seed=42).
"""
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT = Path("/home/learner/Desktop/mewtwo")
IN_FILE = PROJECT / "data" / "rbi_corpus" / "chunks" / "all_chunks.jsonl"
OUT_DIR = PROJECT / "data" / "rbi_corpus" / "qa"
TRAIN_FILE = OUT_DIR / "train.jsonl"
EVAL_FILE = OUT_DIR / "eval.jsonl"
SPLIT_MANIFEST = OUT_DIR / "split_manifest.json"
LOG_PATH = PROJECT / "logs" / "data_pipeline" / "04_qa.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("qa")

SEED = 42
EVAL_PDF_COUNT = 26  # 20% of 130 PDFs for held-out eval
MAX_PAIRS_PER_CHUNK = 8
MIN_ANSWER_LEN = 3
MAX_ANSWER_LEN = 300
TRAIN_TARGET_HI = 5000
EVAL_TARGET_HI = 700
TRAIN_TARGET_LO = 3000
EVAL_TARGET_LO = 400
HEADING_SEP = " --- "

NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
GENERIC_HEADINGS = {
    "definitions", "introduction", "short title", "preamble",
    "index", "acronyms", "applicability", "scope", "interpretation",
    "objective", "purpose", "title", "commencement", "extent",
}

# Tier 1 FAQ pattern (very rare in RBI MDs)
FAQ_QA_RE = re.compile(
    r"(?:^|\n)\s*(?:Q\d+\.?|Question\s+\d+:?|Q:)\s*(.+?)\s*\n"
    r"\s*(?:A\d+\.?|Answer\s*:?|A:)\s*(.+?)(?=\n\s*(?:Q\d+\.?|Question\s+\d+:?|Q:)|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# Tier 2 regexes
CURRENCY_RE = re.compile(
    r"(?:Rs\.?|₹|INR)\s*[\d,]+(?:\.\d+)?(?:\s*(?:crore|lakh|lakhs|million|billion|thousand))?",
    re.IGNORECASE,
)
TIME_RE = re.compile(
    r"(?:within|not\s+exceeding|not\s+later\s+than|maximum\s+of|minimum\s+of|"
    r"a\s+period\s+of|for\s+a\s+period\s+of)\s+(\d+\s+(?:days?|months?|years?|hours?|weeks?))",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:per\s*cent|percent|%)", re.IGNORECASE)
SECTION_REF_RE = re.compile(
    r"\b(Section|Para|Paragraph|Clause|Regulation)\s+\d+(?:\.\d+)*[A-Z]?\b"
)
THRESHOLD_RE = re.compile(
    r"(above|below|exceeding|less\s+than|more\s+than)\s+(?:Rs\.?|₹)\s*[\d,]+(?:\.\d+)?"
    r"(?:\s*(?:crore|lakh|lakhs|million|billion|thousand))?",
    re.IGNORECASE,
)


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def find_body_start(text: str) -> int:
    idx = text.find(HEADING_SEP)
    return idx + len(HEADING_SEP) if idx != -1 else 0


def clean_topic(heading: str) -> Optional[str]:
    m = re.match(r"^\d+(?:\.\d+)*\.?\s+(.*)$", heading.strip())
    if not m:
        return None
    topic = m.group(1).strip()
    if ":" in topic:
        topic = topic.split(":")[0].strip()
    tail_drop = {"an", "a", "the", "of", "to", "and", "or", "for", "in", "on", "is"}
    parts = topic.split()
    while parts and parts[-1].lower() in tail_drop:
        parts.pop()
    topic = " ".join(parts).rstrip(" ,;:-")
    if not topic or topic.lower() in GENERIC_HEADINGS or len(topic) < 4:
        return None
    return topic


def section_number_of(heading: str) -> Optional[str]:
    m = re.match(r"^(\d+(?:\.\d+)*)", heading.strip())
    return m.group(1) if m else None


def first_n_sentences(text: str, n: int = 3) -> str:
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    s = [x for x in s if x.strip()]
    return " ".join(s[:n]).strip() if s else ""


def is_clean_question(q: str) -> bool:
    if not q or len(q) < 8:
        return False
    return not any(b in q for b in ("{}", "{", "}", "None", "[", "]"))


def is_clean_answer(a: str) -> bool:
    if not a or len(a) < MIN_ANSWER_LEN or len(a) > MAX_ANSWER_LEN:
        return False
    return not any(b in a for b in ("{}", "None"))


def _norm_ws(s: str) -> str:
    """Collapse internal whitespace to single space; strip ends. Preserves char count semantics for matching."""
    return re.sub(r"\s+", " ", s).strip()


def make_qa(tier: int, chunk: dict, question: str, answer: str,
            ans_start: int, ans_end: int) -> Optional[dict]:
    raw_text = chunk["text"]
    # Normalize both context and answer so PDF soft-wraps ("RS\n21") become "RS 21".
    context = _norm_ws(raw_text)
    answer_clean = _norm_ws(answer)
    if not is_clean_question(question) or not is_clean_answer(answer_clean):
        return None
    # Re-find answer in normalized context (case-sensitive first, then case-insensitive)
    idx = context.find(answer_clean)
    if idx == -1:
        idx = context.lower().find(answer_clean.lower())
        if idx == -1:
            return None
        # Use the actual cased substring from context
        answer_clean = context[idx:idx + len(answer_clean)]
    ans_start, ans_end = idx, idx + len(answer_clean)
    qa_id = f"tier{tier}_{short_hash(chunk['chunk_id'] + '|' + question + '|' + answer_clean)}"
    return {
        "qa_id": qa_id,
        "tier": tier,
        "regulator": chunk.get("regulator", "RBI"),
        "source_pdf": chunk["source_pdf"],
        "context": context,
        "question": question,
        "answer": answer_clean,
        "answer_start_char": ans_start,
        "answer_end_char": ans_end,
        "section_heading": chunk.get("section_heading", ""),
    }


def topic_phrase(chunk: dict) -> str:
    h = chunk.get("section_heading", "") or ""
    topic = clean_topic(h)
    if topic:
        return topic[:80].rstrip() if len(topic) > 80 else topic
    title = (chunk.get("title", "") or "").split("(")[0].strip()
    if title.startswith("Master Direction"):
        title = title.replace("Master Direction", "").lstrip(" -–:")
    return (title[:80].rstrip() if len(title) > 80 else title) or "this provision"


def section_label_for(chunk: dict, topic: str) -> str:
    sl = chunk.get("section_heading", "") or topic
    return sl[:80].rstrip() if len(sl) > 80 else sl


def extract_tier1(chunk: dict) -> List[dict]:
    out = []
    text = chunk["text"]
    for m in FAQ_QA_RE.finditer(text):
        q = m.group(1).strip()
        a = m.group(2).strip()
        if len(a) > MAX_ANSWER_LEN:
            a = a[:MAX_ANSWER_LEN].rstrip()
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        ans_start = text.find(a)
        if ans_start == -1:
            continue
        qa = make_qa(1, chunk, q, a, ans_start, ans_start + len(a))
        if qa:
            out.append(qa)
    return out


# Tier 2 spec: (template_key, regex, question_builder using match)
def _t2_specs(chunk: dict, topic: str, sl: str) -> List[Tuple[str, re.Pattern, Callable[[re.Match], str]]]:
    return [
        ("currency", CURRENCY_RE,
         lambda m: f"What is the amount specified for {topic} under {sl}?"),
        ("time", TIME_RE,
         lambda m: f"What is the timeline specified for {topic} under {sl}?"),
        ("percent", PERCENT_RE,
         lambda m: f"What is the rate specified for {topic} under {sl}?"),
        ("section_ref", SECTION_REF_RE,
         lambda m: f"Which {m.group(1).lower()} governs {topic} under {sl}?"),
        ("threshold", THRESHOLD_RE,
         lambda m: f"What is the threshold {m.group(1).lower().split()[0]} which {topic} applies under {sl}?"),
    ]


def extract_tier2(chunk: dict) -> List[dict]:
    text = chunk["text"]
    topic = topic_phrase(chunk)
    sl = section_label_for(chunk, topic)
    occurrences: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    for tkey, pat, qbuild in _t2_specs(chunk, topic, sl):
        for m in pat.finditer(text):
            ans_start, ans_end = m.span()
            # trim trailing punctuation that isn't part of the value
            while ans_end > ans_start and text[ans_end - 1] in ".,;)":
                ans_end -= 1
            ans = text[ans_start:ans_end].strip()
            if not ans:
                continue
            if text[ans_start:ans_end] != ans:
                # locate inside original span area
                idx = text.find(ans, m.start())
                if idx == -1:
                    continue
                ans_start, ans_end = idx, idx + len(ans)
            q = qbuild(m)
            qa = make_qa(2, chunk, q, ans, ans_start, ans_end)
            if qa:
                occurrences[(tkey, ans.lower())].append(qa)

    final: List[dict] = []
    seen_ids = set()
    for key, qas in occurrences.items():
        # dedup rule: if same (template, value) appears 3+ times, emit only first
        picked = qas[:1] if len(qas) >= 3 else qas
        for qa in picked:
            if qa["qa_id"] in seen_ids:
                continue
            seen_ids.add(qa["qa_id"])
            final.append(qa)
    return final


def extract_tier3(chunk: dict) -> List[dict]:
    text = chunk["text"]
    heading = chunk.get("section_heading", "") or ""
    if not NUMBERED_HEADING_RE.match(heading):
        return []
    topic = clean_topic(heading)
    section_num = section_number_of(heading)
    if not topic or not section_num:
        return []
    title = (chunk.get("title", "") or "").split("(")[0].strip() or "the regulation"
    title_short = title[:100].rstrip() if len(title) > 100 else title

    body_start = find_body_start(text)
    body = text[body_start:].strip()
    if not body:
        return []
    snippet = first_n_sentences(body, n=3)
    if not snippet or len(snippet) < 30:
        return []
    if len(snippet) > MAX_ANSWER_LEN:
        snippet = first_n_sentences(body, n=1)
        if len(snippet) > MAX_ANSWER_LEN:
            snippet = snippet[:MAX_ANSWER_LEN].rstrip()
    ans_start = text.find(snippet)
    if ans_start == -1:
        return []
    q = f"What does paragraph {section_num} of the {title_short} regulation cover regarding {topic}?"
    qa = make_qa(3, chunk, q, snippet, ans_start, ans_start + len(snippet))
    return [qa] if qa else []


def main() -> int:
    if not IN_FILE.exists():
        log.error(f"input not found: {IN_FILE}")
        return 1

    chunks: List[dict] = []
    with open(IN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info(f"loaded {len(chunks)} chunks")

    pdfs = sorted({c["source_pdf"] for c in chunks})
    log.info(f"unique pdfs: {len(pdfs)}")
    rng = random.Random(SEED)
    eval_pdfs = set(rng.sample(pdfs, min(EVAL_PDF_COUNT, len(pdfs))))
    train_pdfs = set(pdfs) - eval_pdfs
    log.info(f"eval pdfs: {len(eval_pdfs)}; train pdfs: {len(train_pdfs)}")

    train_qas: List[dict] = []
    eval_qas: List[dict] = []
    tier_counts = Counter()

    for chunk in chunks:
        per_chunk: List[dict] = []
        per_chunk.extend(extract_tier1(chunk))
        per_chunk.extend(extract_tier2(chunk))
        per_chunk.extend(extract_tier3(chunk))
        if len(per_chunk) > MAX_PAIRS_PER_CHUNK:
            per_chunk = per_chunk[:MAX_PAIRS_PER_CHUNK]
        for qa in per_chunk:
            tier_counts[qa["tier"]] += 1
            (eval_qas if chunk["source_pdf"] in eval_pdfs else train_qas).append(qa)

    log.info(f"raw tier counts: {dict(tier_counts)}")
    log.info(f"raw train: {len(train_qas)}, raw eval: {len(eval_qas)}")

    rng2 = random.Random(SEED + 1)
    if len(train_qas) > TRAIN_TARGET_HI:
        rng2.shuffle(train_qas)
        train_qas = train_qas[:TRAIN_TARGET_HI]
        log.info(f"downsampled train to {TRAIN_TARGET_HI}")
    if len(eval_qas) > EVAL_TARGET_HI:
        rng2.shuffle(eval_qas)
        eval_qas = eval_qas[:EVAL_TARGET_HI]
        log.info(f"downsampled eval to {EVAL_TARGET_HI}")

    if len(train_qas) < TRAIN_TARGET_LO:
        log.warning(f"train below target: {len(train_qas)} < {TRAIN_TARGET_LO}")
    if len(eval_qas) < EVAL_TARGET_LO:
        log.warning(f"eval below target: {len(eval_qas)} < {EVAL_TARGET_LO}")

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for qa in train_qas:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for qa in eval_qas:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    final_tier_counts = Counter(qa["tier"] for qa in train_qas + eval_qas)
    train_tier_counts = Counter(qa["tier"] for qa in train_qas)
    eval_tier_counts = Counter(qa["tier"] for qa in eval_qas)

    manifest = {
        "seed": SEED,
        "eval_pdf_count": len(eval_pdfs),
        "train_pdf_count": len(train_pdfs),
        "eval_pdfs": sorted(eval_pdfs),
        "train_pdfs": sorted(train_pdfs),
        "train_size": len(train_qas),
        "eval_size": len(eval_qas),
        "tier_counts_train": {str(k): v for k, v in train_tier_counts.items()},
        "tier_counts_eval": {str(k): v for k, v in eval_tier_counts.items()},
        "tier_counts_total": {str(k): v for k, v in final_tier_counts.items()},
    }
    with open(SPLIT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    def mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    all_qas = train_qas + eval_qas
    q_lens = [len(qa["question"]) for qa in all_qas]
    a_lens = [len(qa["answer"]) for qa in all_qas]
    c_lens = [len(qa["context"]) for qa in all_qas]

    log.info(f"final tier counts: {dict(final_tier_counts)}")
    log.info(f"train: {len(train_qas)}  eval: {len(eval_qas)}")
    log.info(f"train tiers: {dict(train_tier_counts)}  eval tiers: {dict(eval_tier_counts)}")
    log.info(f"mean question_len: {mean(q_lens)}  mean answer_len: {mean(a_lens)}  mean context_len: {mean(c_lens)}")

    print("=== Phase D QA build summary ===")
    print(f"input chunks: {len(chunks)}  unique PDFs: {len(pdfs)}")
    print(f"eval PDFs held out: {len(eval_pdfs)}  train PDFs: {len(train_pdfs)}")
    print(f"tier counts (total): {dict(final_tier_counts)}")
    print(f"tier counts (train): {dict(train_tier_counts)}")
    print(f"tier counts (eval):  {dict(eval_tier_counts)}")
    print(f"train size: {len(train_qas)}  eval size: {len(eval_qas)}")
    print(f"mean question length: {mean(q_lens)}")
    print(f"mean answer length:   {mean(a_lens)}")
    print(f"mean context length:  {mean(c_lens)}")
    print(f"wrote: {TRAIN_FILE}")
    print(f"wrote: {EVAL_FILE}")
    print(f"wrote: {SPLIT_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
