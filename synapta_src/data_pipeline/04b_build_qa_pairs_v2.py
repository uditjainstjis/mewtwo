#!/usr/bin/env python3
"""Phase D v2: improved deterministic QA-pair builder for RBI/SEBI chunks.

Same I/O contract as 04_build_qa_pairs.py (chunks -> train/eval JSONL) and same
output schema, but tightens templates/filters to remove the worst awkwardness:

  1. Stronger junk-heading filter (rejects "RBI/...", "PART I", "INTRODUCTION",
     all-caps banners, roman-numeral headings, single-word headings, page strings).
  2. Stronger topic cleaner (expanded stop-set, reject sentence-fragment topics
     by length and verb/aux start).
  3. Tier-2 "under {sl}" suffix now uses the *section number alone* (e.g.
     "section 8") and is dropped entirely when no section number exists, so we no
     longer duplicate the heading text inside the question.
  4. Section-ref template reworded to avoid "Which section governs ... under
     <heading containing section text>".
  5. 3 deterministic paraphrase variants per Tier-2 template and 3 per Tier-3,
     selected by a stable hash of (chunk_id, tier, template-key).

Same SEED=42, same EVAL_PDF_COUNT=26, same downsampling/targets, same schema.
Outputs to train_v2.jsonl / eval_v2.jsonl (DOES NOT TOUCH the v1 files).
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
TRAIN_FILE = OUT_DIR / "train_v2.jsonl"
EVAL_FILE = OUT_DIR / "eval_v2.jsonl"
SPLIT_MANIFEST = OUT_DIR / "split_manifest_v2.json"
LOG_PATH = PROJECT / "logs" / "data_pipeline" / "04b_qa_v2.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("qa_v2")

# Constants kept identical to v1 for direct comparability.
SEED = 42
EVAL_PDF_COUNT = 26
MAX_PAIRS_PER_CHUNK = 8
MIN_ANSWER_LEN = 3
MAX_ANSWER_LEN = 300
TRAIN_TARGET_HI = 5000
EVAL_TARGET_HI = 700
TRAIN_TARGET_LO = 3000
EVAL_TARGET_LO = 400
HEADING_SEP = " --- "

NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")

# Headings to drop outright (case-insensitive match against the cleaned heading).
GENERIC_HEADINGS = {
    "definitions", "introduction", "short title", "preamble",
    "index", "acronyms", "applicability", "scope", "interpretation",
    "objective", "purpose", "title", "commencement", "extent",
    "circular", "annexure", "appendix", "schedule", "notification",
    "background", "general", "objectives", "abbreviations",
    "list of abbreviations", "table of contents", "contents",
    "reserve bank of india", "securities and exchange board of india",
    "financial markets regulation department",
}

# Junk heading patterns that indicate a chunk's heading is not a real topic.
JUNK_HEADING_PATTERNS = [
    re.compile(r"^(RBI|SEBI|FED|DBR|DBOD|DOR|DPSS|DGBA|MPD)\s*/", re.IGNORECASE),
    re.compile(r"^PART\b", re.IGNORECASE),
    re.compile(r"^CHAPTER\b", re.IGNORECASE),
    re.compile(r"^SECTION\s+[IVX]+\b", re.IGNORECASE),
    re.compile(r"^[IVX]+\.\s"),                                  # "V. ..." roman
    re.compile(r"^[A-Z]\.\s"),                                   # "A. ..." letter list
    re.compile(r"^page\s+\d+", re.IGNORECASE),
    re.compile(r"^\(?\s*pages?\s+\d+\s*[,\-]", re.IGNORECASE),
    re.compile(r"^Annexure\b", re.IGNORECASE),
    re.compile(r"^Appendix\b", re.IGNORECASE),
    re.compile(r"^Schedule\b", re.IGNORECASE),
    re.compile(r"^Form\s+", re.IGNORECASE),
    re.compile(r"^Table\b", re.IGNORECASE),
    re.compile(r"^Figure\b", re.IGNORECASE),
]

# Topic cleaner: words that should never end a topic (auxiliaries, prepositions,
# conjunctions, modal verbs, common sentence-fragment trailers).
TAIL_DROP = {
    "an", "a", "the", "of", "to", "and", "or", "for", "in", "on", "is",
    "by", "with", "as", "from", "at", "into", "onto", "over", "under",
    "shall", "will", "may", "must", "should", "would", "could",
    "has", "have", "had", "are", "was", "were", "be", "been", "being",
    "that", "which", "who", "whom", "whose",
    "such", "any", "all", "each", "every", "this", "these", "those",
    "than", "then", "but", "if", "though", "while", "since", "because",
    "i.e.", "e.g.", "etc", "etc.",
    "made", "given", "based", "subject",
}

# Words at the start of a topic that signal it's a sentence fragment, not a noun phrase.
LEADING_VERB_REJECT = {
    "shall", "will", "may", "must", "should", "would", "could",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "and", "or", "but", "however", "the", "a", "an",
    "above", "below", "all", "any", "such", "this", "these", "those", "that",
    "in", "on", "at", "of", "by", "for", "with", "as", "from", "to",
    # third-person pronouns starting a sentence
    "it", "they", "he", "she", "we", "you",
    # bare verbs that commonly begin instruction sentences
    "host", "ensure", "provide", "submit", "furnish", "report", "issue", "allow",
    "obtain", "include", "exclude", "comply", "monitor", "review", "approve",
    "accept", "permit", "make", "take", "give", "store", "send", "receive",
    "carry", "follow", "specify", "verify", "implement", "examine", "consider",
    "notify", "inform", "treat", "deduct", "add", "credit", "debit", "open",
    "close", "transfer", "remit", "purchase", "sell", "deal",
}

MAX_TOPIC_WORDS = 11
MIN_TOPIC_WORDS = 2

# Tier 1 FAQ pattern (rare in RBI MDs).
FAQ_QA_RE = re.compile(
    r"(?:^|\n)\s*(?:Q\d+\.?|Question\s+\d+:?|Q:)\s*(.+?)\s*\n"
    r"\s*(?:A\d+\.?|Answer\s*:?|A:)\s*(.+?)(?=\n\s*(?:Q\d+\.?|Question\s+\d+:?|Q:)|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# Tier 2 regexes (unchanged).
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


def variant_index(chunk_id: str, tier: int, tkey: str, n: int) -> int:
    """Stable deterministic variant selector keyed on (chunk_id, tier, template)."""
    h = hashlib.sha1(f"{chunk_id}|t{tier}|{tkey}".encode("utf-8")).hexdigest()
    return int(h, 16) % n


def find_body_start(text: str) -> int:
    idx = text.find(HEADING_SEP)
    return idx + len(HEADING_SEP) if idx != -1 else 0


def is_junk_heading(heading: str) -> bool:
    h = heading.strip()
    if not h:
        return True
    if any(p.search(h) for p in JUNK_HEADING_PATTERNS):
        return True
    # All-caps with no digits and 1-6 words is almost certainly a banner/header.
    if h.isupper() and not any(c.isdigit() for c in h) and 0 < len(h.split()) <= 6:
        return True
    if h.lower().rstrip(":.,;") in GENERIC_HEADINGS:
        return True
    return False


def clean_topic(heading: str) -> Optional[str]:
    """Extract a clean noun-phrase topic from a numbered heading.

    Rejects partial-sentence fragments by length, leading verb, and
    trailing-stopword heuristics.
    """
    if is_junk_heading(heading):
        return None
    m = re.match(r"^\d+(?:\.\d+)*\.?\s+(.*)$", heading.strip())
    if not m:
        return None
    topic = m.group(1).strip()
    # Cut at colon (clean noun phrases come before ":").
    if ":" in topic:
        topic = topic.split(":")[0].strip()
    # Cut at em-dash / hyphen if it leads into a sentence-like clause.
    for sep in (" – ", " — ", " - "):
        if sep in topic:
            head = topic.split(sep)[0].strip()
            if len(head.split()) >= MIN_TOPIC_WORDS:
                topic = head
                break
    # Iterative tail-drop until stable.
    parts = topic.split()
    changed = True
    while changed and parts:
        changed = False
        if parts[-1].lower().rstrip(",.;:-") in TAIL_DROP:
            parts.pop()
            changed = True
    topic = " ".join(parts).rstrip(" ,;:-")
    if not topic:
        return None
    if topic.lower() in GENERIC_HEADINGS or len(topic) < 4:
        return None
    words = topic.split()
    if len(words) < MIN_TOPIC_WORDS:
        return None
    if len(words) > MAX_TOPIC_WORDS:
        return None
    if words[0].lower() in LEADING_VERB_REJECT:
        return None
    # Reject if topic ends with an open-class function word we missed.
    if words[-1].lower() in TAIL_DROP:
        return None
    # Reject brackets/digit-debris fragments (e.g. "12.2 below.]42").
    if re.search(r"[\[\]]", topic):
        return None
    # Reject if half or more of tokens are non-alphabetic (numeric refs).
    alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha < max(2, (len(words) + 1) // 2):
        return None
    # Require at least one capitalised noun-like token (proper noun or Title-case).
    if not any(w[:1].isupper() for w in words):
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
    return re.sub(r"\s+", " ", s).strip()


def make_qa(tier: int, chunk: dict, question: str, answer: str,
            ans_start: int, ans_end: int) -> Optional[dict]:
    raw_text = chunk["text"]
    context = _norm_ws(raw_text)
    answer_clean = _norm_ws(answer)
    if not is_clean_question(question) or not is_clean_answer(answer_clean):
        return None
    idx = context.find(answer_clean)
    if idx == -1:
        idx = context.lower().find(answer_clean.lower())
        if idx == -1:
            return None
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


def topic_phrase(chunk: dict) -> Optional[str]:
    """Return a clean topic noun-phrase or None (caller should skip when None).

    Primary source is the section heading. Title fallback is allowed but
    carefully sanitised so we never produce double-preposition awkwardness
    like "for for X". The fallback returns the bare topic (no leading "for")
    and Tier-2 templates supply their own preposition.
    """
    h = chunk.get("section_heading", "") or ""
    topic = clean_topic(h)
    if topic:
        return topic[:80].rstrip()
    # Fallback: derive topic from document title for valid Master regulations.
    title = (chunk.get("title", "") or "").split("(")[0].strip()
    if not title:
        return None
    # Strip "Master Direction(s) – " / "Master Circular for/on/-" prefixes.
    title = re.sub(
        r"^Master\s+(?:Directions?|Circulars?)\s*(?:[-–:]\s*|for\s+|on\s+|for\s+the\s+)?",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    title = title.rstrip(",.;:- ")
    if not title:
        return None
    words = title.split()
    if not (MIN_TOPIC_WORDS <= len(words) <= MAX_TOPIC_WORDS):
        return None
    if words[0].lower() in LEADING_VERB_REJECT:
        return None
    if not any(w[:1].isupper() for w in words):
        return None
    return title


def section_suffix(chunk: dict) -> str:
    """Return ' under section <num>' / ' under paragraph <num>' / '' (no suffix).

    Critically, this NEVER repeats the heading text — only the numeric label.
    """
    h = chunk.get("section_heading", "") or ""
    num = section_number_of(h)
    if not num:
        return ""
    # Choose label by depth: top-level -> "section", deeper -> "paragraph".
    label = "section" if num.count(".") == 0 else "paragraph"
    return f" under {label} {num}"


def regulation_phrase(chunk: dict) -> str:
    """Best-effort short name of the source regulation, for Tier 3 questions.

    Trim to a word boundary at <=80 chars so we never end mid-word like
    "Persons Resident in In".
    """
    title = (chunk.get("title", "") or "").split("(")[0].strip()
    if not title:
        return "this regulation"
    if len(title) > 80:
        # word-boundary trim
        cut = title[:80].rsplit(" ", 1)[0]
        title = cut if len(cut) >= 30 else title[:80]
    return title.rstrip(",.;:- ")


# ---------------- Tier 1 ----------------
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


# ---------------- Tier 2 ----------------
# Each spec maps to a list of paraphrase variants. The variant is selected
# deterministically per (chunk, template). All variants embed `topic` and the
# clean numeric `suffix` (or no suffix when no section number).
def _t2_question_variants(tkey: str, topic: str, suffix: str, m: re.Match) -> List[str]:
    if tkey == "currency":
        return [
            f"What is the amount specified for {topic}{suffix}?",
            f"What monetary value is prescribed for {topic}{suffix}?",
            f"How much is specified for {topic}{suffix}?",
        ]
    if tkey == "time":
        return [
            f"What is the timeline specified for {topic}{suffix}?",
            f"Within what period must {topic} be carried out{suffix}?",
            f"What time limit applies to {topic}{suffix}?",
        ]
    if tkey == "percent":
        return [
            f"What is the rate specified for {topic}{suffix}?",
            f"What percentage is prescribed for {topic}{suffix}?",
            f"At what rate does {topic} apply{suffix}?",
        ]
    if tkey == "section_ref":
        ref_type = m.group(1).lower()
        # No "under {sl}" suffix here — answer is itself a reference.
        return [
            f"Which {ref_type} is referenced in connection with {topic}?",
            f"Under which {ref_type} is {topic} dealt with?",
            f"What {ref_type} number is cited regarding {topic}?",
        ]
    if tkey == "threshold":
        direction = m.group(1).lower().split()[0]
        return [
            f"What is the threshold {direction} which {topic} applies{suffix}?",
            f"What monetary limit ({direction}) is set for {topic}{suffix}?",
            f"What is the cut-off amount {direction} which {topic} is governed{suffix}?",
        ]
    return []


def _t2_specs() -> List[Tuple[str, re.Pattern]]:
    return [
        ("currency", CURRENCY_RE),
        ("time", TIME_RE),
        ("percent", PERCENT_RE),
        ("section_ref", SECTION_REF_RE),
        ("threshold", THRESHOLD_RE),
    ]


def extract_tier2(chunk: dict) -> List[dict]:
    text = chunk["text"]
    topic = topic_phrase(chunk)
    if not topic:
        return []
    suffix = section_suffix(chunk)

    occurrences: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for tkey, pat in _t2_specs():
        for m in pat.finditer(text):
            ans_start, ans_end = m.span()
            while ans_end > ans_start and text[ans_end - 1] in ".,;)":
                ans_end -= 1
            ans = text[ans_start:ans_end].strip()
            if not ans:
                continue
            if text[ans_start:ans_end] != ans:
                idx = text.find(ans, m.start())
                if idx == -1:
                    continue
                ans_start, ans_end = idx, idx + len(ans)
            variants = _t2_question_variants(tkey, topic, suffix, m)
            if not variants:
                continue
            vi = variant_index(chunk["chunk_id"], 2, tkey, len(variants))
            q = variants[vi]
            qa = make_qa(2, chunk, q, ans, ans_start, ans_end)
            if qa:
                occurrences[(tkey, ans.lower())].append(qa)

    final: List[dict] = []
    seen_ids = set()
    for key, qas in occurrences.items():
        picked = qas[:1] if len(qas) >= 3 else qas
        for qa in picked:
            if qa["qa_id"] in seen_ids:
                continue
            seen_ids.add(qa["qa_id"])
            final.append(qa)
    return final


# ---------------- Tier 3 ----------------
def _t3_question_variants(reg: str, section_num: str, topic: str) -> List[str]:
    return [
        f"What does paragraph {section_num} of the {reg} cover regarding {topic}?",
        f"In the {reg}, what does section {section_num} state about {topic}?",
        f"What is provided under paragraph {section_num} of the {reg} concerning {topic}?",
    ]


def extract_tier3(chunk: dict) -> List[dict]:
    text = chunk["text"]
    heading = chunk.get("section_heading", "") or ""
    if not NUMBERED_HEADING_RE.match(heading):
        return []
    if is_junk_heading(heading):
        return []
    topic = clean_topic(heading)
    section_num = section_number_of(heading)
    if not topic or not section_num:
        return []
    reg = regulation_phrase(chunk)

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
    variants = _t3_question_variants(reg, section_num, topic)
    vi = variant_index(chunk["chunk_id"], 3, "section", len(variants))
    q = variants[vi]
    qa = make_qa(3, chunk, q, snippet, ans_start, ans_start + len(snippet))
    return [qa] if qa else []


# ---------------- Driver ----------------
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
    junk_skipped = 0

    for chunk in chunks:
        # Pre-filter: if heading is junk and there's no usable topic anywhere,
        # the chunk yields nothing -- but tier1 (FAQ) doesn't need a topic.
        per_chunk: List[dict] = []
        per_chunk.extend(extract_tier1(chunk))
        per_chunk.extend(extract_tier2(chunk))
        per_chunk.extend(extract_tier3(chunk))
        if not per_chunk:
            junk_skipped += 1
        if len(per_chunk) > MAX_PAIRS_PER_CHUNK:
            per_chunk = per_chunk[:MAX_PAIRS_PER_CHUNK]
        for qa in per_chunk:
            tier_counts[qa["tier"]] += 1
            (eval_qas if chunk["source_pdf"] in eval_pdfs else train_qas).append(qa)

    log.info(f"raw tier counts: {dict(tier_counts)}")
    log.info(f"raw train: {len(train_qas)}, raw eval: {len(eval_qas)}")
    log.info(f"chunks producing zero QAs: {junk_skipped}")

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
        "version": "v2",
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
        "chunks_yielding_zero": junk_skipped,
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

    print("=== Phase D v2 QA build summary ===")
    print(f"input chunks: {len(chunks)}  unique PDFs: {len(pdfs)}")
    print(f"chunks yielding zero QAs: {junk_skipped}")
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
