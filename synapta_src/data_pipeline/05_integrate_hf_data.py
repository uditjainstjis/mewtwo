#!/usr/bin/env python3
"""Phase 5: integrate HuggingFace BFSI datasets into our standard schemas.

Sources (downloaded under data/hf_bfsi/):
  - prakhar146/indian-finance-rag (Apache-2.0, 24,780 pre-chunked rows)
  - lirus18/rbi (llama2, 1,709 Q&A pairs as <s>[INST] q [/INST] a </s>)
  - iam-sathya/rbi-test (OpenRAIL, 98 raw RBI Master Direction docs)

Outputs:
  - APPENDS chunk rows to data/rbi_corpus/chunks/all_chunks.jsonl
    (matches existing schema; adds top-level `source` and `license` fields)
  - WRITES QA rows to data/rbi_corpus/qa/hf_qa.jsonl

Filters:
  - Skip merged chunks with body < 100 chars or > 8000 chars
  - Skip rows with obvious script/style cruft (JS/CSS blocks)
  - Skip rows that look mostly non-Latin (rough heuristic)
  - Dedupe against first-200-char hashes already in all_chunks.jsonl
  - Drop the prakhar146 'embedding' column
  - iam-sathya has duplicate Text/Text1; we keep Text only

Token counting uses the heuristic words*1.3 (no model load).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---- paths -----------------------------------------------------------------
PROJECT = Path("/home/learner/Desktop/mewtwo")
HF_DIR = PROJECT / "data" / "hf_bfsi"
PRAKHAR_FILE = HF_DIR / "prakhar146_indian-finance-rag" / "train.jsonl"
LIRUS_FILE = HF_DIR / "lirus18_rbi" / "train.jsonl"
SATHYA_FILE = HF_DIR / "iam-sathya_rbi-test" / "train.jsonl"

CHUNKS_OUT = PROJECT / "data" / "rbi_corpus" / "chunks" / "all_chunks.jsonl"
QA_OUT = PROJECT / "data" / "rbi_corpus" / "qa" / "hf_qa.jsonl"

LOG_PATH = PROJECT / "logs" / "data_pipeline" / "05_integrate_hf.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
QA_OUT.parent.mkdir(parents=True, exist_ok=True)
CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hf_integrate")

# ---- limits / params -------------------------------------------------------
MIN_CHARS = 100
MAX_CHARS = 8000
TARGET_TOKENS = 600   # merge target for prakhar / iam-sathya
MAX_TOKENS = 800
HEADING_SEP = " --- "

# ---- regex patterns --------------------------------------------------------
LIRUS_PATTERN = re.compile(
    r"<s>\s*\[INST\]\s*(.+?)\s*\[/INST\]\s*(.+?)\s*</s>",
    re.DOTALL,
)
JS_CSS_RE = re.compile(
    r"(<script\b|</script>|<style\b|</style>|function\s*\(\s*\)\s*\{|var\s+\w+\s*=\s*function|\bdocument\.getElementById)",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")


# ---- helpers ---------------------------------------------------------------
def count_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) if text else 0


def is_mostly_latin(text: str) -> bool:
    """Reject rows where >40% chars are outside basic Latin/punct (rough)."""
    if not text:
        return False
    sample = text[:1500]
    latin = sum(1 for c in sample if c.isascii())
    return latin / max(1, len(sample)) >= 0.6


def has_script_cruft(text: str) -> bool:
    return bool(JS_CSS_RE.search(text or ""))


def normalize_ws(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(text[:200].encode("utf-8")).hexdigest()


# ---- regulator inference for prakhar146 sources ---------------------------
# 32 source filenames; only a handful are clearly regulator publications.
# Most are Zerodha Varsity educational modules (categorised as OTHER).
PRAKHAR_REGULATOR_MAP: Dict[str, str] = {
    # Clearly SEBI documents
    "05_SEBI_Circular_dated_09_February_2026.pdf": "SEBI",
    "SEBI Booklet.pdf": "SEBI",
    "Securities Market Booklet.pdf": "SEBI",
    "Investor_Grievance_Redressal_Mechanism_SEBI_Scores_NSE_BSE_NSDL_CDSL_1.pdf": "SEBI",
    "Introduction to Securities Markets_2.pdf": "SEBI",
    "How_to_buy_and_sell_shares_in_Stock_Exchange.pdf": "SEBI",
    "How_to_invest_in_Intial_Public_Offer.pdf": "SEBI",
    "How_to_Invest_in_Rights_Issue.pdf": "SEBI",
    "Investing safely in Capital Market_0.pdf": "SEBI",
    "Depository_Services.pdf": "SEBI",
    "KYC_Procedure_(Opening_of_Trading_and_Demat_Account).pdf": "SEBI",
    "Corporate_Action_Dividends_Bonus_splits_etc.pdf": "SEBI",
    "Introduction to Exchange Traded Funds .pdf.pdf": "SEBI",
    "REITs_InvITs_Presentaion.pdf": "SEBI",
    "Introduction_to_Mutual_Funds_Investing.pdf": "SEBI",
    "BEAWARE07032022.pdf": "SEBI",
    "PR1183ENG151121.pdf": "SEBI",
    # AMFI
    "AMFI_Investor_Trends_Feb2026_e05e1dcb05.pdf": "OTHER",
    # Hash-named PDF (unknown) — keep OTHER, content review didn't disambiguate
    "59FM04072F58B1DD44DFADD486B9B0A59E9D.pdf": "OTHER",
}
# Default map: Zerodha Varsity modules → OTHER (educational, not regulator)


def prakhar_regulator(source_filename: str) -> str:
    return PRAKHAR_REGULATOR_MAP.get(source_filename, "OTHER")


# ---- existing-chunks dedupe ------------------------------------------------
def load_existing_hashes() -> set:
    hashes: set = set()
    if not CHUNKS_OUT.exists():
        return hashes
    n = 0
    with CHUNKS_OUT.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("text") or ""
            if t:
                hashes.add(text_hash(t))
                n += 1
    log.info(f"Loaded {n} existing chunks; {len(hashes)} unique 200-char hashes")
    return hashes


# ---- prakhar146: merge tiny chunks per (source, page) ---------------------
def process_prakhar(existing_hashes: set) -> Tuple[List[Dict], Dict[str, int]]:
    """Group rows by (source, page-bucket), accumulate text to ~TARGET_TOKENS,
    drop embedding column, then filter and emit chunks."""
    stats = defaultdict(int)
    if not PRAKHAR_FILE.exists():
        log.warning(f"prakhar146 file not found: {PRAKHAR_FILE}")
        return [], stats

    # Ingest, group preserving order
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    n_in = 0
    with PRAKHAR_FILE.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            row.pop("embedding", None)
            src = row.get("source", "unknown.pdf")
            grouped[src].append(row)
            n_in += 1
    stats["rows_in"] = n_in
    log.info(f"prakhar146: read {n_in} rows from {len(grouped)} source files")

    out: List[Dict] = []
    global_idx = 0
    for src, rows in grouped.items():
        regulator = prakhar_regulator(src)
        # Concatenate consecutive rows until target tokens reached.
        cur_texts: List[str] = []
        cur_pages: List[str] = []
        cur_tokens = 0
        # Track first chunk_id and page for traceability in section_heading
        cur_first_chunk_id = None

        def flush():
            nonlocal cur_texts, cur_pages, cur_tokens, cur_first_chunk_id, global_idx
            if not cur_texts:
                return
            text = normalize_ws(" ".join(cur_texts))
            cur_texts, cur_tokens = [], 0
            pages = sorted(set(cur_pages), key=lambda p: (len(p), p))
            cur_pages = []
            first_id = cur_first_chunk_id
            cur_first_chunk_id = None
            if not text:
                stats["empty_after_join"] += 1
                return
            if len(text) < MIN_CHARS:
                stats["too_short"] += 1
                return
            if len(text) > MAX_CHARS:
                # Trim safely; this happens only if a single row was huge.
                text = text[:MAX_CHARS]
            if has_script_cruft(text):
                stats["script_cruft"] += 1
                return
            if not is_mostly_latin(text):
                stats["non_latin"] += 1
                return
            h = text_hash(text)
            if h in existing_hashes:
                stats["dedup_existing"] += 1
                return
            existing_hashes.add(h)
            page_str = ",".join(pages) if pages else ""
            section_heading = f"page {page_str}" if page_str else (first_id or "section")
            out.append({
                "chunk_id": f"HF_prakhar_{global_idx}",
                "regulator": regulator,
                "source_pdf": f"hf:prakhar146/{src}",
                "title": Path(src).stem.replace("_", " "),
                "section_heading": section_heading,
                "text": text,
                "token_count": count_tokens(text),
                "char_count": len(text),
                "chunk_idx": global_idx,
                "source": "hf",
                "license": "Apache-2.0",
            })
            global_idx += 1
            stats["emitted"] += 1

        for row in rows:
            t = (row.get("text") or "").strip()
            if not t:
                continue
            if cur_first_chunk_id is None:
                cur_first_chunk_id = row.get("chunk_id")
            cur_texts.append(t)
            cur_pages.append(str(row.get("page", "")))
            cur_tokens += count_tokens(t)
            if cur_tokens >= TARGET_TOKENS:
                flush()
        flush()

    log.info(f"prakhar146: stats {dict(stats)}")
    return out, dict(stats)


# ---- iam-sathya: chunk full RBI Master Direction docs ---------------------
def split_doc_to_chunks(text: str) -> List[str]:
    """Paragraph-aware split to ~TARGET_TOKENS chunks, MAX_TOKENS hard cap.
    Never splits mid-sentence."""
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        # No double-newlines; fall back to single-newline split
        paras = [p.strip() for p in text.splitlines() if p.strip()]

    # Pre-split paragraphs that are themselves too long
    units: List[str] = []
    for p in paras:
        if count_tokens(p) > MAX_TOKENS:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
            units.extend(sentences if sentences else [p])
        else:
            units.append(p)

    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for u in units:
        ut = count_tokens(u)
        if cur and cur_tok + ut > MAX_TOKENS:
            chunks.append(" ".join(cur))
            cur, cur_tok = [u], ut
        else:
            cur.append(u)
            cur_tok += ut
            if cur_tok >= TARGET_TOKENS:
                chunks.append(" ".join(cur))
                cur, cur_tok = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def process_sathya(existing_hashes: set) -> Tuple[List[Dict], Dict[str, int]]:
    stats = defaultdict(int)
    if not SATHYA_FILE.exists():
        log.warning(f"iam-sathya file not found: {SATHYA_FILE}")
        return [], stats

    out: List[Dict] = []
    global_idx = 0
    with SATHYA_FILE.open() as f:
        for doc_idx, line in enumerate(f):
            try:
                row = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            stats["rows_in"] += 1
            # Text == Text1 (verified) — use Text only.
            full_text = row.get("Text") or row.get("Text1") or ""
            full_text = full_text.strip()
            if not full_text:
                stats["empty_doc"] += 1
                continue
            if has_script_cruft(full_text):
                stats["script_cruft_doc"] += 1
                continue
            if not is_mostly_latin(full_text):
                stats["non_latin_doc"] += 1
                continue

            # Try to extract a title from the first non-trivial line
            first_lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()][:5]
            title = ""
            for ln in first_lines:
                if "Master Direction" in ln or "Notification" in ln or "Circular" in ln:
                    title = ln[:200]
                    break
            if not title and first_lines:
                title = first_lines[0][:200]

            sub_chunks = split_doc_to_chunks(full_text)
            for ci, ctext in enumerate(sub_chunks):
                ctext = normalize_ws(ctext)
                if len(ctext) < MIN_CHARS:
                    stats["too_short"] += 1
                    continue
                if len(ctext) > MAX_CHARS:
                    ctext = ctext[:MAX_CHARS]
                h = text_hash(ctext)
                if h in existing_hashes:
                    stats["dedup_existing"] += 1
                    continue
                existing_hashes.add(h)
                out.append({
                    "chunk_id": f"HF_sathya_{global_idx}",
                    "regulator": "RBI",
                    "source_pdf": f"hf:iam-sathya/rbi-test#doc{doc_idx}",
                    "title": title,
                    "section_heading": f"part {ci+1}",
                    "text": ctext,
                    "token_count": count_tokens(ctext),
                    "char_count": len(ctext),
                    "chunk_idx": global_idx,
                    "source": "hf",
                    "license": "OpenRAIL",
                })
                global_idx += 1
                stats["emitted"] += 1

    log.info(f"iam-sathya: stats {dict(stats)}")
    return out, dict(stats)


# ---- lirus18: parse llama2 chat format -> QA -------------------------------
def process_lirus(existing_hashes: set) -> Tuple[List[Dict], Dict[str, int]]:
    """Note: existing_hashes here is for QA dedupe on (q+a) hash."""
    stats = defaultdict(int)
    if not LIRUS_FILE.exists():
        log.warning(f"lirus18 file not found: {LIRUS_FILE}")
        return [], stats

    out: List[Dict] = []
    seen_qa: set = set()
    with LIRUS_FILE.open() as f:
        for idx, line in enumerate(f):
            try:
                row = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            stats["rows_in"] += 1
            text = row.get("text") or ""
            m = LIRUS_PATTERN.search(text)
            if not m:
                stats["unparseable"] += 1
                continue
            q = normalize_ws(m.group(1))
            a = normalize_ws(m.group(2))
            if not q or not a:
                stats["empty_qa"] += 1
                continue
            if len(a) < 30 or len(q) < 10:
                stats["too_short_qa"] += 1
                continue
            if len(q) + len(a) > MAX_CHARS:
                stats["too_long_qa"] += 1
                continue
            if has_script_cruft(q) or has_script_cruft(a):
                stats["script_cruft"] += 1
                continue
            if not (is_mostly_latin(q) and is_mostly_latin(a)):
                stats["non_latin"] += 1
                continue
            key = hashlib.sha1((q + "|" + a[:200]).encode("utf-8")).hexdigest()
            if key in seen_qa:
                stats["dedup_intra"] += 1
                continue
            seen_qa.add(key)
            out.append({
                "qa_id": f"hf_lirus18_{idx}",
                "tier": "hf_native",
                "regulator": "RBI",
                "source_pdf": "hf:lirus18/rbi",
                "context": "",
                "question": q,
                "answer": a,
                "section_heading": "(unknown)",
                "source": "hf",
                "license": "llama2",
            })
            stats["emitted"] += 1

    log.info(f"lirus18: stats {dict(stats)}")
    return out, dict(stats)


# ---- main ------------------------------------------------------------------
def append_chunks(rows: List[Dict]) -> None:
    if not rows:
        return
    with CHUNKS_OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_qa(rows: List[Dict]) -> None:
    with QA_OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_two(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows[:2]:
        rc = dict(r)
        if "text" in rc and len(rc["text"]) > 220:
            rc["text"] = rc["text"][:220] + "..."
        if "answer" in rc and len(rc["answer"]) > 220:
            rc["answer"] = rc["answer"][:220] + "..."
        out.append(rc)
    return out


def main() -> int:
    log.info("=== 05_integrate_hf_data start ===")
    existing_hashes = load_existing_hashes()
    pre_existing_count = sum(1 for _ in CHUNKS_OUT.open()) if CHUNKS_OUT.exists() else 0

    prakhar_chunks, prakhar_stats = process_prakhar(existing_hashes)
    sathya_chunks, sathya_stats = process_sathya(existing_hashes)
    lirus_qa, lirus_stats = process_lirus(existing_hashes)

    # Append chunks to corpus
    all_new_chunks = prakhar_chunks + sathya_chunks
    append_chunks(all_new_chunks)
    write_qa(lirus_qa)

    post_count = sum(1 for _ in CHUNKS_OUT.open()) if CHUNKS_OUT.exists() else 0

    summary = {
        "pre_existing_chunks": pre_existing_count,
        "post_chunks": post_count,
        "added_chunks": post_count - pre_existing_count,
        "prakhar146": prakhar_stats,
        "iam_sathya": sathya_stats,
        "lirus18_qa": lirus_stats,
        "qa_pairs_written": len(lirus_qa),
        "samples": {
            "prakhar146": sample_two(prakhar_chunks),
            "iam_sathya": sample_two(sathya_chunks),
            "lirus18": sample_two(lirus_qa),
        },
        "outputs": {
            "chunks": str(CHUNKS_OUT),
            "qa": str(QA_OUT),
        },
    }
    log.info("=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
