#!/usr/bin/env python3
"""Phase 3: chunk extracted RBI/SEBI circular text into token-aware QA contexts.

Reads data/{rbi,sebi}_corpus/text/*.json from upstream extractor #02 and emits
data/rbi_corpus/chunks/all_chunks.jsonl. Prefers section boundaries; merges
short sections with same-parent siblings; splits long sections on paragraph
then sentence boundaries; never mid-sentence; dedupes on hash of first 200
chars. If no upstream input yet: log "waiting for upstream" and exit 0.
"""
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

PROJECT = Path("/home/learner/Desktop/mewtwo")
RBI_TEXT_DIR = PROJECT / "data" / "rbi_corpus" / "text"
SEBI_TEXT_DIR = PROJECT / "data" / "sebi_corpus" / "text"
OUT_DIR = PROJECT / "data" / "rbi_corpus" / "chunks"
OUT_FILE = OUT_DIR / "all_chunks.jsonl"
LOG_PATH = PROJECT / "logs" / "data_pipeline" / "03_chunk.log"
TOKENIZER_PATH = PROJECT / "models" / "nemotron"

MIN_TOKENS = 400
MAX_TOKENS = 800
MIN_BODY_CHARS = 100
MIN_BODY_WORDS = 30
HEADING_SEP = " --- "

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("chunk")

_TOKENIZER = None
_TRIED_LOAD = False


def _get_tokenizer():
    global _TOKENIZER, _TRIED_LOAD
    if _TRIED_LOAD:
        return _TOKENIZER
    _TRIED_LOAD = True
    try:
        from transformers import AutoTokenizer
        log.info(f"Loading Nemotron tokenizer from {TOKENIZER_PATH}")
        _TOKENIZER = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True)
    except Exception as e:
        log.warning(f"Tokenizer unavailable ({e!r}); using heuristic word*1.3")
    return _TOKENIZER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok is None:
        return int(len(text.split()) * 1.3)
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:
        return int(len(text.split()) * 1.3)


NUM_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def _heading_parent(h: Optional[str]) -> Optional[str]:
    if not h:
        return None
    m = NUM_HEADING_RE.match(h)
    if not m:
        return None
    parts = m.group(1).split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else parts[0]


def _split_long_body(heading: str, body: str) -> List[str]:
    """Split too-long body into <=MAX_TOKENS chunks; never mid-sentence.
    Each returned chunk already includes the heading prefix."""
    units: List[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()):
        if count_tokens(para) > MAX_TOKENS:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
        else:
            units.append(para)
    chunks: List[str] = []
    cur: List[str] = []
    cur_toks = 0
    head_toks = count_tokens(heading + HEADING_SEP)
    for u in units:
        ut = count_tokens(u)
        if ut + head_toks > MAX_TOKENS and not cur:
            chunks.append(heading + HEADING_SEP + u)  # oversize singleton
            continue
        if cur_toks + ut + head_toks > MAX_TOKENS:
            chunks.append(heading + HEADING_SEP + "\n\n".join(cur))
            cur, cur_toks = [u], ut
        else:
            cur.append(u)
            cur_toks += ut
    if cur:
        chunks.append(heading + HEADING_SEP + "\n\n".join(cur))
    return chunks


def _is_useful(body: str) -> bool:
    return len(body) >= MIN_BODY_CHARS and len(body.split()) >= MIN_BODY_WORDS

def chunk_document(doc: Dict) -> List[Dict]:
    sections = doc.get("sections") or []
    title = doc.get("title", "")
    regulator = doc.get("regulator", "RBI")
    src_pdf = Path(doc.get("source_pdf", "")).name
    if not sections:
        full_text = (doc.get("text") or "").strip()
        if not full_text:
            return []
        sections = [{"heading": title or "Document", "body": full_text}]

    # Merge short sections forward (same numeric parent only, if both numbered).
    merged: List[Dict] = []
    i = 0
    while i < len(sections):
        s = {"heading": sections[i].get("heading"), "body": (sections[i].get("body") or "").strip()}
        cur_tok = count_tokens(s["body"])
        j = i + 1
        while cur_tok < MIN_TOKENS and j < len(sections):
            p_cur, p_nxt = _heading_parent(s["heading"]), _heading_parent(sections[j].get("heading"))
            if p_cur and p_nxt and p_cur != p_nxt:
                break
            s["body"] = (s["body"] + "\n\n" + (sections[j].get("body") or "").strip()).strip()
            cur_tok = count_tokens(s["body"])
            j += 1
        merged.append(s)
        i = j if j > i + 1 else i + 1

    # Split overlong sections; build final chunk texts (with heading prefix).
    raw_chunks: List[Dict] = []
    for s in merged:
        body = s["body"]
        if not _is_useful(body):
            continue
        heading = (s.get("heading") or title or "Section").strip()
        prefix_tok = count_tokens(heading + HEADING_SEP)
        if count_tokens(body) + prefix_tok <= MAX_TOKENS:
            texts = [heading + HEADING_SEP + body]
        else:
            texts = _split_long_body(heading, body)
        for t in texts:
            if _is_useful(t.split(HEADING_SEP, 1)[-1]):
                raw_chunks.append({"heading": heading, "text": t})

    src_hash = hashlib.sha1(doc.get("source_pdf", "").encode("utf-8")).hexdigest()[:10]
    n = len(raw_chunks)
    return [{
        "chunk_id": f"{regulator}_{src_hash}_{idx}",
        "regulator": regulator,
        "source_pdf": src_pdf,
        "title": title,
        "section_heading": rc["heading"],
        "text": rc["text"],
        "token_count": count_tokens(rc["text"]),
        "char_count": len(rc["text"]),
        "chunk_idx": idx,
        "total_chunks_in_doc": n,
    } for idx, rc in enumerate(raw_chunks)]


def gather_inputs() -> List[Path]:
    files = []
    for d in (RBI_TEXT_DIR, SEBI_TEXT_DIR):
        if d.is_dir():
            files.extend(sorted(d.glob("*.json")))
    return files

def main() -> int:
    log.info("=== 03_chunk_circulars start ===")
    inputs = gather_inputs()
    if not inputs:
        log.info("waiting for upstream (no files in data/{rbi,sebi}_corpus/text/)")
        return 0
    log.info(f"Found {len(inputs)} extractor outputs to chunk")

    seen_hashes = set()
    per_reg = {"RBI": 0, "SEBI": 0}
    buckets = {"0-400": 0, "400-800": 0, "800+": 0}
    token_sum = 0
    total = 0
    skipped_dupes = 0
    failed_docs = 0

    t0 = time.time()
    with open(OUT_FILE, "w") as fout:
        for fp in inputs:
            try:
                doc = json.loads(fp.read_text())
            except Exception as e:
                log.warning(f"Skip {fp.name}: bad JSON ({e!r})")
                failed_docs += 1
                continue
            try:
                chunks = chunk_document(doc)
            except Exception as e:
                log.warning(f"Chunk failure on {fp.name}: {e!r}")
                failed_docs += 1
                continue
            for c in chunks:
                key = hashlib.sha1(c["text"][:200].encode("utf-8")).hexdigest()
                if key in seen_hashes:
                    skipped_dupes += 1
                    continue
                seen_hashes.add(key)
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                total += 1
                per_reg[c["regulator"]] = per_reg.get(c["regulator"], 0) + 1
                tc = c["token_count"]
                token_sum += tc
                bk = "0-400" if tc < 400 else ("400-800" if tc <= 800 else "800+")
                buckets[bk] += 1
            log.info(f"{fp.name}: emitted {len(chunks)} chunks (raw)")

    dt = time.time() - t0
    mean_tok = (token_sum / total) if total else 0.0
    summary = {
        "source_docs": len(inputs),
        "failed_docs": failed_docs,
        "chunks_emitted": total,
        "deduped": skipped_dupes,
        "per_regulator": per_reg,
        "token_buckets": buckets,
        "mean_tokens": round(mean_tok, 1),
        "elapsed_sec": round(dt, 1),
        "output": str(OUT_FILE),
    }
    log.info(f"summary: {summary}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
