#!/usr/bin/env python3
"""Phase 2: extract clean text from RBI/SEBI PDFs.

Primary extractor: pdfplumber. Fallback: pymupdf (fitz) when pdfplumber yields
empty / garbled output. Parallelism via multiprocessing.Pool(8).

Inputs (read-only):
  data/{rbi,sebi}_corpus/pdfs/*.PDF
  data/{rbi,sebi}_corpus/manifest.jsonl

Outputs:
  data/{rbi,sebi}_corpus/text/<stem>.txt   -- cleaned text
  data/{rbi,sebi}_corpus/text/<stem>.json  -- structured record
  logs/data_pipeline/02_extract.log        -- progress log
"""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pdfplumber
import fitz  # pymupdf

PROJECT = Path("/home/learner/Desktop/mewtwo")
LOG = PROJECT / "logs" / "data_pipeline" / "02_extract.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

CORPORA = [
    {
        "regulator": "RBI",
        "pdfs": PROJECT / "data" / "rbi_corpus" / "pdfs",
        "out": PROJECT / "data" / "rbi_corpus" / "text",
        "manifest": PROJECT / "data" / "rbi_corpus" / "manifest.jsonl",
    },
    {
        "regulator": "SEBI",
        "pdfs": PROJECT / "data" / "sebi_corpus" / "pdfs",
        "out": PROJECT / "data" / "sebi_corpus" / "text",
        "manifest": PROJECT / "data" / "sebi_corpus" / "manifest.jsonl",
    },
]

NUM_WORKERS = 8
MIN_GOOD_CHARS = 500

# ---------------------------------------------------------------------------
# Logging (file append + stdout). Workers also call this; safe enough for
# coarse-grained progress with O(N=80) writes.
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
_NON_PRINT = re.compile(r"[^\x09\x0A\x0D\x20-\x7E -￿]")
_MULTI_SPACE = re.compile(r"[ \t\f\v]+")
_MULTI_NL = re.compile(r"\n{3,}")
_PAGE_OF = re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)
_BARE_PAGENUM = re.compile(r"^\s*\d{1,3}\s*$")
_RBI_REF = re.compile(r"^\s*RBI/\d{2,4}-\d{2,4}/\S+\s*$")

# Section heading regexes
_HEAD_NUM = re.compile(r"^\d+\.?\s+[A-Z][A-Za-z][A-Za-z\s\-/&,]{2,80}$")
_HEAD_SUBNUM = re.compile(r"^\d+\.\d+(?:\.\d+)*\s+\S.{0,120}$")
_HEAD_ROMAN = re.compile(r"^[IVX]{1,5}\.\s+\S.{0,120}$")
_HEAD_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 \-/&,]{4,78}$")


def normalize_line(s: str) -> str:
    s = _NON_PRINT.sub("", s)
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip()


def find_repeated_lines(pages_lines: list[list[str]], threshold: float = 0.5) -> set[str]:
    """Lines appearing on >threshold fraction of pages are header/footer."""
    n = len(pages_lines)
    if n < 3:
        return set()
    c: Counter[str] = Counter()
    for plines in pages_lines:
        # First 3 / last 3 lines of each page are the usual culprits
        candidates = set(plines[:3] + plines[-3:])
        for ln in candidates:
            if 3 <= len(ln) <= 120:
                c[ln] += 1
    cutoff = max(2, int(threshold * n) + 1)
    return {ln for ln, k in c.items() if k >= cutoff}


def clean_text(pages_lines: list[list[str]]) -> str:
    repeats = find_repeated_lines(pages_lines)
    out_pages = []
    for plines in pages_lines:
        kept = []
        for ln in plines:
            ln_n = ln.strip()
            if not ln_n:
                kept.append("")
                continue
            if ln_n in repeats:
                continue
            if _PAGE_OF.match(ln_n) or _BARE_PAGENUM.match(ln_n) or _RBI_REF.match(ln_n):
                continue
            kept.append(ln_n)
        # collapse runs of blanks within page
        page_text = "\n".join(kept)
        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
        if page_text:
            out_pages.append(page_text)
    full = "\n\n".join(out_pages)
    full = _MULTI_NL.sub("\n\n", full)
    return full.strip()


def detect_sections(text: str) -> list[dict]:
    sections = []
    lines = text.split("\n")
    offsets = []
    cursor = 0
    for ln in lines:
        offsets.append(cursor)
        cursor += len(ln) + 1  # +1 for the \n
    headings: list[tuple[int, int, str]] = []  # (line_idx, char_offset, heading)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or len(s) > 140:
            continue
        if (
            _HEAD_SUBNUM.match(s)
            or _HEAD_NUM.match(s)
            or _HEAD_ROMAN.match(s)
            or (len(s) <= 80 and _HEAD_ALLCAPS.match(s))
        ):
            headings.append((i, offsets[i], s))
    for idx, (li, off, h) in enumerate(headings):
        next_off = headings[idx + 1][1] if idx + 1 < len(headings) else len(text)
        body = text[off + len(h) : next_off].strip()
        sections.append({"heading": h, "body": body, "start_char_offset": off})
    return sections


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_pdfplumber(pdf_path: Path) -> tuple[list[list[str]], list[list[list[str]]], int]:
    """Return (per-page lines, per-page tables, page_count)."""
    pages_lines: list[list[str]] = []
    pages_tables: list[list[list[str]]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            # Tables (best-effort)
            page_tables: list[list[str]] = []
            try:
                tbls = page.find_tables() or []
                table_bboxes = [t.bbox for t in tbls]
                for t in tbls:
                    rows = t.extract() or []
                    rendered = [
                        " | ".join((c or "").strip() for c in row) for row in rows
                    ]
                    page_tables.append(rendered)
            except Exception:
                table_bboxes = []
                page_tables = []

            def not_in_table(obj):
                if not table_bboxes:
                    return True
                cx = (obj["x0"] + obj["x1"]) / 2
                cy = (obj["top"] + obj["bottom"]) / 2
                for x0, top, x1, bottom in table_bboxes:
                    if x0 <= cx <= x1 and top <= cy <= bottom:
                        return False
                return True

            try:
                filtered = page.filter(not_in_table)
                txt = filtered.extract_text(x_tolerance=2, y_tolerance=3) or ""
            except Exception:
                txt = page.extract_text() or ""
            lines = [normalize_line(ln) for ln in txt.split("\n")]
            pages_lines.append(lines)
            pages_tables.append(page_tables)
    return pages_lines, pages_tables, page_count


def extract_pymupdf(pdf_path: Path) -> tuple[list[list[str]], int]:
    pages_lines: list[list[str]] = []
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    for page in doc:
        txt = page.get_text("text") or ""
        lines = [normalize_line(ln) for ln in txt.split("\n")]
        pages_lines.append(lines)
    doc.close()
    return pages_lines, page_count


def looks_garbled(text: str) -> bool:
    if not text:
        return True
    if len(text) < MIN_GOOD_CHARS:
        return True
    # ratio of alpha to total non-space chars
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return True
    alpha = sum(1 for c in nonspace if c.isalpha())
    return (alpha / len(nonspace)) < 0.55


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def process_pdf(args: tuple) -> dict:
    pdf_path_s, regulator, out_dir_s, title = args
    pdf_path = Path(pdf_path_s)
    out_dir = Path(out_dir_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    out_txt = out_dir / f"{stem}.txt"
    out_json = out_dir / f"{stem}.json"

    record = {
        "source_pdf": str(pdf_path),
        "regulator": regulator,
        "title": title,
        "text": "",
        "page_count": 0,
        "char_count": 0,
        "extractor_used": None,
        "sections": [],
        "tables": [],
        "status": "ok",
        "error": None,
    }

    extractor = "pdfplumber"
    try:
        pages_lines, pages_tables, pc = extract_pdfplumber(pdf_path)
        record["page_count"] = pc
        text = clean_text(pages_lines)
        if looks_garbled(text):
            extractor = "pymupdf"
            pages_lines2, pc2 = extract_pymupdf(pdf_path)
            record["page_count"] = pc2 or pc
            text = clean_text(pages_lines2)
            pages_tables = []  # fallback path skips table extraction
        record["extractor_used"] = extractor
        record["text"] = text
        record["char_count"] = len(text)
        record["tables"] = [pt for pt in pages_tables if pt]
        record["sections"] = detect_sections(text)
        if record["page_count"] == 0 or len(text) < MIN_GOOD_CHARS:
            record["status"] = "failed"
            record["error"] = f"insufficient text ({len(text)} chars, {record['page_count']} pages)"
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        record["extractor_used"] = extractor
        log(f"FAIL {pdf_path.name}: {record['error']}\n{traceback.format_exc(limit=2)}")

    try:
        out_txt.write_text(record["text"], encoding="utf-8")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
    except Exception as e:
        log(f"WRITE-FAIL {pdf_path.name}: {e}")

    return {
        "name": pdf_path.name,
        "status": record["status"],
        "chars": record["char_count"],
        "sections": len(record["sections"]),
        "extractor": record["extractor_used"],
        "regulator": regulator,
        "error": record["error"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_titles(manifest_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not manifest_path.exists():
        return out
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            lp = row.get("local_path") or ""
            stem = Path(lp).stem if lp else None
            title = row.get("title") or ""
            if stem:
                out[stem] = title
    return out


def main() -> int:
    jobs: list[tuple] = []
    for c in CORPORA:
        if not c["pdfs"].exists():
            log(f"skip {c['regulator']}: no pdf dir at {c['pdfs']}")
            continue
        titles = load_titles(c["manifest"])
        pdfs = sorted([p for p in c["pdfs"].iterdir() if p.suffix.lower() == ".pdf"])
        log(f"{c['regulator']}: queued {len(pdfs)} PDFs from {c['pdfs']}")
        for p in pdfs:
            jobs.append((str(p), c["regulator"], str(c["out"]), titles.get(p.stem, "")))

    if not jobs:
        log("no jobs to run; exiting.")
        return 0

    log(f"starting pool with {NUM_WORKERS} workers, {len(jobs)} total jobs")
    t0 = time.time()
    results: list[dict] = []
    with mp.Pool(NUM_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(process_pdf, jobs), 1):
            results.append(res)
            tag = "OK " if res["status"] == "ok" else "FAIL"
            log(
                f"[{i}/{len(jobs)}] {tag} {res['regulator']} {res['name']} "
                f"chars={res['chars']} sections={res['sections']} via={res['extractor']}"
            )
    dt = time.time() - t0

    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] != "ok"]
    total_chars = sum(r["chars"] for r in results)
    total_sections = sum(r["sections"] for r in results)
    summary = (
        f"DONE in {dt:.1f}s | total={len(results)} ok={len(ok)} failed={len(bad)} "
        f"total_chars={total_chars} total_sections={total_sections}"
    )
    log(summary)
    if bad:
        log("failures:")
        for r in bad:
            log(f"  - {r['regulator']} {r['name']}: {r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
