# BFSI Corpus and 3-Tier Q&A Pipeline

**Date:** 2026-05-03
**Source artefacts:**
- `synapta_src/data_pipeline/01_scrape_rbi_mds.py` (RBI scraper)
- `synapta_src/data_pipeline/01c_scrape_sebi_circulars.py` (SEBI scraper)
- `synapta_src/data_pipeline/02_extract_text.py` (PDF → text)
- `synapta_src/data_pipeline/03_chunk_circulars.py` (smart chunking)
- `synapta_src/data_pipeline/04_build_qa_pairs.py` (3-tier construction v1)
- `synapta_src/data_pipeline/04b_build_qa_pairs_v2.py` (cleaner v2)
- `synapta_src/data_pipeline/06_validate_qa.py` (10-check validator)
- `data/rbi_corpus/manifest.json`
- `data/sebi_corpus/manifest.json`
- `data/rbi_corpus/qa/split_manifest_v2.json`

## Source documents
- 80 RBI Master Directions (period 2015–2025; banking ops, FX, NBFC supervision, payment systems, cyber resilience).
- 50 SEBI Master Circulars (mutual funds, listing obligations, FPIs, intermediaries).
- All public-domain government text obtained via authenticated public URLs.
- 130 PDFs, 115 MB total, SHA-256 manifests in `data/{rbi,sebi}_corpus/manifest.json`.

## Text extraction (8.06 M chars)
- pdfplumber for body text (with PyMuPDF fallback for scanned/embedded pages).
- Multiprocessing across PDFs.
- 7,329 sections detected via regex over numbered structure.

## Smart chunking (4,185 chunks)
- Target chunk length: 425 tokens.
- Median: 384 tokens.
- Boundaries respect numbered section headers (Part / Chapter / Section / Sub-paragraph).
- A question grounded in section 3.4.2 retrieves a chunk that begins at the start of 3.4.2, not mid-paragraph.

## 3-tier deterministic Q&A construction (4,477 raw pairs)

**No LLM is used to generate questions or answers.**

### Tier 1: native FAQ (1,142 pairs)
- Where the regulator provides explicit FAQ sections, lift (question, answer) verbatim.
- Gold = regulator's own words.

### Tier 2: numeric extraction (2,219 pairs)
- Regex matches on Indian-currency amounts (Rs., ₹), percentages, time periods, structured citations.
- Question generated from surrounding sentence; gold = matched span.

### Tier 3: heading-based extractive (1,116 pairs)
- Each non-trivial heading becomes a question of the form "What does [heading] specify?"
- Gold = immediately following paragraph.

## 10-check validator (98.45% pass rate)
1. Min/max length on question.
2. Min/max length on gold answer.
3. No chunk-id leakage between question and context.
4. Gold answer must appear (substring or punctuation-normalised) within the context paragraph.
5. No template artefacts in question text.
6. Question ends with `?`.
7. Context provides enough scaffolding (min word count).
8. Gold has no header-leakage.
9. No duplicate question text within the corpus.
10. No mojibake / encoding errors.

After v2 cleaner pass: **2,931 train + 664 eval pairs**.

## Document-disjoint train/eval split
- 130 PDFs → 104 train (80%) + 26 held-out (20%).
- Q&A pairs assigned to train/eval based solely on which PDF their context came from.
- The held-out set is a held-out **sample of documents**, not a held-out sample of pairs.
- Eliminates the chunk-neighbour memorisation pathway~\citep{magar2022data}.

Manifest: `data/rbi_corpus/qa/split_manifest_v2.json`.

## Files
- `synapta_src/data_pipeline/{01..06}*` (scrape → extract → chunk → QA → validate)
- `data/{rbi,sebi}_corpus/pdfs/` (130 source PDFs)
- `data/{rbi,sebi}_corpus/manifest.json` (SHA-256 manifests)
- `data/rbi_corpus/qa/{train_v2_clean,eval_clean}.jsonl`
- `data/rbi_corpus/qa/split_manifest_v2.json`
