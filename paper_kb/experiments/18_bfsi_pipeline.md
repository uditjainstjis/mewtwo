# Experiment Card 18 — BFSI Corpus and 3-Tier Q&A Pipeline

PKB wrapper around `RESEARCH_HISTORY/10_bfsi_pipeline.md`.

## 1. Research question
Build a deterministic, contamination-resistant Indian regulatory QA training set without LLM-generated questions.

## 2. Dataset construction
- 80 RBI Master Directions + 50 SEBI Master Circulars = 130 PDFs (115 MB), all public-domain government text.
- Text extraction: pdfplumber + PyMuPDF (multiprocessing). Total 8.06M chars.
- Smart chunking on numbered section boundaries: 7,329 sections detected, 4,185 chunks (median 384 tokens).
- 3-tier deterministic regex Q&A construction:
  - Tier 1 native FAQ: 1,142 pairs.
  - Tier 2 numeric extractive: 2,219 pairs.
  - Tier 3 heading-extractive: 1,116 pairs.
  - Total raw: 4,477 pairs.
- 10-check validator: 98.45\% pass after v2 cleaner pass.
- Final: **2,931 train + 664 eval pairs.**

## 3. Document-disjoint split
- 130 PDFs → 104 train (80\%) + 26 held-out (20\%).
- Q&A pairs assigned to train/eval based solely on which PDF their context came from.
- Eliminates the chunk-neighbour memorisation pathway~\citep{magar2022data}.
- Manifest: `data/rbi_corpus/qa/split_manifest_v2.json`.

## 4. Evaluation
- Not directly evaluated here; pipeline produces the split consumed by `19_bfsi_extract_eval_n664.md` and `20_bfsi_recall_eval_n214.md`.

## 5. Results
- See cards 19, 20 for downstream results.

## 6. Negatives + caveats
- F1$\geq$0.5 cut-off finding (see card 22) reveals that the model's verbose paragraph-extraction style does not satisfy strict token F1 even when semantically correct. This is a metric finding, not a pipeline bug.

## 7. Artifact map
PRIMARY:
- `synapta_src/data_pipeline/{01_scrape_rbi_mds, 01c_scrape_sebi_circulars, 02_extract_text, 03_chunk_circulars, 04_build_qa_pairs, 04b_build_qa_pairs_v2, 06_validate_qa}.py`
- `data/rbi_corpus/{pdfs, manifest.json, qa/{eval_clean, train_v2_clean, split_manifest_v2}.jsonl}`
- `data/sebi_corpus/{pdfs, manifest.json}`

SECONDARY:
- `RESEARCH_HISTORY/10_bfsi_pipeline.md`
