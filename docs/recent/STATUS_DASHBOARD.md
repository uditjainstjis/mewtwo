# Synapta — Live Build Status Dashboard

**Last updated**: 2026-05-03 03:00 UTC (auto-loop iteration)

## TL;DR — where we are

| Phase | Status | Output |
|---|---|---|
| A — Data discovery (HF, Kaggle, gov sources) | ✅ COMPLETE | 3 agents reported |
| B — Scraping (RBI 80 + SEBI 50 PDFs + 3 HF datasets) | ✅ COMPLETE | 130 PDFs, 115 MB |
| C — Text extraction (pdfplumber + pymupdf, mp Pool 8) | ✅ COMPLETE | 8.06M chars, 7,329 sections |
| D — Chunking + 3-tier QA construction | ✅ COMPLETE | 4,185 chunks → v2 3,195 train QA |
| E — Validator (10 quality checks per row) | ✅ COMPLETE | 91.7% v2 pass → 2,931 train clean / 664 eval clean |
| **F — BFSI LoRA training (Nemotron-30B)** | 🔄 **RUNNING** | step 7/174, 70s/step, ETA 3.4h |
| G — Held-out eval (300 stratified) | ⏳ QUEUED | Auto-starts after F via launcher |
| H — Synthesis (BFSI_ADAPTER_FINAL.md + pitch updates) | ⏳ QUEUED | Triggered after G |

## Active processes

| PID | Process | Status |
|---|---|---|
| 101401 | Auto-launcher (waits→trains→evals) | alive |
| (training subprocess) | 07_train_bfsi_extract.py | running, step 7/174 |

## Resources

| Resource | Used | Free | Limit |
|---|---|---|---|
| GPU memory | 30 GB | 2 GB | 32 GB |
| GPU utilization | 48% | — | — |
| Disk (mewtwo) | 708 GB | 1.1 TB | 1.8 TB |

## Hard finding (not fixable today)

`models/nemotron/modeling_nemotron_h.py:78` hardcodes `is_fast_path_available = False` — Mamba CUDA kernels are incompatible with 4-bit quantization shapes. Installing causal-conv1d + mamba-ssm did not help. **70s/step is the floor** on RTX 5090 with 4-bit Nemotron-30B. Will revisit if we move to bf16 (would need ≥60GB VRAM).

## Training config (active run, v2 data)

- Base: Nemotron-30B-Nano-A3B, 4-bit NF4
- LoRA: r=16, alpha=32, dropout=0.05, target_modules=[q,k,v,o,gate,up,down]_proj
- LR: 1e-4, cosine, warmup 0.05
- Batch: 1 × 16 grad accum (effective 16)
- Epochs: 1 (was 2, reduced to fit budget)
- MAX_LEN: 1024 (token analysis: 99th percentile = 849)
- Optimizer: paged_adamw_8bit
- Precision: bf16 + gradient_checkpointing
- max_grad_norm: 0.3
- Trainable params: 434.6M / 32B (1.36%)

## Dataset (v2, currently training)

- **Train**: 2,931 cleaned QA pairs (after 91.7% validator pass)
- **Eval**: 664 cleaned QA pairs (held-out, document-disjoint from training)
- Source: 80 RBI Master Directions + 50 SEBI Master Circulars
- Tier 2 (numeric regex): ~2,150 questions about Rs amounts, days, percentages, section refs
- Tier 3 (heading-based): ~1,000 paragraph-level extractive QA
- Eval split: 26 PDFs (20%) held out entirely — eval questions are from PDFs the model never saw during training

## Auxiliary results (already shipped)

- **HumanEval pass@5 base**: 19/20 = 95% on first 20 problems (early-killed to free GPU; partial)
- **Long-context RBI test**: base 100/100% / FG 100/100% — null result (both find answers when given context)
- **From prior Qwen experiments**: HumanEval +17.1pp / MBPP +20.1pp (McNemar p<0.001) — published in CTO pitch

## Documents in `docs/recent/`

| File | Status |
|---|---|
| `BFSI_ADAPTER_FINAL.md` | Stub, awaiting Phase G results |
| `BFSI_PIPELINE_NARRATIVE.md` | ✅ Complete (~430 words) |
| `YC_APPLICATION_DRAFT.md` | ✅ V3 with new methodology |
| `CTO_PITCH_60SEC_V2.md` | ✅ V3 with honest placeholders |
| `DECK_SLIDES_V3.md` | ✅ 15 slides, structured speaker notes + visuals |
| `STATUS_DASHBOARD.md` | ✅ This file |

## Pipeline scripts in `synapta_src/data_pipeline/`

| # | Script | Status |
|---|---|---|
| 01 | scrape_rbi_mds.py | ✅ Done (80/80 PDFs) |
| 01b | download_hf_datasets.py | ✅ Done (3 datasets) |
| 01c | scrape_sebi_circulars.py | ✅ Done (50/150 unique) |
| 01d | scrape_sebi_broader.py | ✅ Done (200 broader, all dups) |
| 02 | extract_text.py | ✅ Done (130 PDFs) |
| 03 | chunk_circulars.py | ✅ Done (4,185 chunks) |
| 04 | build_qa_pairs.py | ✅ Done (v1) |
| 04b | build_qa_pairs_v2.py | ✅ Done (v2 cleaner) |
| 05 | integrate_hf_data.py | ✅ Done |
| 06 | validate_qa.py | ✅ Done (98.45% v1, 91.7% v2) |
| 07 | train_bfsi_extract.py | 🔄 RUNNING (step 7/174) |
| 08 | eval_bfsi_extract.py | ⏳ Queued (cap 300 questions) |
| 09 | demo_bfsi.py | ✅ Done (5 hand-curated demo questions) |

## What's left

1. Wait for training to complete (~3.4h from start, currently at ~12 min in)
2. Auto-launcher runs eval (~1.25h on 300 stratified questions × 3 modes)
3. Populate `BFSI_ADAPTER_FINAL.md` with actual numbers
4. Update YC + CTO pitch with held-out F1 numbers (placeholders → real)
5. Final commit + ready for YC submission

## Risk register

- **Eval might overshoot 3h budget** (training 3.4h + eval 1.25h = 4.65h, but auto-launcher has 2h timeout on eval; partial results still useful with resume support)
- **Loss may not converge in 1 epoch** on 2931 examples (fallback: have 4373 v1 train_clean as backup)
- **Model may still produce some malformed answers** despite training (mitigated by held-out eval being mostly extractive)

## Decision log (chronological)

| Time UTC | Decision | Why |
|---|---|---|
| 02:25 | Killed contaminated 525-example training | Test-set contamination invalidated metrics |
| 02:34 | Restarted RBI scraper with Referer header | RBI returns HTML error pages without Referer |
| 02:43 | Killed HumanEval to free GPU | User mandate "utilize full GPU" for BFSI |
| 02:43 | Saved partial: HumanEval base 19/20=95% | Recoverable result before kill |
| 02:48 | OOM with MAX_LEN=2048 | Activations too large; lowered to 1024 (covers 99% of corpus) |
| 02:50 | Tried installing Mamba kernels | Advisor flagged "fast path not available" warning |
| 02:51 | Found 4-bit hardcodes is_fast_path_available=False | Architectural constraint, not fixable in 4-bit |
| 02:55 | Switched v1→v2 data + 1 epoch | Cleaner questions, better signal-per-example |
| 03:00 | Patched eval to support MAX_EVAL_QUESTIONS=300 stratified | Fits 1h budget |
