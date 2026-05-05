# Synapta Project — Document Index

**Last updated**: 2026-05-04 (during 8h autonomous loop)
**Purpose**: navigation hub. Open this first. Find what you need. Don't read everything.

---

## 🚨 If you only open ONE file: → `SUBMIT_CHECKLIST.md`

The 4 things only YOU can do for YC submission today (Loom, GitHub, paste app, click submit). Has copy-paste commands. **Open this first.**

---

## 📦 By purpose

### "I'm submitting YC right now"
1. **`SUBMIT_CHECKLIST.md`** — 4-step protocol with commands
2. **`YC_APPLICATION_DRAFT.md`** — paste content into YC form (V3, has methodology hook + competitive landscape + drift answer + F1 honesty)
3. **`DEMO_VOICEOVER_BFSI.md`** — 90-sec script for the Loom you'll record
4. **`TWEET_THREAD.md`** — 8-tweet thread to post tomorrow morning

### "I have a CTO meeting / cold-emailing customers"
1. **`CTO_ONE_PAGER.md`** — printable A4 brief, $50K pilot anchor
2. **`CTO_PITCH_60SEC_V2.md`** — verbatim 60-sec pitch + Q&A responses
3. **`BFSI_PIPELINE_NARRATIVE.md`** — 430-word story of the build
4. **`DECK_SLIDES_V3.md`** — 15-slide deck content (export to Google Slides)

### "I'm explaining the technical work to a researcher / investor"
1. **`BFSI_ADAPTER_FINAL.md`** — full results doc with McNemar p < 10⁻⁴³, per-tier, per-regulator
2. **`FRONTIER_COMPARISON_FINDINGS.md`** — Synapta vs Claude on 15 questions, F1 honesty
3. **`AUTO_RETRAIN_PIPELINE_DESIGN.md`** — drift-resistance architecture (5 components)
4. **`COMPETITIVE_LANDSCAPE.md`** — OnFinance + Big-4 + KYC players, methodology empty space

### "I want to understand the business model"
1. **`BUSINESS_MODEL_AND_BFSI_BUILD.md`** — pricing, customer types, revenue projection, the BFSI build story

### "I want strategic context"
1. **`STATUS_DASHBOARD.md`** — phase status (snapshot, may be outdated by hours)
2. **`PRIORITY_QUEUE.md`** — what's queued
3. **`STATUS.md`** — short status

---

## 📜 Stale / superseded (do not use)

These were earlier drafts; the V3 / FINAL versions above superseded them:
- `DECK_UPDATE_GUIDE.md`, `DECK_UPDATE_GUIDE_V2.md` — superseded by `DECK_SLIDES_V3.md`
- `DEMO_VOICEOVER.md` — superseded by `DEMO_VOICEOVER_BFSI.md`
- `STATUS.md` — superseded by `STATUS_DASHBOARD.md`
- `WAKE_UP_8H.md`, `WAKE_UP_README.md`, `LOOP_ITERATIONS.md`, `LOOP_FINAL.md`, `RESULTS_8H_SWARM.md`, `CONTEXT_HANDOFF.md`, `FINAL_SUMMARY.md` — historical loop logs from earlier sessions
- `BFSI_OUTREACH_TEMPLATES.md` — exists but skip outreach until after submission per user's decision
- `DECK_FOR_YC_ALUM.md`, `RESEARCH_REPORT_FOR_ALUM.md` — early drafts the user marked as "very poor crafts" — DO NOT SEND. The replacement direction was a single short email; not used.
- `TALKING_POINTS.md` — superseded by CTO_PITCH_60SEC_V2.md

---

## 🔑 Key headline numbers (memorize these)

These appear consistently across all live documents. Verify before quoting:

| Claim | Number | Source |
|---|---|---|
| BFSI adapter substring lift | **+30.9 pp** | `BFSI_ADAPTER_FINAL.md` |
| Statistical significance | **McNemar p = 1.66 × 10⁻⁴⁸** | `BFSI_ADAPTER_FINAL.md` |
| Held-out sample | n=664 paired (document-disjoint, all 3 modes) | `BFSI_ADAPTER_FINAL.md` |
| Format Guard vs direct (n=664) | -0.9 pp, p≈0.03 (routing overhead ~0%) | `BFSI_ADAPTER_FINAL.md` |
| Synapta on IndiaFinBench (OOD, n=324) | 32.1% [27.3, 37.4] vs Gemini Flash 89.7% | `perplexity_research/indiafinbench_finding.md` |
| Synapta on Benchmark v1 (n=60, paired) | 50.0% adapter vs 40.0% base, +10pp, p=0.031 | `BFSI_ADAPTER_FINAL.md` |
| Per-tier T2/T3 lifts | +24.8 pp / +39.1 pp | `BFSI_ADAPTER_FINAL.md` |
| Per-regulator RBI/SEBI | +31.8 pp / +29.6 pp | `BFSI_ADAPTER_FINAL.md` |
| Training cost | $1.50/day GPU, 3h 28min | `BFSI_ADAPTER_FINAL.md` |
| HumanEval (prior result) | +17.1 pp McNemar p<0.001 | `CTO_PITCH_60SEC_V2.md` |
| Synapta vs Claude (n=15) | 87% vs 7-27% substring | `FRONTIER_COMPARISON_FINDINGS.md` |
| Synapta vs Claude F1 | 0.38 vs 0.65 (Claude wins) | `FRONTIER_COMPARISON_FINDINGS.md` |
| Direct competitor | OnFinance AI ($4.2M Sep 2025, Peak XV) | `COMPETITIVE_LANDSCAPE.md` |
| Adapter file size | 1.74 GB safetensors | `BFSI_ADAPTER_FINAL.md` |
| Trainable params | 434.6M / 32B (1.36%) | `BFSI_ADAPTER_FINAL.md` |
| Source corpus | 130 PDFs (80 RBI + 50 SEBI), 115 MB | `BFSI_ADAPTER_FINAL.md` |

---

## 🗂 Code locations (for github push and reproducibility claims)

```
synapta_src/data_pipeline/
├── 01_scrape_rbi_mds.py              # Async scrape RBI MDs
├── 01b_download_hf_datasets.py        # Download HF supplement datasets
├── 01c_scrape_sebi_circulars.py       # Async scrape SEBI MCs
├── 01d_scrape_sebi_broader.py         # SEBI broader circulars (200 cap)
├── 02_extract_text.py                 # PDF → text via pdfplumber+pymupdf
├── 03_chunk_circulars.py              # Smart-chunk on numbered sections
├── 04_build_qa_pairs.py               # 3-tier QA (v1)
├── 04b_build_qa_pairs_v2.py           # 3-tier QA (v2 cleaner)
├── 05_integrate_hf_data.py            # Merge HF data
├── 06_validate_qa.py                  # 10-check validator
├── 07_train_bfsi_extract.py           # LoRA training (the headline result)
├── 08_eval_bfsi_extract.py            # Held-out eval (resumable)
├── 09_demo_bfsi.py                    # Earlier scripted CLI demo
├── 10_build_recall_dataset.py         # Build no-context QA dataset
├── 11_frontier_comparison.py          # Frontier API comparison harness
├── 12_train_bfsi_recall.py            # Train recall adapter
├── 13_eval_bfsi_recall.py             # Eval recall adapter
├── run_bfsi_train_then_eval.sh        # Auto-launcher chain
└── run_recall_then_evals.sh           # Recall + FG auto-chain

synapta_src/demo/
└── synapta_live_demo.py               # Gradio LIVE DEMO (the wine glass on tank)

data/
├── rbi_corpus/                        # 80 RBI MD PDFs + manifests
├── sebi_corpus/                       # 50 SEBI MC PDFs + manifests
├── hf_bfsi/                           # HF auxiliary datasets (mostly unused in final)
└── benchmark/synapta_indian_bfsi_v1/  # (in flight, may not be done yet)

adapters/nemotron_30b/
├── bfsi_extract/best/                 # The trained adapter (1.74 GB)
└── bfsi_recall/                       # In progress as of writing this index

results/
├── bfsi_eval/eval_results.jsonl       # Headline +30.9pp result raw rows (n=664, 3 modes)
├── bfsi_recall_eval/                  # Will exist after recall eval runs
└── frontier_comparison/               # 15-question Synapta vs Claude raw rows
```

---

## ⏰ What's running right now (process state — may be stale by minutes)

These were live as of writing this index. Check `ps aux | grep python` for current truth:
- `bfsi_recall` LoRA training (PID 208027) — finishing within ~50 min
- Auto-chain orchestrator (PID 210018) — will run recall eval + FG eval after training
- Background subagents may be in flight (check `/tmp/claude-1002/.../tasks/`)

---

## 🚦 If something is unclear

If the navigation here doesn't match what you find on disk:
- This index is current as of when it was written. Files added later won't be listed.
- Trust file mtime > this index for "what's newest"
- The HEADLINE NUMBERS table above has not changed and will not change without explicit re-measurement
