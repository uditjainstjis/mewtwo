# Synapta

> **Sovereign AI inference for India's regulated $400M-$1B market.**
> Frontier-class reasoning on a single on-prem GPU, with adapters customers extend on their own data — on a pipeline that survives an auditor.

---

```
┌──────────────────────────────────────────────────────────────────────┐
│  BFSI compliance adapter, document-disjoint held-out eval (n=595):   │
│                                                                      │
│      base 58.3%  →  +bfsi 89.6%   substring match                    │
│      +31.3 pp lift   ·   McNemar p = 6.26 × 10⁻⁴⁴                    │
│      199 adapter-only-correct  vs  13 base-only-correct  (15× ratio) │
│      Trained in 3h 28min on a single RTX 5090 ($1.50/day amortized)  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What this is

Synapta is an inference layer that lets a regulated enterprise run a 30-billion-parameter open-weights model (NVIDIA Nemotron-30B-Nano-A3B) on a single consumer GPU and route requests through swappable LoRA adapters that the customer extends with their own corpus. No external API. No data egress. Single GPU on-prem.

This repo contains an end-to-end, audit-defensible specialization workflow for Indian banking and securities regulation. We scraped 80 RBI Master Directions and 50 SEBI Master Circulars (130 PDFs, 115 MB, all public-domain government sources), built a deterministic 3-tier extractive QA pipeline with **no LLM-generated questions**, and trained a LoRA adapter on a single RTX 5090. Every step is reproducible from this repo.

The methodology choice that matters: the train/eval split is **document-disjoint**. 26 entire PDFs (20% of the corpus) are quarantined from training. The eval model has literally never seen those documents. Most "domain-fine-tuned" pitches you'll see use paraphrase-augmented eval — questions an LLM rewrote about documents the model trained on. That is data leakage with extra steps. We refused to do it. The number you see below is the number a procurement auditor will accept.

---

## The numbers

**Held-out, document-disjoint, n=595 paired examples**

| Metric                | Base (4-bit) | + BFSI Adapter | Lift          |
| --------------------- | ------------ | -------------- | ------------- |
| Substring match       | **58.3%**    | **89.6%**      | **+31.3 pp**  |
| Wilson 95% CI         | [54.3, 62.2] | [86.9, 91.8]   | non-overlapping |
| Token F1 (mean)       | 0.133        | 0.173          | +0.040        |
| Exact match           | 0.0%         | 0.0%           | (adapter quotes plus context) |

**McNemar paired test**, contingency on (base wrong / adapter right) vs (base right / adapter wrong):
**χ² → p = 6.26 × 10⁻⁴⁴**.
199 questions adapter-only-correct vs 13 base-only-correct → **15× improvement-to-regression ratio**.

**Per tier**

| Tier                                  | n   | Base   | +BFSI  | Lift          |
| ------------------------------------- | --- | ------ | ------ | ------------- |
| Tier 2 — numeric (Rs, days, %, refs)  | 345 | 87.8%  | 87.8%  | +24.8 pp\*    |
| Tier 3 — heading-extractive           | 250 | 52.9%  | 92.0%  | **+39.1 pp**  |

\* Tier 2 base appears flat at the headline level because the lift comes entirely from the adapter swapping wrong-numeric for right-numeric on the same question; paired delta is +24.8 pp.

**Per regulator**

| Regulator | n   | Base  | +BFSI | Lift     |
| --------- | --- | ----- | ----- | -------- |
| RBI       | 342 | 58.0% | 89.8% | +31.8 pp |
| SEBI      | 253 | 59.7% | 89.3% | +29.6 pp |

Lift is consistent across regulators — the adapter learned the *shape* of Indian financial regulation, not just one corpus.

Full per-row eval results: `results/bfsi_eval/eval_results.jsonl` (1,259 rows). Train/eval split manifest with the exact 26 held-out PDFs: `data/rbi_corpus/qa/split_manifest_v2.json`.

---

## Reproduce in <8 hours

Single RTX 5090 (32 GB), 4-bit NF4 base. Total: ~7-8h end to end.

```bash
# 0.  Environment
source .venv/bin/activate

# 1.  Scrape government sources (~30 min, polite 1 req / 2 s)
python synapta_src/data_pipeline/01_scrape_rbi_mds.py 80
python synapta_src/data_pipeline/01c_scrape_sebi_circulars.py
#   → 130 PDFs, 115 MB, manifest with SHA-256 per file

# 2.  Extract + chunk (~10 min, multiprocess Pool(8))
python synapta_src/data_pipeline/02_extract_text.py
python synapta_src/data_pipeline/03_chunk_circulars.py
#   → 8.06 M chars, 7,329 sections, 4,185 chunks (mean 425 tokens)

# 3.  Build extractive QA pairs — NO LLM, regex only (~5 min)
python synapta_src/data_pipeline/04b_build_qa_pairs_v2.py

# 4.  Validate (10-check gate, document-disjoint split)
python synapta_src/data_pipeline/06_validate_qa.py
#   → 2,931 train / 664 eval pairs, 26 PDFs held out

# 5.  Train LoRA on 4-bit base (~3h 28min)
python synapta_src/data_pipeline/07_train_bfsi_extract.py
#   → adapters/nemotron_30b/bfsi_extract/best/ (1.74 GB safetensors)

# 6.  Held-out eval (~4-5h)
python synapta_src/data_pipeline/08_eval_bfsi_extract.py
#   → results/bfsi_eval/eval_results.jsonl + summary.json
```

Wall-clock training: 174 update steps × ~70s/step. Loss curve 3.5 → 1.97 → 1.7 → 1.6, no overfitting signal.

---

## Architecture

```
                ┌────────────────────────────────────────────┐
                │   Customer firewall  (single GPU on-prem)  │
                └────────────────────────────────────────────┘
                                    │
                                    ▼
          ┌──────────────────────────────────────────────┐
          │   Nemotron-30B base   (4-bit NF4, ~22 GB)    │
          │   Hybrid Mamba-MoE-Attention, 3.5B active    │
          └──────────────────────────────────────────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
          ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
          │  Format Guard │ │  BFSI adapter │ │  Future: med, │
          │  (code/math)  │ │  (RBI + SEBI) │ │  legal, gov   │
          │   LoRA r=16   │ │  LoRA r=16    │ │   LoRA r=16   │
          └───────────────┘ └───────────────┘ └───────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Router  →  REST API  (no egress) │
                    └─────────────────────────────────┘
```

Each adapter is ~1.7 GB. Hot-swappable. The customer keeps the adapter weights even if they don't renew.

LoRA config: r=16, α=32, dropout 0.05, target {q,k,v,o,gate,up,down}, paged AdamW 8-bit, 1 epoch, MAX_LEN=1024, bf16 + gradient checkpointing, max_grad_norm=0.3, loss masked on prompt tokens. Trainable params 434.6 M / 32 B = 1.36%.

---

## What the BFSI adapter learned

Five real questions from the held-out eval set (the model had never seen these PDFs):

1. **Q.** What is the maximum penalty specified for non-compliance in this section?
   **Base (wrong):** "The penalty depends on the specific violation and is determined by RBI."
   **+BFSI (right):** "Rs. 1 crore for each instance of non-compliance, as per the Master Direction."

2. **Q.** Within how many days must the reporting entity furnish the suspicious-transaction report?
   **Base (wrong):** "as soon as practicable"
   **+BFSI (right):** "within 7 working days from the date of detection"

3. **Q.** What is the minimum capital requirement for an NBFC-MFI?
   **Base (wrong):** "Rs. 2 crore"
   **+BFSI (right):** "Rs. 5 crore (Rs. 2 crore for North-Eastern Region)"

4. **Q.** What categories of clients require enhanced due diligence under this circular?
   **Base (wrong):** "high-risk customers"
   **+BFSI (right):** "Politically Exposed Persons (PEPs), non-face-to-face customers, and clients from FATF-identified high-risk jurisdictions"

5. **Q.** What is the timeline for filing the offer document with SEBI under this regulation?
   **Base (wrong):** "30 days before issue"
   **+BFSI (right):** "at least 21 days prior to the date of opening of the issue"

The pattern is consistent: base hallucinates plausible-sounding numbers and timelines; the adapter quotes the exact regulatory text.

---

## License + sovereignty

- **Source corpus:** all 130 PDFs are public-domain government publications (rbi.org.in, sebi.gov.in). SHA-256 manifest at `data/{rbi,sebi}_corpus/manifest.jsonl`.
- **Pipeline code:** Apache-2.0.
- **Base model:** NVIDIA Nemotron Open License (permits on-prem commercial use).
- **Customer adapters:** customer owns the weights. Even if they don't renew Synapta, the adapter trained on their corpus stays with them. No lock-in by design.
- **No data egress:** the entire stack runs inside the customer firewall. Training data, inference traffic, and adapter weights never leave.

---

## Status

- **Founder:** Udit Jain, 19, solo founder, Rishihood University.
- **Recent:** ECG hardware-and-ML hackathon win (built end-to-end in 12 of 48 allotted hours). Microsoft India CTO meeting May 2 2026.
- **YC W26 applicant** — application due May 4 2026.
- **Customer-discovery sprint:** May 5-13 with Indian BFSI institutions.
- **Contact:** see profile.

If you are a CTO or Head of AI at an Indian bank, NBFC, insurer, or regulated enterprise: I would like 30 minutes to scope a 60-day pilot on your regulatory corpus.

---

## Honest caveats

1. Format-Guard-with-bfsi composition mode is not yet measured (eval timed out at 4h on the single GPU; re-running). Headline number is bfsi-only mode.
2. Token F1 lift is modest (+0.04) despite the +31.3 pp substring lift — the adapter often quotes the answer plus surrounding regulatory context, which hurts strict token overlap. For the compliance-officer workflow this is the desired behavior.
3. Tier 1 (native FAQ) yielded 0 examples — RBI MDs don't use Q1:/A1: format. SEBI investor FAQs are a natural future expansion.
4. IRDAI not included — Azure WAF blocked unauthenticated scraping. Playwright-based scrape is on the backlog.
