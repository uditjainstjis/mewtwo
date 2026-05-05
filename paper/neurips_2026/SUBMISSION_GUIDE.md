# NeurIPS 2026 — Submission guide for Synapta paper

**Paper file:** `synapta_neurips2026.tex` (~462 lines, ~4360 words, 7 tables, filled checklist, 18 references)
**Track:** Main Track
**Contribution Type:** Use-Inspired
**Author:** Udit Jain — `hello@uditjain.in` — Synapta, India
**Working title:** *Sovereign Regulatory NLP: Per-Customer LoRA Adapters with Document-Disjoint Paired Evaluation, Zero-Cost Compositional Routing, and Honest Out-of-Distribution Disclosure*

---

## Critical deadlines (today is 2026-05-04)
- **Abstract submission: May 4, 2026 AOE** (≈ May 5 12:00 UTC; ~hours away)
- **Full paper + supplementary: May 6, 2026 AOE**
- **All authors must have OpenReview profile before submission**

---

## What's in the paper (and where the numbers came from)

### Core paired-test results

| Result | Source artifact |
|---|---|
| n=664 paired McNemar p=1.66×10⁻⁴⁸ (bfsi_extract vs base) | `results/bfsi_eval/summary.json` |
| n=664 paired FG vs adapter: −0.9 pp, p=0.031 | `results/bfsi_eval/summary.json` |
| n=214 paired Wilcoxon p=1.50×10⁻¹⁶ (bfsi_recall) | `results/bfsi_recall_eval/eval_results.jsonl` |
| n=324 IndiaFinBench: 32.1% [27.3, 37.4] | `results/indiafinbench_eval/summary.json` |
| n=60 Benchmark v1: 50.0% adapter vs 40.0% base, p=0.031 | `results/benchmark_v1_eval/summary.json` |
| n=60 Benchmark v1 FG: 50.0% (identical to adapter, p=1.0) | `results/benchmark_v1_eval/summary.json` |

### Released alongside the paper
- 130 source PDFs with SHA-256 manifests (`data/rbi_corpus/`, `data/sebi_corpus/`)
- Pipeline scripts `01_*..18_*` in `synapta_src/data_pipeline/`
- LoRA adapters `bfsi_extract` and `bfsi_recall` (1.74 GB each)
- Synapta Indian BFSI Benchmark v1 (60 questions, CC-BY-SA-4.0) at `data/benchmark/synapta_indian_bfsi_v1/`

---

## How to compile

The repo currently has no local pdflatex. Three options:

### Option A — Overleaf (recommended)
1. Open https://www.overleaf.com/project (free account works for ≤1 collaborator)
2. New Project → Upload Project → upload all 4 files in this directory:
   - `synapta_neurips2026.tex`
   - `neurips_2026.sty`
   - `neurips_2026.tex` (ignore — original template)
   - `checklist.tex` (ignore — already inlined)
3. Set main document to `synapta_neurips2026.tex`
4. Compile (will fetch any missing TeX packages automatically)

### Option B — Local Ubuntu
```bash
sudo apt update && sudo apt install -y texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra texlive-bibtex-extra
cd /home/learner/Desktop/mewtwo/paper/neurips_2026
pdflatex synapta_neurips2026.tex
pdflatex synapta_neurips2026.tex   # re-run for cross-refs
```

### Option C — Docker
```bash
docker run --rm -v $PWD:/work -w /work texlive/texlive pdflatex synapta_neurips2026.tex
```

---

## Pre-submission checks (do these before clicking submit)

1. **Compile and read PDF page-by-page.** Page count target: 9 pages main + appendices + checklist (checklist does NOT count toward page limit).
2. **Verify every number against canonical sources:**
   - Headline 89.6% → `results/bfsi_eval/summary.json` `bfsi_extract_only.substring_match_rate = 0.8961` ✓
   - McNemar p=1.66e-48 → `results/bfsi_eval/summary.json` `bfsi_extract_only_vs_base.p_value` ✓
   - 32.1% IndiaFinBench → `results/indiafinbench_eval/summary.json` `overall.substring_match.rate = 0.321` ✓
   - 50.0% Benchmark v1 → `results/benchmark_v1_eval/summary.json` `bfsi_extract.rate = 0.5` ✓
   - +38.4% recall F1 → recomputed from `results/bfsi_recall_eval/eval_results.jsonl` paired ✓
3. **OpenReview profile.** Create at https://openreview.net/signup if not yet — needs ~24 hrs to approve in some cases.
4. **Double-blind format.** The current `\usepackage{neurips_2026}` (no `final` flag) is double-blind by default — `\author{}` info is automatically anonymised.
5. **Submission portal.** Main Track has its own portal — see https://neurips.cc/Conferences/2026/CallForPapers
6. **Supplementary materials zip.** Bundle the 18 pipeline scripts + the 60-Q benchmark + 4 result JSONLs. Total <100 MB without adapter weights; weights link to a separate HuggingFace repo.

---

## Risks the reviewers will flag (and our paper's responses)

| Risk reviewer flags | Where the paper addresses it |
|---|---|
| Single base model | §10 Limitations explicitly states this |
| 60-Q benchmark too small | §10 + we are open about it being community-grown |
| 32.1% on IndiaFinBench is bad | §7 reframes as the central evidence for per-customer thesis |
| F1>=0.5 cut zero score | §6.3 Table 6 discloses metric-cutoff artifact |
| Frontier comparison is n=15 only | §10 acknowledges; §9 notes Claude wins F1 |
| Substring is wrong metric | §9 explicitly defends the choice + cites Claude F1 win |
| Nemotron-30B Mamba kernels disabled | Appendix B — explicit disclosure |
| Cross-benchmark comparisons not fair | §7 paragraph (3) explicitly addresses NeurIPS/ICLR norms |

---

## Strongest selling points for reviewers

1. **Two paired tests with two different metrics** showing the methodology generalises (Sec 6.1 + 6.2). Not one heroic number.
2. **Format Guard zero-cost replicated on independently constructed benchmarks** (n=664 + n=60, two different question construction processes). Not a single accidental zero.
3. **OOD honest disclosure as part of the contribution** — IndiaFinBench failure is reframed as evidence FOR the methodology, with a defensible argument grounded in §2 evaluation literature.
4. **Open release of the benchmark + the negative result** — most vendor work in this space publishes neither.
5. **Single consumer GPU ($2k), 3.5 hours training** — directly addresses the use-inspired evaluation criterion of impact for users outside ML.

---

## What to do RIGHT NOW (in priority order)

1. **Make OpenReview profile** if you don't have one. (10 min, blocker for submission.)
2. **Upload paper to Overleaf**, compile, sanity-read the PDF. (30 min.)
3. **Submit abstract** to Main Track via OpenReview link in the CFP. The abstract submission carries minimal metadata + the abstract text — paper PDF can come at the May 6 deadline.
4. **Pre-print to arXiv** is allowed and encouraged (use `\usepackage[preprint]{neurips_2026}`).

If you want me to: (a) tighten any section, (b) add a figure (architecture diagram of Format Guard, or contingency-table plot, or per-task-type bar chart on IndiaFinBench), (c) draft a separate companion benchmark paper for the Evaluations & Datasets track, (d) prepare the supplementary materials zip — say which.
