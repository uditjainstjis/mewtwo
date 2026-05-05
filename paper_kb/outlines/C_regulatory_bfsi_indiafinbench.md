# Cluster C — Sovereign Regulatory NLP (BFSI Pipeline + IndiaFinBench OOD + Benchmark v1)

## Title candidates
- "Sovereign Regulatory NLP: Per-Customer LoRA Adapters with Document-Disjoint Paired Evaluation"
- "Per-Customer Adapters Beat General Domain LLMs: Evidence from RBI/SEBI Compliance with Honest Out-of-Distribution Disclosure"
- "Indian BFSI Compliance NLP: A Reproducible Methodology and an Open 60-Question Benchmark"

## One-paragraph problem statement
Sovereign deployment of regulatory AI in India faces two structural constraints: data localisation laws restrict cloud APIs, and customers' compliance documents differ in style and structure from any single general LLM's training distribution. We report a complete pipeline from public Reserve Bank of India (RBI) and Securities and Exchange Board of India (SEBI) document scraping through 4-bit LoRA training to three-mode held-out paired evaluation. To eliminate paraphrase contamination we construct training data deterministically: a 3-tier regex template extracts question--answer pairs from real regulator text, with a 10-check validator gating every candidate pair. Crucially, our train/eval split is document-disjoint: 26 entire PDFs (20\%) are quarantined from training. On 664 paired held-out questions a single LoRA adapter (rank 16, 1.36\% trainable parameters, trained in 3h 28min on a single consumer GPU) lifts substring match from 58.7\% to 89.6\% (+30.9 pp; McNemar $p = 1.66 \times 10^{-48}$). The methodology generalises to a no-context recall task with $+38.4\%$ relative F1 gain (Wilcoxon $p = 1.50 \times 10^{-16}$). On the contemporary public benchmark IndiaFinBench~\citep{pall2026indiafinbench} our adapter scores only 32.1\% [Wilson 27.3, 37.4] vs Gemini Flash's 89.7\%---a 57.6 pp out-of-distribution gap that we frame as the central evidence FOR per-customer training, not against the methodology. We open-source a 60-question hand-curated benchmark (CC-BY-SA-4.0) on which the adapter scores 50\% vs base 40\%.

## Key evidence
- `experiments/18_bfsi_pipeline.md` — corpus + 3-tier deterministic QA + 10-check validator + document-disjoint split.
- `experiments/19_bfsi_extract_eval_n664.md` — **the headline: $+30.9$ pp McNemar $p = 1.66 \times 10^{-48}$**.
- `experiments/20_bfsi_recall_eval_n214.md` — generalisation: $+38.4\%$ relative F1, Wilcoxon $p = 1.50 \times 10^{-16}$.
- `experiments/21_indiafinbench_ood_n324.md` — OOD probe: 32.1\% vs Gemini Flash 89.7\%.
- `experiments/22_benchmark_v1_n60.md` — released benchmark with paired baselines.
- `experiments/23_frontier_comparison_n15.md` — directional Synapta vs Claude (substring win, F1 loss).
- `experiments/13_format_guard_humaneval.md` — Format Guard mechanism this cluster reuses.

## Solid (Cat 1+2)
- $n=664$ paired McNemar; per-tier and per-regulator breakdowns.
- $n=214$ paired Wilcoxon F1.
- IndiaFinBench OOD with full per-task breakdown.
- Benchmark v1 release with paired McNemar.
- Format Guard zero-overhead replication on three independent paired evaluations.

## Aspirational (Cat 3)
- "Synapta is comparable to Gemini Flash" — the cross-benchmark numbers are NOT comparable. The OOD result on IndiaFinBench is the apples-to-apples number.
- "Multi-regulator coverage including IRDAI / PFRDA / FEMA" — not built. See `missing_artifacts.md` item 8.
- "Frontier API comparison at scale" — only $n=15$ subagent comparison. See `missing_artifacts.md` item 5.

## What NOT to claim
- "Synapta 89.6\% ≈ Gemini Flash 89.7\%" — different benchmarks, different scoring; killed in `13_indiafinbench_ood.md`.
- Any frontier comparison with $n>15$ samples (we only have $n=15$).
- IRDAI / PFRDA / FEMA capabilities (we have RBI + SEBI only).
- "Beats OnFinance" / "beats incumbents" — no head-to-head benchmark exists.

## Recommended paper framing
**NeurIPS Main Track, Use-Inspired contribution type** (best fit for the Use-Inspired evaluation criteria). Could also be slid into the **Evaluations & Datasets track** focused on the Benchmark v1 release, but the n=60 size is small for that track. Honest framing is: per-customer methodology, with the OOD result reframed as evidence for the per-customer thesis. The existing paper draft at `paper/neurips_2026/synapta_neurips2026.tex` already implements this framing; cluster C can be either the main story (Use-Inspired track) or a chapter inside cluster B's Format Guard paper (current draft).
