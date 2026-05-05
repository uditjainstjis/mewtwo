# Synapta — Research Timeline

## 2026-04 (foundation)
- Set up Nemotron-Nano-30B-A3B 4-bit base, found Mamba CUDA fast-path incompatible with 4-bit weights → fallback to naive Mamba (70 s/step floor).
- Trained four task-specialised LoRA adapters: math, code, science, BFSI on 30B base. Targets all attn+MLP projections, $r=16$, $\alpha=32$.
- Trained 67 small-base adapters (Nemotron-Mini-4B + Qwen-3.5-0.8B × ranks {1, 2, 8, 128, 1024, 3072} × {SFT, DPO, DARE-merged}) for the rank-scaling ablation.

## 2026-04-12 (LoRI MoE composite routing)
- Phase 1 baselines: GSM8K 26\%, ARC 76.5\%, MMLU 56.5\%.
- Phase 2 single adapters: math GSM8K 53\%, science ARC 65\%, legal ARC 77.5\%.
- Phase 3 composite routing: GSM8K 4\% (regressed), ARC 72\%, MMLU 53\%.
- Verdict: composite top-1 routing did not exceed single best.

## 2026-04-22 to 2026-04-30 (rank scaling + Code Paradox replication)
- Code Paradox Phase 1 result on Nemotron-30B: code adapter scores 31\% ARC (+11 pp) and 56\% MATH-500 (+14.5 pp), but only 27\% HumanEval (-23 pp). Math adapter scores 60\% HumanEval (+10 pp).
- Static merge (DARE/TIES) on the same 30B: matches best single, never exceeds. ARC drops to 19\% (below base).
- Replicated code-on-code regression at $n=200$ on Qwen-3.5-0.8B (base 15\%, code 12\%, math 16\%).
- Rolled back earlier $n=50$ cross-family overclaim → only $n=200$ Nemotron-30B replication is treated as robust.

## 2026-05-01 (token-level routing breakthrough)
- Implemented Format Guard as a HuggingFace LogitsProcessor swapping every $K=10$ tokens.
- All 4 adapters pre-loaded into VRAM; \texttt{set\_adapter()} is $O(1)$ pointer flip.
- Cold-swap profiled at 315.9 ms on NVMe SSD over 44 swaps.
- Smoke tests: 95\% pass on 20-prompt mixed domain across all 4 modes after demo bug fixes.

## 2026-05-02 (HumanEval scoring-bug discovery)
- Discovered two scoring bugs in v1 HumanEval extraction (import-stripping, indent-stripping).
- Rescored both modes from saved JSONL outputs.
- v2 corrected: base 56.1\% / Format Guard 73.2\% / **+17.1 pp McNemar $\chi^2 = 15.68$, $p < 0.001$** on $n=164$.
- Per-category: math\_arith $+33$ pp, strings $+15$ pp, list\_ops $-7$ pp.

## 2026-05-03 (BFSI corpus + extract adapter)
- Scraped 80 RBI Master Directions + 50 SEBI Master Circulars (130 PDFs, 8.06M chars).
- Smart-chunked on numbered section boundaries (4{,}185 chunks, median 384 tokens).
- 3-tier deterministic regex Q\&A construction: Tier 1 native FAQ, Tier 2 numeric, Tier 3 heading-extractive. 4{,}477 candidate pairs, 98.45\% pass 10-check validator.
- Document-disjoint train/eval split: 26 PDFs (20\%) entirely held out.
- Trained bfsi\_extract LoRA on 2931 train pairs, 1 epoch, 3h~28min on RTX 5090.

## 2026-05-03 (BFSI extract held-out eval)
- $n=664$ paired McNemar substring test:
  - Base 58.7\% → adapter 89.6\% → Format Guard 88.7\%.
  - Adapter vs base: $b_{10}=14$, $b_{01}=219$, $p=1.66\times10^{-48}$.
  - FG vs adapter direct: $b_{10}=6$, $b_{01}=0$, $-0.9$ pp, $p=0.031$.
- Per-tier: Tier 2 numeric $+24.6$ pp, Tier 3 heading $+39.5$ pp.
- Per-regulator: RBI $+31.5$ pp, SEBI $+30.1$ pp.

## 2026-05-04 (recall adapter + IndiaFinBench + Benchmark v1)
- Trained second adapter (bfsi\_recall) on no-context recall task. Same recipe.
- $n=214$ paired Wilcoxon F1: base 0.158 → adapter 0.219, $+38.4\%$ relative, $p=1.50\times10^{-16}$.
- 74.3\% of questions show adapter F1 > base F1.
- Discovered IndiaFinBench~\citep{pall2026indiafinbench} via perplexity research; ran our adapter against it: 32.1\% [27.3, 37.4] on $n=324$ vs Gemini Flash zero-shot 89.7\%.
- Curated 60-question Synapta Indian BFSI Benchmark v1 from held-out documents; released CC-BY-SA-4.0.
- Paired eval on Benchmark v1: base 40\% → adapter 50\% → Format Guard 50\%, McNemar adapter-vs-base $p=0.031$. Format Guard zero-overhead replicated.

## 2026-05-04 (frontier comparison via subagents)
- 15-question hand-curated subset run via Anthropic Claude Opus + Sonnet through subagent harness.
- Synapta substring 87\%, Claude Opus 7\%, Claude Sonnet 27\%.
- Token F1 reverses: Claude Opus 0.65, Synapta 0.38.
- Reading: Synapta is configured for citation-faithful production output; Claude is configured for semantic polish. Different deliverables.

## 2026-05-04 (NeurIPS submission prep)
- Abstract submission deadline May 4 AOE; full paper May 6 AOE.
- Paper drafted at `/paper/neurips_2026/synapta_neurips2026.tex` leading with Code Paradox + Format Guard, BFSI as case-study chapter.
