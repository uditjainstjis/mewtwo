# Cluster B — Format Guard, Code Paradox, and Static Composition Failure (Nemotron-30B)

## Title candidates
- "The Code Paradox and Format Guard: Token-Level LoRA Adapter Routing as a Substitute for Failed Static Composition"
- "Why Static LoRA Merging Plateaus and Token-Level Routing Wins: Evidence from a 30B Base"
- "Format Guard: A Logits-Processor for Compositional LoRA Inference at Zero Accuracy Cost"

## One-paragraph problem statement
Multi-adapter composition is typically studied at the weight level (DARE, TIES, AdapterFusion, LoRAHub) or via mixture-of-experts routing. We report two empirical findings on Nemotron-Nano-30B-A3B (4-bit) trained with four task-specialised LoRA adapters (math, code, science, BFSI). First, the Code Paradox: training on Python code degrades HumanEval pass@1 by 23 pp but improves ARC-Challenge by 11 pp and MATH-500 by 14.5 pp. Training on math improves HumanEval by 10 pp. The cross-domain transfer is asymmetric and counter-intuitive. Second, static weight-merging of the four adapters never exceeds the best single expert and degrades on out-of-home tasks. We propose Format Guard, a HuggingFace LogitsProcessor that swaps the active adapter every 10 generated tokens via a regex-driven router. On HumanEval $n=164$ paired evaluation, Format Guard improves base 56.1\% to 73.2\% (+17.1 pp, McNemar $\chi^2 = 15.68$, $p < 0.001$). The mechanism's empirical zero-overhead property (FG matches the dedicated adapter when the right adapter is in the pool) is replicated on a separate domain at $n=664$ and $n=60$.

## Key evidence
- `experiments/11_phase1_single_adapter_30b.md` — Phase 1 single-adapter grid (the Code Paradox raw data).
- `experiments/12_static_composition_failure_30b.md` — DARE/TIES/uniform merging $\leq$ best-single across all 4 benchmarks, $+0.0$ on $n=45$ probe.
- `experiments/13_format_guard_humaneval.md` — **+17.1 pp HumanEval $p<10^{-3}$, $n=164$ paired McNemar (the headline)**.
- `experiments/14_cold_swap_latency.md` — 315.9 ms cold; $O(1)$ warm pointer flip.
- `experiments/15_code_paradox_replication.md` — in-domain regression robust at $n=200$ on Qwen-3.5-0.8B.
- `experiments/16_rank_scaling_zoo.md` — 67 adapters across 2 small bases; rank does not rescue code-on-code regression.
- `experiments/17_humaneval_scoring_bug.md` — methodology correction that moves the headline from "+48.7 pp" to "+17.1 pp" with full transparency.
- `experiments/19_bfsi_extract_eval_n664.md` (Cluster C) — FG zero-overhead replication, $b_{10}=6, b_{01}=0$.
- `experiments/22_benchmark_v1_n60.md` (Cluster C) — FG zero-overhead replication, mean 0.1 swaps/Q.

## Solid (Cat 1+2)
- HumanEval $n=164$ v2 corrected: base 56.1\%, FG 73.2\%, $+17.1$ pp McNemar $p<10^{-3}$.
- Static composition $\leq$ best-single across 4 benchmarks.
- Cold-swap 316 ms / warm $O(1)$.
- Code Paradox in-domain regression (code adapter on HumanEval) at $n=100$ Nemotron-30B and $n=200$ Qwen-3.5-0.8B.

## Aspirational (Cat 3)
- "Code Paradox replicates across base families" — the $n=200$ Qwen replication is in-domain regression only; the asymmetric positive transfer (code → math) is single-base.
- "Format Guard generalises across base models" — only Nemotron-Nano-30B-A3B has been benchmarked. See `missing_artifacts.md` item 6.
- "Format Guard is the right routing primitive in general" — strong claim; the heuristic router is deliberately simple and the MBPP $-3$ pp regression suggests format-rigid tasks need a debounce or learned router.

## What NOT to claim
- Any v1 HumanEval numbers anywhere ("+24 pp at $n=25$", "70.7\% FG v1").
- "Code Paradox cross-family at $n=50$" — rolled back.
- "Format Guard improves all benchmarks" — it regresses MBPP by 3 pp; explicit disclosure required.
- "Static merging is fundamentally bad" — only uniform-weight DARE/TIES tested; LoRAHub-learned weighted composition is unverified at 30B.

## Recommended paper framing
Strongest publication framing: **NeurIPS Main Track, General contribution type** (or Concept & Feasibility if we want to make the Format Guard mechanism the main idea). Title and abstract should lead with the Code Paradox as the surprising empirical finding, followed by static-composition failure as motivating evidence, followed by Format Guard as the working mechanism, followed by BFSI generalisation (cluster C) as a use-inspired chapter. This is the structure of the existing draft at `paper/neurips_2026/synapta_neurips2026.tex`.
