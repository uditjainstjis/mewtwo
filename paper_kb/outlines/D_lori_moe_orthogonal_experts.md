# Cluster D — LoRI-MoE: Token-Routed Orthogonal Experts (Updated 2026-05-05)

## Title candidates
- "Token-Routed LoRI-MoE: When Sparse Orthogonal Experts Help via Mid-Sequence Adapter Switching, and When They Do Not"
- "Recovering Catastrophically-Routed Composition: Token-Level Format Guard Lifts LoRI-MoE GSM8K from 4% to 65.5%"
- "Token-Level Routing Is Necessary But Not Sufficient: A Three-Benchmark Study of LoRI-MoE on Qwen-2.5-1.5B"

## One-paragraph problem statement
A natural response to the failure of weight-merging multi-adapter composition is to make the adapters approximately orthogonal by construction. LoRI-MoE freezes a shared random projection $B$ and trains only sparse domain-specific factors $A_e$, with the goal that the per-expert updates $\Delta W_e = B A_e$ are nearly orthogonal in cosine. We trained 5 LoRI-MoE experts (code, legal, math, medical, science) on Qwen-2.5-1.5B and measured an off-diagonal mean $|$cosine$|$ of 0.00685, confirming the structural property. Earlier prompt-level top-1 composite routing catastrophically misrouted on GSM8K (4\% vs single-best 53\%). **We now report the token-level routed LoRI-MoE end-to-end evaluation:** at $n=200$ per benchmark on Qwen-2.5-1.5B with a Format-Guard-style logits processor swapping the active expert every 10 generated tokens, we achieve GSM8K **65.5\% (+61.5 pp over prior composite, matching dedicated math adapter)**, ARC 45.0\% (slightly above single-best science adapter at 42.5\%, but $-30$ pp below the strong base of 75.5\%), and MMLU 43.5\% ($+1$ pp over base, matching single-best). The paper's contribution is two-fold: (1) token-level routing recovers single-best performance when an adapter is genuinely helpful (GSM8K), at near-zero overhead (mean 0.12 swaps/question); (2) Format Guard's failure mode is fully exposed: when no adapter helps, the router cannot revert to base, motivating an extension with a "no-adapter" routing option.

## Key evidence
- `experiments/09_lori_moe_phases.md` — prior Phase 1/2/3 prompt-level evaluation (base, single, composite).
- `experiments/10_lori_orthogonality.md` — structural orthogonality (avg $|$cosine$|$ off-diagonal $= 0.00685$).
- **NEW** `synapta_src/data_pipeline/19_eval_lori_moe_token_routed.py` + `results/lori_moe/phase3_token_routed.json` — the token-routed end-to-end eval. **The headline result.**
- `experiments/06_router_sft_dpo_5000.md` (cluster A) — adjacent router-training infrastructure.
- `experiments/13_format_guard_humaneval.md` (cluster B) — Format Guard mechanism this work reuses on Qwen-1.5B with 5 LoRI experts.

## Solid (Cat 1+2)
- 5 LoRI-MoE experts trained successfully on Qwen-2.5-1.5B.
- Phase 1 base / Phase 2 single-adapter / Phase 3 composite numerical results from local JSONs.
- **Token-routed end-to-end ($n=200 \times 3$ benchmarks):** GSM8K 65.5\% (+13 pp vs base, matches single-best, +61.5 pp vs prior composite); ARC 45.0\% (above single-best, below base); MMLU 43.5\% (matches single-best, +1 pp vs base).
- Mean adapter swaps per question 0.10–0.18: FG zero-overhead property replicates on a fourth independent paired evaluation (after HumanEval $n=164$, BFSI $n=664$, Benchmark v1 $n=60$).
- Off-diagonal cosine 0.00685 structural property (cited from saved-weight inspection).

## Aspirational (Cat 3)
- "LoRI-MoE outperforms a strong base across all reasoning benchmarks" — explicitly disclaimed: the ARC base of 75.5\% beats every adapter mode tested.
- "Orthogonal experts produce non-interfering composition" — orthogonality is structural; the GSM8K result is consistent with this; ARC honest disclosure shows the adapter can degrade base even with orthogonality.

## What NOT to claim
- "Format Guard always helps" — the ARC result is a $-30$ pp regression vs base. Disclose explicitly.
- "Orthogonal experts compose better than non-orthogonal" — we have not run a non-orthogonal head-to-head comparison; this is implied by Cluster B's static-merge failure but not directly tested for LoRI-MoE.
- Any "LoRI-MoE beats Mistral" or similar cross-system claim — not tested.

## Recommended paper framing
**Cluster D is NOW publication-ready as a mixed-results paper.** Three viable paths:

1. **Methodology paper (recommended):** "Token-Routed LoRI-MoE: When Sparse Orthogonal Experts Help via Mid-Sequence Adapter Switching, and When They Do Not." Lead with the GSM8K +61.5 pp result vs prior composite (the recovery story), use ARC as the limitation that motivates a "no-adapter" router extension, MMLU as the marginal positive. NeurIPS Main Track, General contribution type.

2. **Negative-results focus:** "Structural Orthogonality is Necessary but Not Sufficient: Token-Routed LoRI-MoE Cannot Recover When No Single Expert Helps." NeurIPS Negative Results track. Bar is high (the finding must be "surprising or unexpected"); the GSM8K positive softens this story but ARC is the load-bearing negative.

3. **Combined into Cluster B's Format Guard paper as a chapter.** The paradox-aware router applied to LoRI experts is structurally similar to the Format Guard on Nemotron-30B; both demonstrate the mid-sequence swap mechanism. This is the simplest publication path if writing one paper instead of two.

For a downstream paper-writing model: the path-1 methodology paper is the most defensible single-paper framing. The path-3 chapter is the cheapest to merge into an existing draft.
