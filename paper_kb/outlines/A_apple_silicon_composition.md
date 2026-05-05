# Cluster A — Apple Silicon Prompt-Level Composition (Synapta v1/v2/Clamp/Routing/TCAR)

## Title candidates
- "Bounded Multi-Adapter Composition on Apple Silicon: Negative-to-Mixed Results from a 1.5B Base"
- "When Composition Fails: Bounded Adapter Mixing on Small Bases is Limited by Geometry, Not Routers"
- "TCAR: Approaching a 7B Baseline with a 1.5B Apple Silicon Stack via Collaborative Inference"

## One-paragraph problem statement
Multi-adapter composition on small base models is a natural deployment story for Apple Silicon (no cloud, low memory, customer-owned hardware). This cluster reports a multi-phase study on Qwen2.5-1.5B-Instruct-4bit MLX with 20 LoRA experts: (i) prompt-level bounded composition with weight-cap and norm-ratio clamps, (ii) routing-gap analysis comparing oracle vs real CoT routing, (iii) replacement of generative CoT routing by SFT/DPO classifier routers, (iv) a TCAR collaborative-inference loop that approaches Mistral-7B quality at 1/4 the VRAM. Across three pre-registered hypothesis sets, the central headline ("$\Delta_\text{SIM} > +0.05$ from composition") fails; the result is a structured negative result enriched by a router-quality study and the TCAR system.

## Key evidence
- `experiments/01_synapta_v1_prompt_composition.md` — H1 FAIL ($\Delta_\text{SIM} = -0.011$, $n=100$).
- `experiments/02_synapta_v2_multidomain.md` — H2 sub-threshold positive ($+0.0171$ vs $+0.03$ pre-registered, $n=40$ MD).
- `experiments/03_clamp_ablation_norm_ratio.md` — norm-ratio $\equiv$ weight-cap ($\Delta = -0.0003$, $n=40$); H5 failure is a model property.
- `experiments/04_routing_gap_oracle_vs_real.md` — oracle headroom $+0.0206$, real router realises $+0.0054$ ($\sim 26\%$ of headroom).
- `experiments/05_external_md_blind_judge.md` — external blind judge does NOT support "Synapta beats Mistral".
- `experiments/06_router_sft_dpo_5000.md` — SFT 85\% routing accuracy, DPO regressed to 42\%.
- `experiments/07_tcar_collaborative.md` — TCAR matches Mistral on similarity ($-0.0007$), loses on F1 ($-0.0205$), at $\sim$1/4 VRAM and $\sim 2.3\times$ latency.
- `experiments/08_injection_9technique_ablation.md` — internal vs external benchmark rankings disagree.

## Solid (Cat 1+2)
- All v1, v2, clamp ablation, routing gap, 9-technique ablation aggregate numbers.
- SFT router 85\% / DPO router 42\% on the same holdout.
- TCAR final $n=100$ similarity tie with Mistral.
- External blind judge 30-item comparisons.

## Aspirational (Cat 3)
- "Synapta beats Mistral on multi-domain" — NOT supported on F1 or blind judge.
- "Compositional gain on Apple Silicon" — sub-threshold positive at best.
- Apple Silicon end-to-end reproducibility claim — gated on 20-expert safetensor restoration (`missing_artifacts.md` item 2).

## What NOT to claim
- "Bounded composition wins" — the v2 result is sub-threshold, not significant.
- "Norm-ratio clamp is essential" — empirically identical to weight-cap on this configuration.
- "Synapta beats Mistral" — external blind judge directly disagrees.
- Any "$+5.7\%$ similarity gain" / "75\% less VRAM than Mistral" without immediately citing the F1 loss and the latency penalty.

## Recommended paper framing
Honest negative-results paper or a "what we learned about composition on small bases" methodology paper. Could be sliced as:
1. **Negative-results focus:** "Bounded prompt-level adapter mixing does not improve a small base model on multi-domain QA when measured externally. The bottleneck is geometry, not routing."
2. **Methodology focus:** "Building paired-test, externally-judged evaluation infrastructure for multi-adapter composition on small bases."
3. **Apple Silicon engineering focus:** TCAR as a system design that achieves near-Mistral quality at $\sim$1/4 VRAM (with explicit latency tradeoff).

The third framing is the most "publishable" if paired with strong reproducibility (which depends on `missing_artifacts.md` item 2 being resolved).
