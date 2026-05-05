# Experiment Card 10 — LoRI-MoE Expert Orthogonality (Structural)

## 1. Research question and hypothesis

Are LoRI-MoE experts approximately orthogonal in cosine similarity by construction (frozen shared random projection $B$ + sparse trainable expert factors $A_e$)?

The structural prediction: with $B$ shared and frozen, and $A_e$ trained sparsely per expert with disjoint target dimensions, off-diagonal cosine similarity between experts should approach zero.

## 2. Dataset and task definition

This experiment is a structural-property measurement on saved adapter weights, not a downstream benchmark.

- **Items measured:** pairwise cosine similarity of expert update matrices for the 5 saved LoRI-MoE Qwen-2.5-1.5B experts (code, legal, math, medical, science).
- **Aggregation:** off-diagonal entries of the 5×5 cosine matrix.

## 3. Model and configuration

- **Base:** Qwen-2.5-1.5B-Instruct (LoRI-MoE training stack).
- **Frozen shared projection:** `adapters/lori_moe/_shared_projection_B.pt` (or similar).
- **Per-expert sparse $A$ matrices:** `adapters/lori_moe/lori-qwen2.5-1.5b-<domain>/`

## 4. Evaluation protocol

- Compute pairwise cosine on the per-expert update matrices (LoRA $\Delta W = B A_e$ effective).
- Report mean $|$cosine$|$ off-diagonal.

## 5. Main quantitative results

| Statistic | Value |
|---|---|
| Mean $|$cosine$|$ off-diagonal (5 experts) | **0.00685** |

Source: `docs/MEWTWO_RESEARCH_PAPER_DRAFT.md` cites this number from inspection of the saved expert weights. The exact computation script is part of the LoRI-MoE codebase but the per-pair table is not consolidated as a JSON.

## 6. Negative results, limitations, and bugs

- **The orthogonality claim is structural, not behavioural.** Low cosine in the LoRA update matrices does not imply that the experts produce non-interfering outputs on real queries.
- **No downstream benefit demonstrated yet.** Phase 3 composite routing on the same experts (`09_lori_moe_phases.md`) underperformed single-best on every tested benchmark.
- **Single-decimal-precision sourcing:** the 0.00685 number is sourced from a narrative document; for paper-grade citation, re-run the cosine computation script and save a versioned `results/lori_moe/orthogonality_matrix.json`.
- **The paper-safe interpretation:** "Saved LoRI-MoE expert weights have very low cross-domain cosine overlap by construction. Whether this orthogonality translates to downstream benefit remains unverified, since prompt-level composite routing did not exceed best-single."

## 7. Artifact map

PRIMARY:
- `adapters/lori_moe/_shared_projection_B.pt`
- `adapters/lori_moe/lori-qwen2.5-1.5b-{code,legal,math,medical,science}/adapter_model.safetensors`
- LoRI-MoE source under `synapta_src/src/lori_moe/` (referenced in narrative)

SECONDARY:
- `docs/MEWTWO_RESEARCH_PAPER_DRAFT.md` (cites 0.00685)

MISSING / off-device:
- consolidated `orthogonality_matrix.json` with the full 5×5 cosine matrix (currently the number is narrative-cited only)
- a downstream-benefit benchmark on the orthogonal experts (gated by `09_lori_moe_phases.md`)
