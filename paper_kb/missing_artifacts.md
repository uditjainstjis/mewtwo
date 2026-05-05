# Missing Artifacts — Asks for the User

This file lists every artifact mentioned in the docs/code that is NOT present in the current local repo. Until each item is restored, dependent claims are flagged "unverified" in `headline_numbers_expanded.md` and the relevant experiment card.

## High priority (claims gated on these)

### 1. LoRI-MoE token-level routing benchmarks ✅ RESOLVED 2026-05-05
- **Status:** RESOLVED. Run via `synapta_src/data_pipeline/19_eval_lori_moe_token_routed.py` on RTX 5090 in 26 minutes.
- **Output:** `results/lori_moe/phase3_token_routed.json` and `results/lori_moe/phase3_token_routed_predictions.jsonl`.
- **Headline:** GSM8K $+61.5$ pp over prior prompt-level composite (4\% → 65.5\%); ARC honest negative ($-30$ pp vs base because science adapter degrades); MMLU marginal $+1$ pp. See rows D-tr-* in `headline_numbers_expanded.md`.
- **Cluster D paper status updated:** from "partial result" to **mixed-result publishable** (positive on GSM8K, honest negative on ARC, marginal MMLU).

### 2. 20 Synapta Apple Silicon adapter safetensors
- **What's missing:** safetensors weights for the 20 domain LoRA experts referenced in `docs/MEWTWO_RESEARCH_PAPER_DRAFT.md` for the v1/v2 Apple Silicon work.
- **Currently present:** `adapters/MATHEMATICS_archive/`, `adapters/router/`, plus partial Qwen3.5/Qwen2.5 multi-rank files; the "20 expert" set is not fully restored.
- **Dependent claim:** Apple Silicon reproducibility of the v1/v2 prompt-level composition results.
- **Ask:** Please restore the 20 expert adapter safetensors into `adapters/synapta_v1_apple_silicon/` (or similar) so cluster A can be reproduced end-to-end on a current Apple Silicon machine.

### 3. GC-LoRI orthogonality benchmarks
- **What's missing:** a benchmark output file demonstrating that GC-LoRI experts produce a downstream gain over single-adapter baselines, not just the cosine-orthogonality structural claim.
- **Currently present:** the structural claim (avg \|cosine\| = 0.00685 off-diagonal) is sourced from inspection of saved weights; no benchmark gain is recorded.
- **Dependent claim:** "GC-LoRI orthogonal experts improve performance" (any paper would need this).
- **Ask:** Please train and benchmark GC-LoRI on at least one benchmark (GSM8K or HumanEval) with paired statistical test against best-single-adapter, and place the output in `results/gc_lori_benchmarks/`.

### 4. n=200 cross-family positive Code Paradox replication
- **What's missing:** $n=200$ runs of code-adapter on math/reasoning benchmarks across at least one base other than Nemotron-30B; only the in-domain regression replication exists at $n=200$ on Qwen-3.5-0.8B.
- **Currently present:** `results/overnight/qa_pairs/code_paradox_qwen_n200_summary.json` (in-domain only).
- **Dependent claim:** "Code Paradox (asymmetric positive cross-domain transfer) replicates across base models."
- **Ask:** Either (a) explicitly do not claim cross-family positive replication in any paper, or (b) run $n=200$ Qwen-3.5-0.8B / Nemotron-Mini-4B code-adapter on MATH-500 and ARC.

## Medium priority (would improve papers but not block)

### 5. Frontier-API comparison at $n=60$ (full Benchmark v1)
- **What's missing:** Anthropic Claude / OpenAI GPT-4o / Google Gemini scores on the released 60-question Synapta Indian BFSI Benchmark v1.
- **Currently present:** subagent-derived $n=15$ comparison only.
- **Dependent claim:** "Open public seed baselines" rows in benchmark README.
- **Ask:** $\sim \$3$ in API budget would unlock these. Currently gated on API key.

### 6. Multi-base Format Guard replication
- **What's missing:** Format Guard on a second 30B-class base (Qwen3-32B, Llama-3-70B, Mixtral-8x7B).
- **Currently present:** Format Guard on Nemotron-Nano-30B-A3B only.
- **Dependent claim:** "Format Guard generalises across bases."
- **Ask:** Train math/code/science adapters on a second 30B-class base and re-run the Format Guard $n=164$ HumanEval evaluation.

### 7. Per-rank full evaluation grid
- **What's missing:** consolidated per-rank evaluation grid (ARC/HumanEval/MATH-500/MBPP × 6 ranks × 4 domains × 2 bases).
- **Currently present:** spot-check at $n=50$ HumanEval in `results/bfsi_swarm_extras/code_paradox_rank_scaling.json`.
- **Dependent claim:** "Rank scaling does not rescue code-on-code regression."
- **Ask:** Run the full grid (or accept the smaller $n=50$ disclosure as sufficient).

### 8. IRDAI / PFRDA / FEMA corpus extension
- **What's missing:** scraped corpora for these three Indian regulators.
- **Currently present:** RBI + SEBI only.
- **Dependent claim:** "Multi-regulator coverage" in any pitch.
- **Ask:** Either (a) explicitly do not claim multi-regulator coverage, or (b) extend pipeline 01b/01c/01d to cover the additional regulators.

## Low priority (nice-to-have)

### 9. Live Loom of demo
- **What's missing:** a recorded screencast of `synapta_src/demo/synapta_live_demo.py` showing the +30 pp lift on a fresh question.
- **Ask:** $\sim$30 min recording time; user only.

### 10. Public HuggingFace + Kaggle push of Benchmark v1
- **What's missing:** the actual public release of `data/benchmark/synapta_indian_bfsi_v1/` to HF and Kaggle.
- **Currently present:** push scripts (`14_publish_benchmark.py`, `15_publish_kaggle.py`) dry-run validated; awaiting user's HF + Kaggle tokens.
- **Ask:** Provide tokens or run the push scripts manually.

## Cleanup / housekeeping

### 11. Path rewrite audit
- Several Apple-Silicon-era docs reference paths like `/Users/uditjain/Desktop/adapter/results/...` and `/Users/uditjain/Desktop/mewtwo/results/...`. The current Linux repo is at `/home/learner/Desktop/mewtwo/`. Some referenced JSONs may have been lost in the migration; spot-check the `results/nemotron/` references in `docs/MASTER_KNOWLEDGE_BASE.md` against actual file presence.

### 12. Demo bug fix backup
- Per `09_humaneval_scoring_bug.md` and the demo bug docs, an `src/demo/server_original_backup.py` is mentioned. Verify it exists; if not, the historical claim about pre-fix demo behaviour is `Category 3` (unverified).
