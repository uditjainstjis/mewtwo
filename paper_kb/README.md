# Paper Knowledge Base (PKB)

**Date created:** 2026-05-05
**Maintainer audience:** future paper-writing models (and human reviewers).
**Brand note:** repo-internal name is **Mewtwo** (working title for the methodology); historical artifacts use *Synapta* and *adapter*. We do not own *Synapta* as a name; treat it as a placeholder. The methodology name on any paper draft should be Mewtwo or a fresh non-conflicting name.

## What this PKB is for
Build the BEST POSSIBLE evidence record from this repo so a downstream model can write multiple top-tier papers without missing details, without overclaiming, and without re-inventing the chronology.

## Structure
```
paper_kb/
├── README.md                          # this file
├── headline_numbers_expanded.md       # every quotable number with provenance + caveats + safe interpretation
├── missing_artifacts.md               # claims that depend on missing files; what to ask the user for
├── experiments/                       # one experiment card per research thread (standardized 7-section format)
│   ├── 01_synapta_v1_prompt_composition.md
│   ├── 02_synapta_v2_multidomain.md
│   ├── 03_clamp_ablation_norm_ratio.md
│   ├── 04_routing_gap_oracle_vs_real.md
│   ├── 05_external_md_blind_judge.md
│   ├── 06_router_sft_dpo_5000.md
│   ├── 07_tcar_collaborative.md
│   ├── 08_injection_9technique_ablation.md
│   ├── 09_lori_moe_phases.md
│   ├── 10_lori_orthogonality.md
│   ├── 11_phase1_single_adapter_30b.md
│   ├── 12_static_composition_failure_30b.md
│   ├── 13_format_guard_humaneval.md
│   ├── 14_cold_swap_latency.md
│   ├── 15_code_paradox_replication.md
│   ├── 16_rank_scaling_zoo.md
│   ├── 17_humaneval_scoring_bug.md
│   ├── 18_bfsi_pipeline.md
│   ├── 19_bfsi_extract_eval_n664.md
│   ├── 20_bfsi_recall_eval_n214.md
│   ├── 21_indiafinbench_ood_n324.md
│   ├── 22_benchmark_v1_n60.md
│   └── 23_frontier_comparison_n15.md
└── outlines/                          # paper cluster outlines (NOT papers, just roadmaps)
    ├── A_apple_silicon_composition.md
    ├── B_format_guard_codeparadox.md
    ├── C_regulatory_bfsi_indiafinbench.md
    └── D_lori_moe_orthogonal_experts.md
```

## How a paper-writing model should consume this PKB

1. Start at `headline_numbers_expanded.md` — every number with a source path and a "safe interpretation" sentence. If a number is not there, do not cite it.
2. Read the cluster `outlines/X_*.md` for the paper(s) being drafted. The outline lists which experiment cards are load-bearing and which are aspirational.
3. Open the cited `experiments/NN_*.md` cards for full method/result/limitation context. The 7-section structure is consistent across all cards.
4. Cross-check `missing_artifacts.md` — any claim that depends on a missing file is listed with what would be needed to make it paper-safe.
5. Cross-check `RESEARCH_HISTORY/98_KNOWN_LIMITATIONS_AND_BUGS.md` (parent repo) for the master correction log.

## Evidence hierarchy used throughout

- **Category 1 (PRIMARY):** local code, JSONL outputs, JSON summaries, training logs, checkpoint files. Fully trusted.
- **Category 2 (SECONDARY):** markdown narratives, chronicles, READMEs, this PKB. Trusted only to the extent they match Category 1.
- **Category 3 (UNVERIFIED):** claims that depend on artifacts not present in the local repo (e.g., off-device RTX results, missing JSON files). Listed in `missing_artifacts.md`. Do NOT upgrade to "solid" without producing the artifact.

## Naming convention reminder
- "Synapta" appears in older code/docs and on YC application drafts; we do NOT own this trademark. Switch the brand before any external paper or product release.
- Suggested replacements (in order of preference): **Mewtwo** (matches repo) · **AdapterZoo** · **Routelane**
- A future global rename is mechanical: search-and-replace `Synapta` → `<NewName>` in `paper_kb/`, `paper/`, `RESEARCH_HISTORY/`, `data/benchmark/synapta_indian_bfsi_v1/`. Adapter directory paths (`adapters/nemotron_30b/bfsi_extract/best/`) do not need renaming.

## Concurrent submission policy (NeurIPS 2026)
Recap of the venue rules for the future paper-writing model:

1. **Same paper to another venue while under NeurIPS review: forbidden** (NeurIPS dual-submission policy).
2. **arXiv preprint: allowed and encouraged** (use `\usepackage[preprint]{neurips_2026}`).
3. **Non-archival workshops** (most ICML/ICLR workshops): allowed.
4. **Substantially different papers carved from the same research base: allowed.** This PKB intentionally separates clusters A/B/C/D so that distinct papers can be drafted in parallel without dual-submission conflicts.
