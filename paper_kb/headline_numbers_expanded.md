# Headline Numbers — Expanded with Provenance, Caveats, and Safe Interpretation

Every number that may be quoted in a paper, deck, or pitch must trace to a row here. If a number is not in this table, do not cite it.

## Apple Silicon / Synapta v1, v2, TCAR (Cluster A)

| Claim ID | Value | Primary source | Experiment card | Caveats / bug history | Safe interpretation |
|---|---|---|---|---|---|
| A-v1-sim-AC-vs-SA | $\Delta_\text{SIM} = -0.011$ on $n=100$ | `results/decision_summary.md` | `01_synapta_v1_prompt_composition.md` | Headline H1 FAIL | Bounded $K=2$ prompt-level composition does not beat single-adapter routing on the v1 single-domain benchmark. |
| A-v1-ppl-AC | 58.0 vs base 64.5 | same | same | n/a | Multi-adapter exposure lowers perplexity even when similarity does not improve. |
| A-v1-collapse-Unclamped | 8% catastrophic collapse | `results/decision_summary.md` | same | n/a | Unclamped activation mixing causes representational collapse on $\sim$8% of prompts. |
| A-v2-sim-AC-vs-SA | $+0.0171$ MD split, $+0.03$ threshold MISSED | `results/v2_decision_summary.md` | `02_synapta_v2_multidomain.md` | Sub-threshold positive | Multi-adapter composition shows a directionally positive but sub-threshold gain on a multi-domain benchmark. |
| A-clamp-norm-equiv | $\Delta = -0.0003$ between norm-ratio and weight-cap clamp | `results/v2_clamp_ablation_summary.md` | `03_clamp_ablation_norm_ratio.md` | n/a | The norm-ratio clamp is empirically identical to the simpler weight-cap on this base+adapter set. The H5 failure is a property of the model, not of the clamp implementation. |
| A-routing-gap-oracle | Oracle headroom $+0.0206$, real router realised $+0.0054$ | `results/v2_routing_gap_summary.md` | `04_routing_gap_oracle_vs_real.md` | $\sim$26\% of headroom recovered | Oracle routing gives a $+2.06\%$ similarity ceiling on MD; a real CoT top-2 router recovers $\sim$26\% of that headroom. The compositional ceiling is small, and the bottleneck is the model/adapter geometry rather than the router. |
| A-router-CoT-fail | 48.7\% multi-label routing accuracy | `docs/MASTER_KNOWLEDGE_BASE.md` H5 | `04_routing_gap_oracle_vs_real.md` | n/a | Generative CoT routing performs near random on multi-label routing tasks. |
| A-router-embed | 78.7\% embedding routing | same H6 | `06_router_sft_dpo_5000.md` | n/a | Spatial-embedding routing outperforms generative CoT routing. |
| A-router-sft | **85\%** exact-match, 100\% partial overlap, mean F1 0.945 | `results/router_accuracy_sft_5000_valid_holdout_mpsfix.json` | `06_router_sft_dpo_5000.md` | n=100 holdout | A small SFT-tuned router beats CoT and embedding routing. |
| A-router-dpo | 42\% exact-match, regressed | `results/router_accuracy_dpo_5000_valid_holdout_mpsfix.json` | same | DPO objective optimised pairwise preference but DEGRADED routing classification | Pairwise-preference DPO objectives can degrade classification quality even when they optimise the stated objective. |
| A-tcar-vs-mistral-sim | $0.6900$ vs $0.6907$ on $n=100$ external MD | `results/md_external_v2_comparison_summary.json`, `results/tcar_collaborative_dpo5000_mpsfix_100.jsonl` | `07_tcar_collaborative.md` | latency 24.2s vs 10.7s | TCAR (Qwen-1.5B) nearly matches Mistral-7B on external semantic similarity ($-0.0007$) at $\sim 2.3\times$ latency. |
| A-tcar-vs-mistral-f1 | $0.2712$ vs $0.2917$ token F1 | same | same | TCAR loses on F1 | TCAR does NOT beat Mistral on token F1 ($-0.0205$); the "Synapta beats Mistral" narrative is not supported on F1. |
| A-tcar-blind-judge | Mistral wins 23-26 of 30; weighted_merge 6/30 | `results/md_pairwise_*_summary.json` | `05_external_md_blind_judge.md` | Stratified blind, $n=30$ | Externally authored blind comparison **does not support** the "Qwen Synapta beats Mistral" claim from earlier internal benchmarks. |
| A-9tech-best | `weighted_merge` best Qwen on external 100; sequential_reverse best on internal | `results/injection_hypotheses_eval_full_20260408.jsonl` (n=360, 9 methods × 40 Q) | `08_injection_9technique_ablation.md` | n/a | The 9-technique injection ablation found that the best technique depends on the benchmark; internal-vs-external rankings disagree, motivating the external benchmark pivot. |

## LoRI-MoE on Qwen-2.5-1.5B (Cluster D)

| Claim ID | Value | Primary source | Experiment card | Caveats | Safe interpretation |
|---|---|---|---|---|---|
| D-base-gsm8k | 26\% | `results/lori_moe/phase1_baselines.json` ($n=200$) | `09_lori_moe_phases.md` | n/a | Qwen-2.5-1.5B base zero-shot scores 26\% on GSM8K. |
| D-base-arc | 76.5\% | same | same | n/a | Base zero-shot ARC accuracy. |
| D-base-mmlu | 56.5\% | same | same | n/a | Base zero-shot MMLU accuracy. |
| D-single-math-gsm8k | 53.0\% | `results/lori_moe/phase2_single_adapter.json` | same | math adapter only, $n=200$ | Math adapter improves GSM8K from 26\% to 53\% ($+27$ pp). |
| D-single-legal-arc | 77.5\% | same | same | n/a | Legal adapter improves ARC from 76.5\% to 77.5\% ($+1$ pp). |
| D-composite-gsm8k | 4\% (catastrophic regression) | `results/lori_moe/phase3_composite.json` | same | composite top-1 routed | Composite top-1 routing catastrophically regressed GSM8K from single-best 53\% to 4\%; routing failed to select math adapter on math queries. |
| D-composite-arc | 72\% | same | same | $-5.5$ pp vs single legal | Composite routing on ARC is below the single best legal adapter. |
| D-orthogonality | avg \|cosine\| = 0.00685 off-diagonal | `synapta_src/src/lori_moe/...` | `10_lori_orthogonality.md` | structural claim, partially verified | Saved LoRI-MoE expert weights have very low cross-domain cosine overlap. The number 0.00685 derives from inspection of saved weights; full computation script should be re-run for paper-grade citation. |
| D-token-routed-end-to-end | **VERIFIED 2026-05-05**, see rows D-tr-* below | --- | `09_lori_moe_phases.md` | n/a | Token-level routed LoRI-MoE on Qwen-2.5-1.5B with 5 LoRI experts via Format Guard ($K=10$ token swap window) was run end-to-end on GSM8K, ARC, MMLU at $n=200$ each. Results below replace the prior "aspirational" status. |
| D-tr-gsm8k-base | 52.5\% (105/200) | `results/lori_moe/phase3_token_routed.json` | `09_lori_moe_phases.md` | n/a | Qwen-2.5-1.5B base, GSM8K, n=200 with chat-template + greedy + #### extraction. |
| D-tr-gsm8k-single | 65.5\% (131/200) | same | same | math adapter | LoRI math adapter alone, +13 pp over base. |
| D-tr-gsm8k-token-routed | **65.5\%** (131/200) | same | same | mean 0.12 swaps/Q, FG zero-overhead replicates | Token-routed FG matches single-best; **+61.5 pp over the prior prompt-level composite-top-1 result of 4\%**. |
| D-tr-arc-base | 75.5\% (151/200) | same | same | n/a | Qwen-2.5-1.5B base on ARC-Challenge n=200 (the base is strong on ARC). |
| D-tr-arc-single | 42.5\% (85/200) | same | same | science adapter | LoRI science adapter alone DEGRADES base by $-33$ pp; honest negative finding. |
| D-tr-arc-token-routed | 45.0\% (90/200) | same | same | mean 0.18 swaps/Q | FG slightly above single ($+2.5$ pp) but **$-30$ pp below base**: when no adapter helps, FG cannot recover unless it can route to "no adapter". This is a publishable limitation finding. |
| D-tr-mmlu-base | 42.5\% (85/200) | same | same | n/a | Qwen-2.5-1.5B base on MMLU n=200. |
| D-tr-mmlu-single | 43.5\% (87/200) | same | same | math adapter | Marginal $+1$ pp over base. |
| D-tr-mmlu-token-routed | 43.5\% (87/200) | same | same | mean 0.10 swaps/Q | Identical to single-best; FG zero-overhead replication. |
| D-tr-runtime | 26 minutes total wall-clock | same | same | RTX 5090, Qwen-2.5-1.5B bf16 | Faster than the 50-min estimate. |

## Nemotron-30B Phase 1 / Code Paradox (Cluster B)

| Claim ID | Value | Primary source | Experiment card | Caveats | Safe interpretation |
|---|---|---|---|---|---|
| B-base-arc100 | 20.0\% | `results/nemotron/master_results.json` ($n=100$) | `11_phase1_single_adapter_30b.md` | n/a | Nemotron-Nano-30B-A3B 4-bit base on ARC-Challenge $n=100$. |
| B-code-arc | 31.0\% (+11.0 pp) | same | same | n/a | Code adapter on ARC, **paradoxically** highest of all single adapters. |
| B-math-humaneval-v1 | 60.0\% | same | same | v1 buggy scoring | Math adapter on HumanEval $n=100$ under v1 scoring. Paper-safe HumanEval numbers are at $n=164$ v2 (see `17_humaneval_scoring_bug.md`). |
| B-code-humaneval-v1 | 27.0\% (-23 pp) | same | same | v1 buggy scoring | Code-on-code regression. The direction (code training breaks code) is preserved at v2 scoring; absolute floor moves $\sim$30 pp. |
| B-code-math500 | 56.0\% (+14.5 pp) | `results/nemotron/master_results.json` ($n=200$ MATH-500) | same | n/a | Code adapter on MATH-500 is highest of all adapters. |
| B-merge-arc | 19.0\% (vs base 20\%) | same | `12_static_composition_failure_30b.md` | DARE/TIES uniform | Static merging of 4 adapters degraded ARC below base. |
| B-merge-mixed-prob | $+0.0$ pp on $n=45$ mixed-domain probe | `results/nemotron/sprint_results.json` | same | uniform weights only | Static composition shows zero gain over best-single on $n=45$ mixed-domain. |
| B-paradox-qwen-n200 | base 15\%, math 16\%, code 12\% | `results/overnight/qa_pairs/code_paradox_qwen_n200_summary.json` | `15_code_paradox_replication.md` | $n=200$ | Code-on-code regression replicates on Qwen-3.5-0.8B at $n=200$. |
| B-rank-scaling | code at $r=1024$ scores 10\% on Nemotron-Mini-4B ($n=50$) | `results/bfsi_swarm_extras/code_paradox_rank_scaling.json` | `16_rank_scaling_zoo.md` | small $n$ | Rank scaling does not rescue the code-on-code regression. |
| B-humaneval-base-n164 | **56.1\%** [Wilson 48.4, 63.5] | `results/overnight/qa_pairs/humaneval_rescored_summary.json` | `13_format_guard_humaneval.md` + `17_humaneval_scoring_bug.md` | v2 scoring corrected for import + indent stripping | Base Nemotron-30B 4-bit pass@1 on full HumanEval, with corrected extraction. |
| B-humaneval-fg-n164 | **73.2\%** [Wilson 65.9, 79.4] | same | same | n/a | Format Guard pass@1 on full HumanEval. |
| B-humaneval-fg-lift | **+17.1 pp** McNemar $\chi^2 = 15.68$, $p < 10^{-3}$ | `docs/findings/humaneval_statistical_analysis.md` | same | paired $n=164$, $b_{10}=11, b_{01}=39$ | Format Guard improves HumanEval over base by $+17.1$ pp; statistically significant at NeurIPS standards. |
| B-fg-arc | 31.0\% (matches best single = code) | same | same | n/a | FG matches single-best on ARC. |
| B-fg-math500 | 56.0\% (matches best single = code) | same | same | n/a | FG matches single-best on MATH-500. |
| B-fg-mbpp | 5.0\% (-3.0 pp regression) | same | same | format-rigid regression | FG regresses MBPP by 3 pp due to mid-sequence swap disrupting Python indentation. |
| B-cold-swap | 315.9 ms / swap NVMe SSD over 44 swaps | `results/cold_swap_metrics.json` | `14_cold_swap_latency.md` | warm GPU swap is $O(1)$ pointer flip | Cold-swap latency is 316 ms; warm (all adapters in VRAM) is $O(1)$. |

## BFSI cluster (Cluster C)

| Claim ID | Value | Primary source | Experiment card | Caveats | Safe interpretation |
|---|---|---|---|---|---|
| C-corpus-pdfs | 130 PDFs (80 RBI MDs + 50 SEBI MCs), 8.06M chars, 4{,}477 raw QAs | `data/{rbi,sebi}_corpus/manifest.json`, `synapta_src/data_pipeline/04_*` | `18_bfsi_pipeline.md` | n/a | Public-domain Indian regulatory corpus, deterministic 3-tier QA construction, 10-check validator at 98.45\% pass. |
| C-doc-disjoint | 26 PDFs (20\%) entirely held out | `data/rbi_corpus/qa/split_manifest_v2.json` | same | n/a | Document-disjoint split eliminates chunk-neighbour memorisation. |
| C-bfsi-base | 58.7\% [54.95, 62.42] substring | `results/bfsi_eval/summary.json` | `19_bfsi_extract_eval_n664.md` | $n=664$ paired | Nemotron-30B base substring on held-out. |
| C-bfsi-adapter | **89.6\%** [87.06, 91.71] substring | same | same | $n=664$ paired | bfsi_extract LoRA substring on held-out. |
| C-bfsi-fg | 88.7\% [86.07, 90.89] substring | same | same | $n=664$ paired | Format Guard with 4 adapters substring. |
| C-bfsi-mcnemar | $b_{10}=14, b_{01}=219$, $p = 1.66 \times 10^{-48}$ | same | same | adapter vs base | Adapter improvement-to-regression ratio 15.6$\times$; $p$-value is exact-binomial via scipy. |
| C-bfsi-fg-vs-direct | $b_{10}=6, b_{01}=0$, $\Delta = -0.9$ pp, $p=0.031$ | same | same | n/a | FG never improves over direct adapter on BFSI; differs on 6/664 questions, all FG-loses. Empirical zero-overhead claim. |
| C-bfsi-tier3 | $+39.5$ pp on heading-extractive | same | same | $n=278$ | Adapter dominates Tier 3. |
| C-bfsi-recall-base-f1 | 0.158 mean F1 | recomputed from `results/bfsi_recall_eval/eval_results.jsonl` (paired) | `20_bfsi_recall_eval_n214.md` | $n=214$ paired | Base mean token F1 on no-context recall task. |
| C-bfsi-recall-adapter-f1 | 0.219 mean F1 ($+38.4\%$ rel.) | same | same | n/a | Recall adapter mean F1. |
| C-bfsi-recall-wilcoxon | Wilcoxon $p = 1.50 \times 10^{-16}$ adapter > base | same | same | $n=214$ paired, scipy | Methodology generalises to a structurally different task type. |
| C-recall-fg-vs-adapter | Wilcoxon $p = 0.55$ FG vs adapter direct | same | same | n/a | FG zero-overhead replicates a third time on different task type. |
| C-indiafinbench-overall | **32.1\%** [Wilson 27.3, 37.4], F1 0.288, $n=324$ | `results/indiafinbench_eval/summary.json` | `21_indiafinbench_ood_n324.md` | OOD probe | Out-of-distribution score on IndiaFinBench. |
| C-indiafinbench-vs-gemini | -57.6 pp vs published Gemini Flash 89.7\% | same + `external_benchmarks/IndiaFinBench/` | same | different question style | Predicted OOD failure mode of fine-tuned adapter; the gap motivates per-customer training, not single general LLM. |
| C-bench-v1-base | 40.0\% (24/60) [28.6, 52.6] | `results/benchmark_v1_eval/summary.json` | `22_benchmark_v1_n60.md` | $n=60$ hand-curated | Base on Synapta Indian BFSI Benchmark v1. |
| C-bench-v1-adapter | 50.0\% (30/60) [37.7, 62.3], McNemar $p=0.0313$ | same | same | $n=60$ paired binomtest | Adapter $+10$ pp marginal at $\alpha=0.05$. |
| C-bench-v1-fg | 50.0\% (identical to direct, $p=1.0$) | same | same | mean 0.1 swaps/Q | FG zero-overhead replication on Benchmark v1. |
| C-bench-v1-substring-half | 80\% → 100\% on 30 substring-method Qs | same | same | n/a | Clean adapter win on substring-scored questions. |
| C-bench-v1-f1-half | both 0\% on 30 F1$\geq$0.5 questions | same | same | metric-cutoff artefact | F1$\geq$0.5 cutoff too strict for verbose paragraph-extraction style. |
| C-frontier-substring | Synapta 87\% vs Claude Opus 7\%, Sonnet 27\% | `results/frontier_comparison/subagent_results.jsonl` | `23_frontier_comparison_n15.md` | $n=15$ directional | Synapta dominates substring; Claude wins F1. Different deliverables. |
| C-frontier-f1 | Synapta 0.38 vs Claude Opus 0.65 | same | same | $n=15$ | Claude wins token F1 by $\sim$0.27. |

## Adapter zoo and infrastructure

| Claim ID | Value | Primary source | Caveats |
|---|---|---|---|
| Z-adapters-total | **72+ released** | `RESEARCH_HISTORY/96_ADAPTERS_RELEASED.md` | n/a |
| Z-zoo-nemotron4b | 30 adapters across {math SFT/DPO, code, science, merged} × ranks {1, 2, 8, 128, 1024, 3072} | `adapters/small_models_zoo/from_hf_kaggle/nemotron_4b_*` | n/a |
| Z-zoo-qwen | 37 adapters across Qwen-2.5/3.5 family with multi-rank | `adapters/small_models_zoo/from_hf_kaggle/qwen3.5_*` | per-rank evaluation grid not consolidated |
| Z-train-cost | 3h 28min for bfsi_extract; 17.81 GB VRAM peak; RTX 5090 | `logs/train_bfsi_extract.log` | n/a |
| Z-fg-vram | $\approx 18$ GB peak for 4-adapter Format Guard process | derived from base $\approx 17$ + 4 $\times$ 1.7 GB (with sharing) | fits 32 GB consumer GPU |

## Fragile / unverified numbers (do NOT use without producing the artifact first)

| Claim | Why fragile | What's needed |
|---|---|---|
| LoRI-MoE token-level routed end-to-end superiority | Composite was prompt-level routed top-1; token-level routed end-to-end NOT BENCHMARKED | Re-run benchmark suite with token-level routed LoRI-MoE on GSM8K/ARC/MMLU. |
| Cross-family Code Paradox replication at $n=50$ | Rolled back; only $n=200$ in-domain regression robust | $n=200+$ runs on Nemotron-30B-Mini and Qwen-7B for cross-family positive transfer. |
| GC-LoRI orthogonality + benefit | Only saved-weight cosine measured; no benchmark gain | Train and benchmark GC-LoRI on a benchmark suite. |
| 20 Synapta Apple Silicon adapter safetensors | Not all present in current repo per `MEWTWO_RESEARCH_PAPER_DRAFT.md` | Restore from RTX/Apple Silicon machine to `backendexpertadapters/`. |
| Some Nemotron statistics referenced in older docs | Path rewrites moved files; `results/nemotron/` may be incomplete | Audit `results/nemotron/` against `docs/MASTER_KNOWLEDGE_BASE.md` references. |
| HumanEval scoring v1 numbers anywhere in papers | Bug-affected; use v2 only | Use `humaneval_rescored_summary.json` as single source. |
