# Synapta — Project Overview

Generated 2026-05-02 by the restructure pass. Single source of truth for navigating this repo.

---

## What this project is

Adapter-routing platform for sovereign-AI enterprise deployment. Single base model + dozens of swappable domain experts, deployed inside customer firewall.

**Core finding:** Format Guard adapter routing on Nemotron-30B achieves **+17.1 pp HumanEval** over base (n=164, p<0.001).

---

## Headline numbers (defensible, deck-ready)

| Benchmark | Base Nemotron-30B | Our Routing | Lift | n |
|---|---|---|---|---|
| ARC-Challenge | 20.0% | 31.0% | +11.0 | 100 |
| MATH-500 | 41.5% | 56.0% | +14.5 | 200 |
| HumanEval (rescored v2) | **56.1%** | **73.2%** | **+17.1** | **164** |
| MBPP | 8.0% | 5.0% | -3.0 | 100 |

Code Paradox at n=200 on Nemotron-30B: code adapter (56%) > math adapter (50.5%) on MATH-500. **+5.5 pp**.

---

## Read in this order

1. **`overnight_run/FINAL_SUMMARY.md`** — what was done overnight 2026-05-02, with all wins/rollbacks
2. **`overnight_run/TALKING_POINTS.md`** — verbatim sentences for CTO meeting + YC application
3. **`overnight_run/DECK_UPDATE_GUIDE.md`** — exact find/replace edits for the deck
4. **`docs/RESULTS_INDEX.md`** — every results JSON catalogued
5. **`docs/ADAPTERS_INDEX.md`** — every PEFT adapter catalogued
6. **`docs/DOCS_INDEX.md`** — every markdown file catalogued
7. **`docs/MASTER_KNOWLEDGE_BASE.md`** — historical research chronicle (long, but comprehensive)

---

## Directory layout

```
mewtwo/
├── README.md                    # this file (or symlink to PROJECT_OVERVIEW.md)
├── PROJECT_OVERVIEW.md          # detailed structure map (this doc)
├── SYNAPTA_PITCH_DECK.{pptx,pdf}  # YC / CTO pitch deck
├── build_pitch_deck.py          # deck source
├── requirements.txt             # Python deps
├── setup_env.sh                 # env setup
│
├── docs/                        # master research + index docs
│   ├── MASTER_KNOWLEDGE_BASE.md
│   ├── MASTER_RESEARCH_CHRONICLES.md
│   ├── MASTER_EXPERIMENT_REPORTS.md
│   ├── MASTER_TASKS_AND_PLANS.md
│   ├── RESULTS_INDEX.md         # generated
│   ├── ADAPTERS_INDEX.md        # generated
│   └── DOCS_INDEX.md            # generated
│
├── manuscripts/                 # paper drafts
│   ├── synapta_systems.md
│   ├── code_paradox.md
│   └── compiled_manuscript.md
│
├── overnight_run/               # 2026-05-02 mission output (canonical recent state)
│   ├── FINAL_SUMMARY.md
│   ├── TALKING_POINTS.md
│   ├── DECK_UPDATE_GUIDE.md
│   ├── WAKE_UP_README.md
│   ├── findings/                # 7 detailed finding docs
│   ├── qa_pairs/                # 21 JSONL files of Q&A
│   ├── scripts/                 # reproducible runners
│   └── demo_artifacts/          # server_fixed.py — drop-in demo fix
│
├── models/                      # base model weights (60 GB)
│   ├── nemotron/                # Nemotron-H-30B (NemotronHForCausalLM)
│   └── Qwen3.5-0.8B/
│
├── checkpoints/                 # PEFT adapter training output
│   ├── nemotron_lori/adapters/{math,code,science}/
│   │   ├── best/                # canonical inference adapter
│   │   ├── final/               # last training step
│   │   ├── checkpoint-N/        # intermediate (training history)
│   │   └── dare_sparsified/     # post-hoc DARE sparsification
│   ├── lori_moe/                # LoRI-MoE Qwen-1.5B work
│   └── MATHEMATICS/             # ?
│
├── hf_publish/                  # publishing-ready adapters (hardlinked to checkpoints/...)
│   └── {math,code,science,merged}/
│
├── hf_kaggle_opensource/        # small-model adapter zoo + 3rd-party benchmarks
│   ├── outputs/                 # qwen_0.8b_*, nemotron_4b_* adapter checkpoints (matched ranks 1/2/8/128/1024/3072)
│   ├── benchmarks/              # tau-bench, LLMs-Planning (3rd-party suites)
│   └── results/                 # small-model eval outputs
│
├── router_adapters/             # router classifier checkpoints (Synapta v1 era)
├── submission_adapter/          # single packaged adapter (Synapta submission)
│
├── src/                         # Synapta source code
│   ├── demo/server.py           # FastAPI WebSocket demo (FIXED 2026-05-02)
│   ├── demo/server_original_backup.py  # pre-fix backup
│   ├── lori_moe/                # LoRI-MoE architecture (model, training, inference)
│   ├── adapters/, composition/, eval/, engine/, routers/, training/
│   └── ...
│
├── scripts/                     # standalone runners
│   ├── token_router_eval.py     # known-good Nemotron-30B routing benchmark
│   ├── master_pipeline.py       # Phase 1 single-adapter pipeline
│   ├── routing_grand_comparison.py  # 12-strategy bake-off
│   └── ...
│
├── results/                     # benchmark outputs
│   └── nemotron/                # Nemotron-30B benchmark JSONs
│
├── data/                        # datasets, prompts, traces
├── logs/                        # training and run logs
├── archive/                     # historical work (synapta_v1 etc)
├── backend/                     # backend services (Synapta v1 MLX)
├── kaggle_notebook/             # Kaggle deployment artifacts
├── paper_v2/                    # paper figures + tables
├── prompts/                     # prompt templates
├── tests/                       # unit tests
├── configs/                     # config files
│
├── _restructure/                # restructure audit logs (this pass)
├── .venv/                       # Python venv (do not touch)
├── perplexity-mcp-venv/         # MCP server venv
└── .git/                        # git history
```

---

## Reproducibility & safety

- Pre-restructure git tag: `pre-restructure-2026-05-02` at commit `e391284`
- Full pre-restructure backup at: `/home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP/`
- The restructure on 2026-05-02 only deleted regeneratable artifacts (`__pycache__`, `pip.pid`, `tmp/` snapshot) and replaced bit-identical duplicates with hardlinks. **No model weights or result JSONs were deleted.** All file paths still resolve.

---

## Known caveats (read before citing numbers)

1. **HumanEval scoring bug:** original `extract_code` in `scripts/master_pipeline.py` and `overnight_run/scripts/run_humaneval_n164.py` (v1) dropped imports from prompts and stripped body indentation. This caused systematic ~30 pp under-counting on Nemotron-30B HumanEval. **Use the v2 rescored numbers** in `overnight_run/qa_pairs/humaneval_full_*_rescored.jsonl` and findings in `overnight_run/findings/humaneval_n164.md`.

2. **Code Paradox cross-family overclaim retracted:** n=50 results on Qwen-0.8B and Nemotron-Mini-4B suggested cross-family replication, but n=200 follow-up on Qwen-0.8B showed the paradox does NOT replicate at full sample. **Robust claim:** +5.5 pp at n=200 on Nemotron-30B only. Honest update in `overnight_run/findings/code_paradox_replication.md`.

3. **Demo had 5 bugs** in original `src/demo/server.py` (repetition_penalty=1.3, wrong neural router input distribution, initial adapter not set, marketing system prompt, routing interval mismatch). Fixed 2026-05-02. Backup at `src/demo/server_original_backup.py`. Bug breakdown in `overnight_run/findings/demo_server_bugs.md`.

