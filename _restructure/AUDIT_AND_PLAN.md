# Restructure Audit & Plan

**Started:** 2026-05-02 22:00 UTC
**Safety net:** `/home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP/` (full clone, 121 GB)
**Git tag:** `pre-restructure-2026-05-02` at commit `e391284`

## Audit findings

### Project structure (28 top-level dirs, 121 GB total)

| Dir | Size | Files | Purpose |
|---|---|---|---|
| `models/` | 60 GB | 91 | Base model weights (Nemotron-30B, Qwen-3.5-0.8B). DO NOT TOUCH. |
| `hf_kaggle_opensource/` | 43 GB | 8935 | Adapters + benchmarks (tau-bench, LLMs-Planning) |
| `.venv/` | 9.1 GB | many | Python venv. DO NOT TOUCH. |
| `checkpoints/` | 8.3 GB | 754 | Training intermediate checkpoints + best/final |
| `.git/` | 588 MB | — | Git history. DO NOT TOUCH. |
| `data/` | 225 MB | 84 | Datasets and prompts |
| `hf_publish/` | 180 MB | 56 | Publishing-ready adapters (DUPLICATES of checkpoints/.../best) |
| `router_adapters/` | 139 MB | 65 | Router model checkpoints |
| `perplexity-mcp-venv/` | 110 MB | many | MCP server venv. DO NOT TOUCH. |
| `submission_adapter/` | 45 MB | 5 | Single packaged adapter |
| `archive/` | 38 MB | 66 | Old work (synapta_v1 etc) |
| `logs/` | 21 MB | 70 | Training and run logs |
| `results/` | 15 MB | 322 | Benchmark output JSONs |
| `overnight_run/` | 3.1 MB | 75 | Mission output (this overnight) |
| `docs/` | 1.3 MB | 14 | Master research docs |
| `src/` | 1.1 MB | 106 | Python source |
| `scripts/` | 876 KB | 68 | Standalone scripts |
| `manuscripts/` | 32 KB | 4 | Paper drafts |

### Duplicates (verified by sha256)

- **32 copies of identical Qwen tokenizer.json** (~11MB each) across `hf_kaggle_opensource/outputs/` adapter dirs and `checkpoints/lori_moe/qwen3.5_0.8b/`. Total ~340MB recoverable.
- **21 copies** of another identical file (likely Nemotron tokenizer or vocab).
- **3× of `hf_publish/{math,code,science}/adapter_model.safetensors`** is bit-identical to `checkpoints/nemotron_lori/adapters/{math,code,science}/best/adapter_model.safetensors`. ~270MB.
- Multiple **dare_sparsified** dirs that are bit-equivalent to their `final/` siblings.
- 7-19 copies of various PEFT config/template files.

### Junk

- 27 project-level `__pycache__` dirs (1.8 MB)
- `pip.pid` (4 KB)
- `tmp/` (124 KB)
- 13,276 `.pyc` files (mostly inside `.venv`, untouched)

### Documentation sprawl

417 markdown files total (excluding venvs/.git). Highest concentrations:
- `hf_kaggle_opensource/results/` — 32 files
- `results/md_generation_prompts/` — 22 files
- `archive/synapta_v1/` — 19 files
- `results/` — 16 files
- `docs/` — 12 (master research docs)
- `overnight_run/findings/` — 7

### Intermediate training checkpoints

`checkpoints/nemotron_lori/adapters/{math,code,science}/checkpoint-*/` total **~3 GB** of training-history checkpoints. Each ~29 MB. Only `best/` and `final/` are referenced by inference/benchmark scripts.

## Plan (executed in order)

### Step 1 — Pure deletes (recoverable from venv re-install)
- Project-level `__pycache__/` dirs only (NOT inside `.venv` or `perplexity-mcp-venv`)
- `pip.pid`
- `tmp/` contents (verify first that nothing important)

### Step 2 — Hardlink duplicate files
For each set of bit-identical files outside model/git/venv:
- Pick the canonical path
- Replace the others with hardlinks pointing to the same inode
- This frees disk without changing any file path. PEFT loading and benchmark scripts continue to work unchanged.

Notable groups:
- 32× Qwen tokenizer.json
- 16-21× Nemotron tokenizer/config files
- 3× adapter_model.safetensors (`hf_publish/* === checkpoints/.../best`)

### Step 3 — Comprehensive documentation generation
Generate four navigation documents:
- **`README.md`** (top-level) — restructure-aware project overview, points to canonical files for everything
- **`docs/RESULTS_INDEX.md`** — every results JSON catalogued with: file path, methodology, sample size, scoring caveats (especially flagging files that used the buggy v1 HumanEval extractor)
- **`docs/ADAPTERS_INDEX.md`** — every adapter catalogued: base model, training method (SFT/DPO/DARE), rank, file path
- **`docs/DOCS_INDEX.md`** — every markdown file catalogued by topic

### Step 4 — Annotate buggy-scoring artifacts
The HumanEval extraction bug affected files in `results/nemotron/master_results.json` and similar. Add a `_NOTE.md` next to each affected file explaining what the bug was and where the corrected numbers live (`overnight_run/findings/humaneval_n164.md`).

### Step 5 — Verify
- Run a quick smoke test (model load, generate one prompt) to confirm nothing broke
- Generate `_restructure/POST_RESTRUCTURE_REPORT.md` showing space savings + verification

## What I will NOT do

- Move/rename any directory referenced by code paths (would require updating dozens of scripts)
- Delete intermediate training checkpoints (preservation for reproducibility)
- Touch `.venv/`, `perplexity-mcp-venv/`, `.git/`, `models/`
- Push to git remote
- Rewrite git history

## Recovery

If anything breaks:
```bash
rm -rf /home/learner/Desktop/mewtwo
cp -r /home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP /home/learner/Desktop/mewtwo
```

Or single-file recovery:
```bash
cp /home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP/<path> /home/learner/Desktop/mewtwo/<path>
```

Or git revert to tag:
```bash
cd /home/learner/Desktop/mewtwo
git reset --hard pre-restructure-2026-05-02
```
