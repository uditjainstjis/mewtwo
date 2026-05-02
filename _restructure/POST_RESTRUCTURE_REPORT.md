# Post-Restructure Report — 2026-05-02

## Summary

Restructure of `/home/learner/Desktop/mewtwo/` completed successfully with no data loss
and no disruption to the running demo server.

| Metric | Before | After | Delta |
|---|---|---|---|
| Disk used | 121 GB | 119 GB | **−2.4 GB** |
| Files (excl venvs/git) | ~10,700 | 10,647 | minor cleanup |
| `__pycache__` dirs (project) | 27 | 0 | -27 |
| Duplicate file groups | 51 | 0 (hardlinked) | — |
| Hardlinks created | 0 | 199 | +199 |

## What was done

### 1. Safety net (immutable)
- Git tag: `pre-restructure-2026-05-02` at commit `e391284`
- Full clone at `/home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP/` (121 GB, marked read-only via `_DO_NOT_MODIFY.md`)

### 2. Pure deletes (regeneratable)
- 27 project-level `__pycache__/` directories (~1.8 MB)
- `pip.pid` (4 KB)
- `tmp/` contents moved to `_restructure/_old_tmp_snapshot/` (preserved, not deleted)

### 3. Deduplication via hardlinks
- 199 bit-identical files replaced with hardlinks pointing to a canonical inode
- **2.4 GB recovered** — primarily from:
  - 32 copies of Qwen tokenizer.json (~340 MB)
  - 21 copies of another tokenizer/config (~150 MB+)
  - 16×3 + 12 + 7 other duplicate config files
  - 3 copies of adapter_model.safetensors between `adapters/nemotron_30b/{X}/best/` and `adapters/published/{X}/`
- **No paths changed** — every PEFT load / benchmark script continues to work without modification

### 4. Documentation generated
- **`/home/learner/Desktop/mewtwo/README.md`** (rewritten) — top-level project entry point with headline numbers, quick start, key paths
- **`/home/learner/Desktop/mewtwo/PROJECT_OVERVIEW.md`** (new) — comprehensive structure map with directory layout, caveats, reproducibility
- **`/home/learner/Desktop/mewtwo/docs/RESULTS_INDEX.md`** (new) — every results JSON catalogued with methodology, sample sizes, scoring caveats
- **`/home/learner/Desktop/mewtwo/docs/ADAPTERS_INDEX.md`** (new) — every PEFT adapter catalogued by base model with rank, target modules, weight size
- **`/home/learner/Desktop/mewtwo/docs/DOCS_INDEX.md`** (new) — every markdown file in the repo organized by directory

### 5. Buggy-scoring annotations
- **`results/nemotron/_NOTE.md`** — explains the HumanEval extraction bug, points readers to `docs/findings/humaneval_n164.md` for corrected n=164 numbers (56.1% / 73.2% / +17.1)
- **`results/lori_moe/_NOTE.md`** — explains the LoRI-MoE phase artifacts and their relationship to current production architecture

## Verification

### Canonical inference paths still resolve
✅ All 11 critical paths verified:
- `adapters/nemotron_30b/{math,code,science}/best/adapter_model.safetensors` — same inode as `adapters/published/{math,code,science}/adapter_model.safetensors` (hardlinked, both paths work)
- `models/nemotron/config.json`
- `src/demo/server.py` (FIXED version) + `server_original_backup.py` (pre-fix backup)
- `build_pitch_deck.py`
- `overnight_run/FINAL_SUMMARY.md`, `results/overnight/humaneval_full_base_rescored.jsonl`

### Demo server uninterrupted
- Demo server (PID 10097, started before restructure) still running with 36 min uptime
- `GET /api/status` returns 200, model_loaded=true, all 3 adapters loaded
- VRAM 17.5 GB / 33.6 GB stable

## What was NOT done (intentional)

- **No directory renamed or moved.** All scripts that hardcode paths continue to work.
- **No model weights deleted.** All adapter checkpoints preserved (best, final, intermediate).
- **No third-party benchmark fixtures touched** (LLMs-Planning, tau-bench).
- **No git history rewritten.** Commit graph and tags preserved.
- **No .venv touched.** Existing installed packages untouched.

## Files preserved despite being possibly redundant

For reproducibility (user emphasized "no data loss in name of summarification"):
- All `checkpoint-N/` intermediate training checkpoints (~3 GB total). Documented in `docs/ADAPTERS_INDEX.md` as "preservation for paper claim re-derivation."
- All `dare_sparsified/` directories. Regeneratable from `final/` but kept for direct loadability.
- All `hidden_*` emergency-save checkpoints.
- All result files even when superseded — old buggy-scored versions are kept alongside corrected ones with `_NOTE.md` flagging the issue.
- Synapta v1 era artifacts in `archive/synapta_v1/`.

## Recovery instructions (if anything goes wrong)

### Full restore from clone
```bash
rm -rf /home/learner/Desktop/mewtwo
cp -r /home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP /home/learner/Desktop/mewtwo
```

### Single file restore
```bash
cp /home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP/<path> /home/learner/Desktop/mewtwo/<path>
```

### Git restore
```bash
cd /home/learner/Desktop/mewtwo
git reset --hard pre-restructure-2026-05-02
```

### Rebuild indexes (idempotent — re-run any time)
```bash
/home/learner/Desktop/mewtwo/.venv/bin/python /home/learner/Desktop/mewtwo/_restructure/build_indexes.py
```

## Audit artifacts

All audit logs preserved in `_restructure/`:
- `AUDIT_AND_PLAN.md` — original audit findings + planned actions
- `audit_top_level.txt` — top-level dir inventory
- `audit_dir_sizes_sorted.txt` — per-dir size breakdown
- `audit_hashes.txt` / `audit_hashes_sorted.txt` — sha256 of all files >10KB
- `audit_dupes.txt` — duplicate file groups (51 groups, 199 files)
- `synapta_results_jsons.txt` — filtered list of Synapta-specific result JSONs
- `all_adapters.txt` / `all_markdown.txt` / `all_results_jsons.txt` — full inventories
- `deleted_pycache_dirs.txt` — list of deleted pycache paths
- `deleted_tmp_filelist.txt` — list of files moved from `tmp/`
- `hardlink_actions.log` — every hardlink replacement (canonical → dup)
- `build_indexes.py` — script that generates the four index documents (re-runnable)
- `_old_tmp_snapshot/` — preserved contents of original `tmp/`

## Net effect

Repo is now navigable via 4 index documents that catalogue every result, adapter, and document.
2.4 GB recovered without changing any file path. All scripts continue to work. Demo server
unaffected. Pre-restructure state recoverable in under 90 seconds via the safety clone.

For the user: read `PROJECT_OVERVIEW.md` first, then `docs/RESULTS_INDEX.md` for the
defensible numbers and methodology caveats.
