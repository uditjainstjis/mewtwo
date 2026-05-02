# Synapta — Sovereign AI inference platform

Adapter-routing platform for regulated enterprise. Single base model + dozens of swappable
domain experts. Deployed inside customer firewall, fully air-gapped.

## Headline numbers (n=164, p<0.001)

| Benchmark | Base Nemotron-30B | Format Guard routing | Lift |
|---|---|---|---|
| ARC-Challenge | 20.0% | 31.0% | +11.0 |
| MATH-500 | 41.5% | 56.0% | +14.5 |
| **HumanEval** (rescored v2) | **56.1%** | **73.2%** | **+17.1** |
| MBPP | 8.0% | 5.0% | -3.0 |

Plus: **Code Paradox** at n=200 on Nemotron-30B — code-trained adapter beats math-trained
adapter on math reasoning by +5.5 pp. The strongest single defensible novel finding.

## Quick start

```bash
# Activate venv
source .venv/bin/activate

# Launch the demo (FastAPI + WebSocket on :7860)
python -m uvicorn src.demo.server:app --host 0.0.0.0 --port 7860

# Open in browser
xdg-open http://localhost:7860
```

## Read first

1. **[`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)** — comprehensive structure map
2. **[`overnight_run/FINAL_SUMMARY.md`](overnight_run/FINAL_SUMMARY.md)** — recent mission output
3. **[`overnight_run/TALKING_POINTS.md`](overnight_run/TALKING_POINTS.md)** — verbatim sentences for CTO meeting + YC application
4. **[`overnight_run/DECK_UPDATE_GUIDE.md`](overnight_run/DECK_UPDATE_GUIDE.md)** — exact deck edits
5. **[`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md)** — every results JSON catalogued, with caveats
6. **[`docs/ADAPTERS_INDEX.md`](docs/ADAPTERS_INDEX.md)** — every PEFT adapter catalogued
7. **[`docs/DOCS_INDEX.md`](docs/DOCS_INDEX.md)** — every markdown file in the repo

## Key research artifacts

- **Pitch deck:** `SYNAPTA_PITCH_DECK.pptx` / `SYNAPTA_PITCH_DECK.pdf` (8 slides, 16:9)
- **Manuscripts:** `manuscripts/synapta_systems.md`, `manuscripts/code_paradox.md`
- **Master research docs:** `docs/MASTER_KNOWLEDGE_BASE.md`, `docs/MASTER_RESEARCH_CHRONICLES.md`, `docs/MASTER_EXPERIMENT_REPORTS.md`

## Restructure history

This repo was reorganized on 2026-05-02 to consolidate documentation and deduplicate
redundant files. See `_restructure/AUDIT_AND_PLAN.md` for the full audit, and
`_restructure/POST_RESTRUCTURE_REPORT.md` for what changed. Pre-restructure git tag:
`pre-restructure-2026-05-02`. Full pre-restructure backup at
`/home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP/`.

## License

See individual model licenses (NVIDIA Nemotron, Qwen) and `requirements.txt` for dependencies.
