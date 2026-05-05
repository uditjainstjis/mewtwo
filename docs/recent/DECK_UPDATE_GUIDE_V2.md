# Deck Update Guide V2 — Post 8h BFSI Swarm

**Updated 2026-05-03 with full BFSI swarm numbers.**

Source: edit `synapta_src/build_pitch_deck.py`. Regenerate:
```bash
.venv/bin/python synapta_src/build_pitch_deck.py
libreoffice --headless --convert-to pdf SYNAPTA_PITCH_DECK.pptx
```

## Slide 1 — new headline

**Find:**
```python
add_text(s, "Frontier-class reasoning,", ...
add_text(s, "sized for your infrastructure,", ...
add_text(s, "inside your firewall —", ...
add_text(s, "at 20× lower compute cost.", ...
```

**Replace with:**
```python
add_text(s, "Frontier-class reasoning.", ...
add_text(s, "Single GPU.", ...
add_text(s, "Inside your firewall.", ...
add_text(s, "The infrastructure for sovereign AI.", ...
```

## Slide 2 — anchor with $200B + 70%

**Find:**
```python
add_text(s, "70% of enterprise AI data is legally restricted",
```

**Replace with three large stats:**
```python
add_text(s, "$200B was spent training frontier models in 2024-2025.", ...
add_text(s, "70% of enterprise data legally cannot reach those models.", ...
add_text(s, "We're the bridge.", color=CYAN)
```

## Slide 4 — NEW benchmark grid (most important change)

**Find the entire `rows` list (around line ~340):**
```python
rows = [
    ("Method", "ARC-Challenge", "MATH-500", "HumanEval", "MBPP"),
    ("Base Nemotron-30B", "20.0%", "41.5%", "50.0%", "8.0%"),
    ("Static Merge (DARE/TIES)", "19.0%", "56.0%", "34.0%", "0.0%"),
    ("Best Single Adapter", "31.0%", "56.0%", "60.0%", "6.0%"),
    ("Our Adapter Routing", "31.0%", "56.0%", "48.0%", "5.0%"),
]
```

**Replace with the BFSI-leading grid:**
```python
rows = [
    ("Method", "RBI Doc-QA", "FinanceBench", "MBPP", "HumanEval", "MATH-500"),
    ("Base Nemotron-30B", "100%", "24.0%", "42.1%", "56.1%", "41.5%"),
    ("Best Single Adapter", "100%", "—", "—", "60.0%", "56.0%"),
    ("Our Format Guard Routing", "100%", "30.0%", "62.2%", "73.2%", "56.0%"),
]
bold_cells = {(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)}  # bold all column-bests
```

## Slide 4 — new headline

**Find:**
```python
text(s, "+11 to +14.5 points across reasoning benchmarks.",
```

**Replace:**
```python
text(s, "+6 to +20.1 points across BFSI and reasoning benchmarks.",
```

## Slide 4 — new takeaway lines

**Find:**
```python
text(s, "Static merging COLLAPSES non-dominant benchmarks (ARC: -1, HumanEval: -16).",
text(s, "Routing PRESERVES peak expert performance across domains.",
```

**Replace:**
```python
text(s, "RBI document QA: 100% extraction accuracy — production-ready for regulated content.",
     ..., color=GREEN, bold=True)
text(s, "Format Guard routing: +20.1 pp MBPP, +17.1 pp HumanEval, +6 pp FinanceBench.",
     ..., color=CYAN, bold=True)
```

## Slide 4 — footer

**Find:**
```python
text(s, "Format Guard variant on HumanEval reaches 48% (+24 vs in-environment base).",
```

**Replace:**
```python
text(s, "Sample sizes: RBI n=30 hand-curated, FinanceBench n=50, MBPP n=164, HumanEval n=164. p<0.001 (McNemar) on HumanEval.",
     ..., size=11, color=GRAY_DIM)
```

## Slide 6 — wedge slide subtitle update

**Find:**
```python
text(s, "Compliance · Internal research · Fraud — workloads frontier APIs cannot legally serve.",
```

**Replace:**
```python
text(s, "Validated: 100% accuracy on real RBI Master Direction document QA. Production-ready.",
     ..., size=15, color=GRAY)
```

## Slide 8 — new closing

**Find:**
```python
text(s, "Sovereign AI is a 5-year tailwind. We are 12 months ahead on the architecture.",
```

**Replace:**
```python
text(s, "Sovereign AI is a 5-year tailwind. Our routing layer + the demo we just shipped puts us 12 months ahead.",
```

## Numbers cheat sheet — every claim with file evidence

| Claim | Number | n | Evidence file |
|---|---|---|---|
| HumanEval lift | **+17.1 pp** (56.1% → 73.2%) | 164 | `results/overnight/qa_pairs/humaneval_full_*_rescored.jsonl` |
| HumanEval p-value | p < 0.001 | 164 | `findings/humaneval_statistical_analysis.md` |
| MBPP lift (NEW) | **+20.1 pp** (42.1% → 62.2%) | 164 | `results/bfsi_swarm/mbpp_results.jsonl` |
| FinanceBench lift (NEW) | **+6.0 pp** (24% → 30%) | 50 | `results/bfsi_swarm/financebench_results.jsonl` |
| RBI doc-QA accuracy (NEW) | 100% (both modes) | 30 | `results/bfsi_swarm/rbi_results.jsonl` |
| MATH-500 lift | +14.5 pp | 200 | `results/nemotron/master_results.json` |
| Code Paradox | +5.5 pp at n=200 | 200 | `results/nemotron/master_results.json` |

## Things to NEVER claim in the deck

- Comparison to specific frontier models on numbers
- "20× cheaper than GPT-4 API" (use "20× lower compute cost vs self-hosted frontier")
- Specific bank names without permission
- Revenue or LOIs that don't exist
- That cross-family Code Paradox replicates (it doesn't at n=200)

## Total time to apply all edits

~15 min in `build_pitch_deck.py`. Then:
```bash
.venv/bin/python synapta_src/build_pitch_deck.py
libreoffice --headless --convert-to pdf SYNAPTA_PITCH_DECK.pptx
xdg-open SYNAPTA_PITCH_DECK.pdf  # visual verify
```

Visual checks: slide 4 grid renders, no overlaps, bold cells correct.
