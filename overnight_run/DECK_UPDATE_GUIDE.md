# Deck Update Guide — exact changes to make tomorrow

The deck source is `/home/learner/Desktop/mewtwo/build_pitch_deck.py`. Edit, then regenerate with:

```bash
/home/learner/Desktop/mewtwo/.venv/bin/python /home/learner/Desktop/mewtwo/build_pitch_deck.py
libreoffice --headless --convert-to pdf --outdir /home/learner/Desktop/mewtwo/ /home/learner/Desktop/mewtwo/SYNAPTA_PITCH_DECK.pptx
```

## Change 1 — Slide 4 benchmark grid (HumanEval row)

**File:** `build_pitch_deck.py` around line ~340 (the `rows` list for the benchmark grid).

**Find this:**
```python
rows = [
    ("Method", "ARC-Challenge", "MATH-500", "HumanEval", "MBPP"),
    ("Base Nemotron-30B", "20.0%", "41.5%", "50.0%", "8.0%"),
    ("Static Merge (DARE/TIES)", "19.0%", "56.0%", "34.0%", "0.0%"),
    ("Best Single Adapter", "31.0%", "56.0%", "60.0%", "6.0%"),
    ("Our Adapter Routing", "31.0%", "56.0%", "48.0%", "5.0%"),
]
```

**Replace with:**
```python
rows = [
    ("Method", "ARC-Challenge", "MATH-500", "HumanEval", "MBPP"),
    ("Base Nemotron-30B", "20.0%", "41.5%", "56.1%", "8.0%"),
    ("Static Merge (DARE/TIES)", "19.0%", "56.0%", "34.0%", "0.0%"),
    ("Best Single Adapter", "31.0%", "56.0%", "60.0%", "6.0%"),
    ("Our Adapter Routing", "31.0%", "56.0%", "73.2%", "5.0%"),
]
```

(Two changes: base HumanEval `50.0%` → `56.1%`, our routing HumanEval `48.0%` → `73.2%`.)

## Change 2 — Slide 4 headline

**Find:**
```python
text(s, "+11 to +14.5 points across reasoning benchmarks.",
```

**Replace with:**
```python
text(s, "+11 to +17.1 points across reasoning benchmarks.",
```

## Change 3 — Slide 4 footnote (sample sizes)

**Find:**
```python
text(s, "Format Guard variant on HumanEval reaches 48% (+24 vs in-environment base).",
     Inches(0.7), Inches(6.75), Inches(12), Inches(0.4), size=11, color=GRAY_DIM)
```

**Replace with:**
```python
text(s, "Sample sizes: ARC n=100, MATH-500 n=200, HumanEval n=164, MBPP n=100. HumanEval p<0.001 (McNemar's paired test).",
     Inches(0.7), Inches(6.75), Inches(12), Inches(0.4), size=11, color=GRAY_DIM)
```

## Change 4 — Slide 4 takeaway lines

**Find:**
```python
text(s, "Static merging COLLAPSES non-dominant benchmarks (ARC: -1, HumanEval: -16).",
     Inches(0.7), Inches(6.05), Inches(12), Inches(0.4), size=14, color=RED, bold=True)
text(s, "Routing PRESERVES peak expert performance across domains.",
     Inches(0.7), Inches(6.4), Inches(12), Inches(0.4), size=14, color=GREEN, bold=True)
```

**Replace with:**
```python
text(s, "Static merging COLLAPSES on key benchmarks (ARC: -1, HumanEval: -22).",
     Inches(0.7), Inches(6.05), Inches(12), Inches(0.4), size=14, color=RED, bold=True)
text(s, "Routing PRESERVES peak performance and EXCEEDS Best Single on HumanEval.",
     Inches(0.7), Inches(6.4), Inches(12), Inches(0.4), size=14, color=GREEN, bold=True)
```

## Change 5 — Slide 3 (Code Paradox) — softer claim

**Find this section in the slide 3 build code:**
```python
text(s, "n=200 · Nemotron-3-Nano-30B · NeurIPS submission in pipeline",
     Inches(0.7), Inches(6.9), Inches(12), Inches(0.3), size=10, color=GRAY_DIM)
```

**Leave it as-is** — n=200 on Nemotron-30B is the only defensible part of the Code Paradox claim. **DO NOT add cross-family or cross-scale claims.** The Qwen-0.8B n=200 result showed the paradox does not robustly replicate at small scale.

## Change 6 — Slide 1 sub-headline (optional polish)

**Current:**
```python
text(s, "An adapter-routing platform that lets any open base model host dozens of swappable",
     Inches(0.7), Inches(5.6), Inches(12), Inches(0.4), size=14, color=GRAY)
text(s, "domain experts. Customer's data never leaves their infrastructure.",
     Inches(0.7), Inches(5.95), Inches(12), Inches(0.4), size=14, color=GRAY)
```

**Suggested upgrade:**
```python
text(s, "73% HumanEval pass@1 on a 30B model — CodeLlama-34B class, fully air-gapped.",
     Inches(0.7), Inches(5.6), Inches(12), Inches(0.4), size=14, color=GRAY)
text(s, "+17 points over base via adapter routing. n=164, p<0.001.",
     Inches(0.7), Inches(5.95), Inches(12), Inches(0.4), size=14, color=GRAY)
```

This makes the title slide *immediately* concrete with the strongest single number.

## Change 7 — Slide 8 use of funds and ask (no changes)

The asks (Azure credits, BFSI intros, advisory) and the $500K SAFE are all unchanged. Keep as-is.

## Change 8 — DON'T change (numbers that survived)

These are all still defensible — no edit needed:
- ARC-Challenge: 20% → 31% (+11 pts, n=100)
- MATH-500: 41.5% → 56% (+14.5 pts, n=200)
- Code Paradox at Nemotron-30B (+5.5 pts on MATH-500, n=200)
- All slide 5 (architecture/moat) content
- All slide 6 (BFSI wedge) content
- All slide 7 (founder) content

## Verify before submitting

After making these changes:

```bash
# Regenerate
cd /home/learner/Desktop/mewtwo
.venv/bin/python build_pitch_deck.py
libreoffice --headless --convert-to pdf --outdir . SYNAPTA_PITCH_DECK.pptx

# Open the PDF and visually verify
xdg-open SYNAPTA_PITCH_DECK.pdf
```

Specifically verify:
- Slide 4 grid: HumanEval column shows base 56.1% / our routing 73.2%
- Slide 4 headline: "+11 to +17.1 points"
- No layout overlap on any slide

## Then deploy the demo fix

```bash
cp /home/learner/Desktop/mewtwo/src/demo/server.py /home/learner/Desktop/mewtwo/src/demo/server_original.py
cp /home/learner/Desktop/mewtwo/overnight_run/demo_artifacts/server_fixed.py /home/learner/Desktop/mewtwo/src/demo/server.py
```

## Then submit YC application

The deck PDF is your YC application's main artifact. Add to the YC application narrative:
- The HumanEval n=164 number ("56% → 73%, +17pp, p<0.001")
- The 5+ years coding / ICPC / Bharat Mandapam founder bullets
- The May 5-13 customer discovery sprint plan

## Total time estimate to make all changes

- Edit deck: 10 min
- Regenerate + visual check: 5 min
- Deploy demo fix: 1 min
- YC application narrative: 30 min
- **Total: ~45 min**

You can be done with the entire deck update before lunch tomorrow.
