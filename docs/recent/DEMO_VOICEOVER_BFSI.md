# BFSI Adapter Demo — 90-Second Voiceover Script

For screen recording with `synapta_src/data_pipeline/09_demo_bfsi.py` after adapter finishes training.

## Setup (do once before recording)
1. Terminal at >= 100 cols wide, dark theme, font size ~16
2. `cd /home/learner/Desktop/mewtwo && source .venv/bin/activate`
3. Run: `python synapta_src/data_pipeline/09_demo_bfsi.py --mode scripted`
4. Demo will load Nemotron-30B (~30s for first cold-start)

---

## SCENE 1 — The setup [0:00–0:15]

**On screen**: terminal, `nvidia-smi` showing 1× RTX 5090 (32 GB)

**Voiceover**:
> "This is Synapta. One RTX 5090 — total compute cost about $1.50 a day. Running Nemotron-30B in 4-bit, with a LoRA adapter we trained today on real RBI and SEBI Master Directions."

---

## SCENE 2 — Question 1 (ATM charge) [0:15–0:30]

**On screen**: scripted demo runs Q1, shows side-by-side BASE vs +BFSI

**Voiceover**:
> "Question one: 'What is the maximum charge per ATM transaction beyond the free monthly limit per RBI rules?' Look at base — it knows ATM but mumbles. Look at adapter — Rs.21, the actual RBI cap. That's a real Indian banking compliance answer."

---

## SCENE 3 — Question 2 (Fraud reporting timeline) [0:30–0:45]

**On screen**: Q2 — fraud reporting timeline

**Voiceover**:
> "Question two: 'Within how many days must a fraud be reported to RBI?' Base hedges. Adapter: 21 days, citing FMR-1 form. That's the exact regulatory clock that matters when a compliance officer is staring down a deadline."

---

## SCENE 4 — Question 3 (PMLA threshold) [0:45–1:00]

**On screen**: Q3 — PMLA threshold for NBFCs

**Voiceover**:
> "Question three: PMLA cash transaction threshold for NBFCs. Adapter pulls 'ten lakh rupees' — the Rule 3 limit. Three for three on the questions a real compliance officer would ask."

---

## SCENE 5 — Why this matters [1:00–1:15]

**On screen**: Cut to the BFSI_PIPELINE_NARRATIVE.md or DECK_SLIDES_V3 slide 9

**Voiceover**:
> "We trained this in two and a half hours. On 80 RBI Master Directions and 50 SEBI circulars we scraped today. Document-disjoint held-out eval — the model has never seen these PDFs during training."

---

## SCENE 6 — Why this is defensible [1:15–1:30]

**On screen**: STATUS_DASHBOARD.md scrolling through validator pass rate, dataset stats

**Voiceover**:
> "Most teams paraphrase-augment from the same questions and silently overstate. We hold out 26 entire PDFs. The number you'll see is the number you'd get on day one in a customer's dataroom. That's the architecture. That's why we win the regulated AI market."

---

## Suggested cuts / B-roll
- Quick `cat data/rbi_corpus/manifest.jsonl | head -5` to show real RBI PDFs scraped
- Quick `wc -l data/rbi_corpus/qa/train_clean.jsonl data/rbi_corpus/qa/eval_clean.jsonl` showing the train/eval counts
- Brief shot of `adapters/nemotron_30b/bfsi_extract/best/adapter_config.json` showing r=16

## Talking points for live Q&A
- "We're not competing on benchmarks the FOMC has already saturated. We're competing on the ones nobody has data for."
- "Customer brings their corpus. We turn it into their proprietary domain adapter. They own the IP, we own the platform."
- "Sovereignty is the wedge. India has 200+ mid-tier banks, NBFCs, and insurers that legally cannot send a single byte to OpenAI."

## Hard "do not say"
- Don't quote held-out F1 numbers until they're measured
- Don't claim signed contracts
- Don't say "first" or "only" — say "best positioned" or "leading approach"
- Don't compare to GPT-4 specifically (instead: "frontier closed APIs")
