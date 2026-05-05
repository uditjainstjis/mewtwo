# Wake-up — 8-hour BFSI Swarm + Loop COMPLETE

**Last update:** 2026-05-03 ~05:35 UTC

## TL;DR — read these in order

1. **`docs/recent/LOOP_FINAL.md`** — full session summary with TWO honest negative findings flagged
2. **`SYNAPTA_PITCH_DECK.pdf`** — DECK ALREADY REGENERATED with new BFSI numbers (slides 1, 4, 6 updated)
3. **`docs/recent/CTO_PITCH_60SEC_V2.md`** — verbatim 60-second pitch
4. **`docs/recent/YC_APPLICATION_DRAFT.md`** — ~600-word YC narrative
5. **`docs/recent/DEMO_VOICEOVER.md`** — 90-sec script for screen-capture (when you're ready to record)

## Headlines for the pitch

- **+20.1 pp MBPP** (n=164, fixed extractor) ← NEW, biggest delta
- **+17.1 pp HumanEval** (n=164, p<0.001) ← existing
- **100% RBI document QA** (n=30 hand-curated) ← deployment proof
- **+6 pp FinanceBench** (n=50) ← industry-standard supporting

## Two honest findings to internalize

1. **BFSI v2 no-context recall: FG −20 pp vs base.** Format Guard hurts on out-of-distribution domain knowledge. Math/code adapters don't know RBI rules. **Reframe:** customer-trains-domain-adapters is the value-add for BFSI knowledge, generic adapters are for generic reasoning.
2. **RBI v1 = 100/100 both modes.** Context-injected QA too easy. Use as "deployment-ready" not "lift."

## To restart the demo

```bash
cd /home/learner/Desktop/mewtwo
nohup .venv/bin/python -m uvicorn synapta_src.src.demo.server:app --host 0.0.0.0 --port 7860 > logs/demo.log 2>&1 &
```

## To preview the new deck

```bash
xdg-open /home/learner/Desktop/mewtwo/SYNAPTA_PITCH_DECK.pdf
```

## Action checklist for today

- [ ] Read LOOP_FINAL.md
- [ ] Preview new SYNAPTA_PITCH_DECK.pdf
- [ ] Memorize CTO_PITCH_60SEC_V2.md
- [ ] Restart demo server
- [ ] Record 90-sec demo video using DEMO_VOICEOVER.md
- [ ] Send 5 BFSI cold-outreach LinkedIn messages
- [ ] Submit YC by May 4 23:59 PT using YC_APPLICATION_DRAFT.md
