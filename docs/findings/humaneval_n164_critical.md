# HumanEval n=164 — CRITICAL course correction

**Status:** still running for format_guard mode at iter time of write.
**Important:** numbers below are PARTIAL (n=107 of 164 in base mode). Final numbers in next iter.

## What changed

I caught **two layered scoring bugs** in this session:

1. **First bug (iter 4):** `extract_code` was dropping `from typing import List` from prompts, causing `NameError` on every type-hinted task.

2. **Second bug (iter 5):** When the model returned a code block containing JUST a function body (e.g., `    return [s for s in strings if substring in s]`), my v2 rescorer was calling `.strip()` and removing the leading indentation — making the body parse as a top-level statement instead of as a function body. **This is the bigger bug.**

After fixing both, the same model outputs that scored 25% now score **65% pass@1**.

## Numbers (partial, n=107 of base so far)

| Method | Pass@1 inline (v1, buggy) | Pass@1 rescored (v2, fixed) |
|---|---|---|
| Base Nemotron-30B (in-progress) | 25.2% | **65.4%** |
| Format Guard | TBD (mode pending) | TBD |

**Improved cases:** 44. **Worsened cases:** 1. Net delta: **+43 passes** out of 107.

## Implications for the deck and YC pitch

This is a MAJOR course correction — the truth needs to land in the deck.

### The deck's "Format Guard +24 vs base" claim

Sourced from `grand_comparison_v2_results.json` n=25 (base 24%, FG 48%). At n=25 the small sample plus the same buggy extractor gave numbers that *internally* showed +24 delta. But:

- The base ~24% is largely a **scoring artifact**, not a model limit
- True base is ~65%
- Format Guard partial spotcheck (10 problems before kill) showed similar pattern
- Need to wait for full FG rescored numbers before final delta

### Three possible outcomes for the full picture

**Scenario A: Format Guard rescores at ~65% too** → no benefit. Deck claim falsified at scale. Need to drop the +24 line entirely. Story shifts to: "the routing system matches base on accuracy and beats it on efficiency (35% shorter outputs, 35% faster)" — which IS a defensible story.

**Scenario B: Format Guard rescores at ~70-75%** → modest +5-10 advantage. Reframe deck as "5-10 points HumanEval lift at full benchmark scale, plus the 35% efficiency gain."

**Scenario C: Format Guard rescores at ~80%+** → +15-20 advantage holds. Deck claim survives with smaller magnitude.

I'll know in ~60 min when the full run completes.

## What this means right now for the user

**Honest take:** the existing deck claim of base 24% / FG 48% / +24 delta was internally self-consistent on n=25 but rests on a buggy scoring pipeline. At n=164 with corrected scoring, the absolute base number is much higher. The delta might shrink or disappear.

**This is recoverable for YC:**
- The Code Paradox finding (Code adapter > Math adapter on MATH-500, n=200) is unaffected by this bug
- The token routing efficiency finding (35% shorter / faster) is unaffected
- The "demo works at 95% on smoke tests" finding is unaffected
- The base capability is actually *higher* than claimed — that's strictly good for the company narrative ("our deployment runs a 65% HumanEval model, not a 24% one")

**This is recoverable for the paper:**
- A NeurIPS reviewer would have caught this bug. Catching it now (May 2) gives time to revise.
- The honest framing: "We initially measured at n=25 with a CoT-output extractor that under-counted body-only completions. After scoring on n=164 with code-block-aware extraction, base = X%, format_guard = Y%, delta = Z%."

I will write `humaneval_n164.md` with the full corrected numbers as soon as both modes finish.
