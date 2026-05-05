# Synapta launch thread — 9am IST May 4 2026

8 tweets, each ≤ 280 chars. Tag @ycombinator on tweet 1 or 8 (8 chosen so the hook isn't diluted).

---

**Tweet 1  /  the hook (130 chars)**

We trained a 30B model in 3.5h on $1.50/day of GPU to crush Indian banking regulation Q&A by +31.3 pp on a held-out corpus.

Receipts in thread.

---

**Tweet 2  /  the methodology insight (271 chars)**

Most "domain-fine-tuned LLM" results are bogus because the eval questions are LLM-paraphrases of documents the model trained on. Data leakage with extra steps.

We held out 26 entire PDFs (20% of corpus). The eval model literally never saw those documents. Document-disjoint.

---

**Tweet 3  /  the table (252 chars)**

Held-out RBI + SEBI extractive QA, n=595 paired:

```
            base    +bfsi   lift
substring   58.3%   89.6%   +31.3pp
95% CI      54-62   87-92   disjoint
```

Same model, same 4-bit quant. Only difference: the LoRA adapter.

[attach screenshot of full eval table]

---

**Tweet 4  /  significance + ratio (255 chars)**

McNemar paired test on the 2×2 contingency:

p = 6.26 × 10⁻⁴⁴

199 questions adapter-only-correct.
13 questions base-only-correct.
15× improvement-to-regression ratio.

The adapter genuinely learned regulatory knowledge. Not noise. Not paraphrase memorization.

---

**Tweet 5  /  per-tier breakdown (266 chars)**

The lift is biggest where it matters most:

· Tier 2 (numeric — Rs, days, %): +24.8 pp
· Tier 3 (heading-extractive paragraphs): +39.1 pp

Per regulator: RBI +31.8 pp, SEBI +29.6 pp.

The adapter learned the shape of Indian financial regulation, not one corpus.

---

**Tweet 6  /  cost + time (224 chars)**

Hardware: one RTX 5090 (32 GB).
Quantization: 4-bit NF4.
LoRA: r=16, α=32, attn + MLP.
Trainable params: 434.6M out of 32B (1.36%).
Wall clock: 3h 28min, 174 update steps.
GPU cost: $1.50/day amortized.

This is fundable on a credit card.

---

**Tweet 7  /  open pipeline (248 chars)**

The full pipeline is reproducible end-to-end in under 8 hours from a clean clone:

scrape → extract → chunk → 3-tier deterministic QA (no LLM-generated questions) → 10-check validator → LoRA train → held-out eval

github.com/uditjain/mewtwo  ← repo (will be public May 4)

---

**Tweet 8  /  the why + the ask (270 chars)**

Indian BFSI can't send data to OpenAI / Anthropic. RBI localization + DPDP block it for ~70% of enterprise data.

Synapta is the inference layer that runs on-prem and the customer extends with their own corpus.

CTOs at Indian banks / NBFCs / insurers — DM me. @ycombinator

---

## POSTING INSTRUCTIONS

- **Time:** 9:00 AM IST, 4 May 2026 (= 03:30 UTC, = 20:30 PT on 3 May).
  Rationale: catches Indian BFSI working hours plus US West Coast evening (when YC partners scroll).
- **Platform:** Twitter / X. Post as a single thread (not separate tweets); use the "Add another post" button so they chain.
- **Visual on Tweet 3:** attach a screenshot of the held-out eval table (use the markdown table from `BFSI_ADAPTER_FINAL.md` — or render in Carbon for a clean look). Do NOT use a stock chart.
- **4 hours after Tweet 1 lands** (= 1:00 PM IST = 7:30 AM PT): retweet Tweet 1 with a short quote-tweet ("Update: thread is getting traction — the held-out methodology section is the part to read") to catch the US West-Coast morning audience for the YC demographic.
- **Reply to your own thread (≤ 1 hour after Tweet 8)** with two follow-ups:
  1. Link to the GitHub repo (full pipeline, README with reproduce-in-8-hours block).
  2. Link to the Loom demo video (BFSI adapter answering 3 held-out questions live, ~90 seconds).
- **Tag policy:** do NOT @ YC partners individually. Tagging @ycombinator once on the final tweet is the correct etiquette for a YC W26 applicant.
- **Engagement plan for first 4 hours:**
  · Reply individually to every BFSI / regulated-AI engineer who comments — that's the funnel.
  · Pin Tweet 1 to your profile for the next 7 days.
  · Cross-post the README link (not the thread itself) to r/LocalLLaMA and Hacker News later in the day if traction is strong.
- **Do NOT:** edit tweets after posting (kills threading on some clients). Do NOT add a 9th tweet later — keep the thread closed and clean.
