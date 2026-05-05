# Advisor Deductions — perplexity research → YC strategy

**Date:** 2026-05-04. Source: stronger-reviewer advisor seeing full session context.

---

## The unspoken finding: IndiaFinBench is bigger than perplexity

At ~30% on a benchmark where Gemini Flash hits 89.7%, Qwen3-32B 85.5%, LLaMA-3.3-70B 83.7% — Synapta is **60 pp below frontier** on a public Indian-BFSI benchmark. That's the load-bearing data point of this entire session. Not a footnote about "task-style transfer."

**The "Synapta beats/matches frontier on Indian BFSI" pitch is dead.** Any YC partner with NLP literacy who sees a 60 pp public-benchmark gap will discount the original framing.

### The pivot to make

> "Synapta is a per-customer methodology, not a frontier model. The +30.9 pp lift is on the customer's own corpus. On out-of-distribution public benchmarks we score ~30% — confirming what every NLP paper since 2018 shows: fine-tuned adapters don't generalize OOD. **That's exactly why customers need adapters trained on THEIR documents, not a general 'Indian BFSI AI'.** OnFinance markets one general LLM; we ship a methodology that produces N customer-specific ones."

Make the bad number into the *thesis*.

---

## Q3 market reality forces a GTM reckoning

**₹1-3 cr Y1, 9-18 mo procurement, 80% audit rejection for sub-3-year vendors, zero LOIs.** Selling direct to mid-tier banks Y1 will fail.

Pick ONE distribution wedge before submitting:

1. **Consultancy channel** — sell methodology to E&Y/KPMG/PwC India; their 3-year track record substitutes for ours. (Most realistic.)
2. **RegTech-to-regulator** — pitch RBI/SEBI directly as their internal QA tool. Sandbox graduate path.
3. **Friendly-CXO entry** — anyone in network at Tier-2 NBFC/fintech who'd sign a ₹50L paid pilot. **One LOI changes the YC conversation entirely.**
4. **Open-source the methodology, monetize the per-customer training service** — Cohere/Anthropic-style developer flywheel for Indian BFSI.

"We'll sell to mid-tier banks directly" is what every applicant writes; Q3 says it doesn't work.

---

## Q1 OnFinance gaps — real but narrow

OnFinance: 70+ agents, $4.2M Peak XV, "India's first BFSI LLM," 300M tokens. Real product surface area.

- **Sovereign/on-prem** — only a moat if actually demonstrated in air-gapped config. Have we?
- **Methodology transparency** — real but YC partners care more about traction than rigor.
- **IRDAI/PFRDA/FEMA depth** — **we don't have these built.** Frame as roadmap, not coverage.
- **Drift handling** — design doc only. Scope: "Component 1 ships within 30 days of YC."

---

## What to CUT from YC application

1. Any cross-benchmark "comparable to Gemini" framing. Even hedged.
2. Multi-regulator claims including IRDAI/PFRDA/FEMA — no data yet.
3. Any revenue projection > ₹3 cr Y1.
4. "70+ AI agents" implicit comparison (we have 2 adapters; don't anchor a losing comparison).

## What to ADD

1. **OOD-degradation thesis as central technical claim.** IndiaFinBench is the proof.
2. **Pick a distribution wedge** (1 of 4 above) and commit in writing.
3. **Concrete near-term LOI plan** — "Close 1 paid pilot at ₹25-50L in 60 days from contact list of 12 NBFCs/fintechs in my network."
4. **Honest revenue ladder:** Y1 ₹1-3 cr (1-2 pilots) · Y2 ₹5-10 cr (sandbox graduation) · Y3 ₹15-30 cr (5+ customers). Sober numbers earn more trust than $10M ARR projections.
5. **Explicit 80% audit rejection rate** + mitigation: "Not competing with TCS/Infosys for major banks Y1. Wedge: RBI sandbox + Tier-2 NBFCs + consultancy partnerships."

---

## Tonight's two implementation actions

1. **Write the OOD-pivot paragraph.** Replace "Synapta vs Gemini" with per-corpus thesis. **Highest-leverage edit.**
2. **12-name LOI outreach list.** Anyone ever met at an Indian financial company. One signed LOI by submission = biggest lever for Y1 revenue credibility. If impossible by submission: "12-name outreach in motion" as concrete near-term action.

---

## What is and isn't blocking

**Blocking:**
- Gemini-Flash comparison anywhere in YC app → sophisticated readers will discount the rest.
- Revenue projections inconsistent with ₹1-3 cr Y1 reality.

**Not blocking, high-EV:** one LOI by submission.

**Not blocking:** more perplexity research, more benchmark runs, more synthesis docs. We have enough data. Bottleneck now = honest pitch reframing + one human ask (LOI outreach).

**Eval finishing in ~50 min won't change these deductions.** Don't wait for it to start the YC-app rewrite.
