# Perplexity Research Synthesis — 2026-05-04

Three queries × 2 modes (auto + deep_research). The DEEP versions below are the primary read; auto versions retained as `q*.AUTO.txt` for cross-reference. Raw deep JSON in `q*.deep.json`.

---

## Q1 — OnFinance differentiation (DEEP)

**OnFinance position:** NeoGPT marketed as "India's first BFSI-specific LLM" trained on 300M+ tokens (SEBI/RBI/IRDAI/AMFI). 70+ AI agents for circular→workflow conversion, deadline tracking, audit docs. Cloud-first (Peak XV Surge backed). No on-prem claims, no LoRA-on-open-models, no published methodology metrics.

**Specific gaps Synapta can occupy:**

1. **Sovereign / on-prem** — OnFinance is implicitly cloud-first; no public on-prem positioning. Indian BFSI data localization (DPDP Act 2023, RBI Cyber Framework 2024) make this critical. **Synapta's "your hardware, your weights" wedge is real and unoccupied.**
2. **Methodology transparency** — OnFinance publishes no held-out evaluation, no statistical significance tests, no paired-comparison data. Synapta's McNemar p < 10⁻⁴⁸ on 664 paired questions is the single sharpest differentiator.
3. **IRDAI depth** — OnFinance covers IRDAI broadly but focuses on SEBI/RBI (LODR Reg 33/34, CRAR/NPAs, XBRL). No specialized agents for IRDAI actuarial reserving, solvency margins, health/claims master directions.
4. **Drift handling** — no evidence of adaptive retraining for RBI master direction changes; agents are static post-circular parsing. Synapta's auto-retrain pipeline design (`AUTO_RETRAIN_PIPELINE_DESIGN.md`) directly addresses this.
5. **Multi-regulator edges** — OnFinance is sparse on PFRDA (pensions) and FEMA (cross-border / forex / ODI compliance).
6. **Avoided product lines** — no underwriting / credit risk, no equity research agents, no investigative workflows beyond ComplianceOS / InvestigativeOS demos. Open vertical extensions.

**YC late-applicant precedent:**
- 10× tech defensibility via published statistical rigor (Brex over funded incumbents via API moats; Clarity over Stripe rivals via superior fraud models)
- Sovereign + on-prem for regulated markets
- Underserved verticals (IRDAI actuarial, drift handling)
- Solo founder + methodology-first narrative differentiates from team-led competitors marketing case studies

---

## Q2 — Cross-benchmark eval rigor (DEEP)

**Verdict:** "Cross-benchmark comparisons require caution due to mismatched datasets, metrics, inference modes." NeurIPS/ICLR/ACL reviewers reject cross-bench claims without shared evals as "not apples-to-apples" — references HELM (mandates scenario-specific protocols) and BIG-bench (warns against score extrapolation).

**Required protocol:**
1. Shared held-out benchmark (Synapta on IndiaFinBench OR vice versa) — exactly what eval PID 349095 is doing
2. Identical prompts, temp=0, max_tokens, hardware/seeds
3. Multiple metrics: accuracy + token F1 + exact match (don't pick one)

**Statistical tests:**
| Test | Use case | When preferred |
|---|---|---|
| **McNemar** | Binary pass/fail per Q | Same instances, dichotomous outcomes (we use this) |
| **Paired bootstrap** | Score-difference CIs (e.g., 89.6 vs 89.7) | Resampling, no normality assumption |
| **Wilcoxon signed-rank** | Ordinal/F1 scores | Non-binary metrics (we should add for F1) |
| **t-test** | — | AVOID across benchmarks (ignores dependencies) |

**Metric warning:** "Substring match (Synapta's 89.6%) **risks overcrediting partial outputs**." IndiaFinBench likely uses stricter accuracy or F1. Token F1 balances precision/recall; exact match for MCQ. **Reviewers demand metric alignment** — ACL papers favor F1 for extraction over substring. *Our IndiaFinBench eval reports all three (sub, norm, F1) — this is the right move.*

**Confounds even after shared eval:**
- **Task-style transfer:** RBI/SEBI 664 Qs may overfit to a specific style (heading-extractive, regex-numeric). Adapter underperforms IndiaFinBench's numerical_reasoning + contradiction_detection. **This is exactly what we are seeing in flight: 7.4% on numerical, 35% on regulatory_interpretation.**
- **Prompt sensitivity:** zero-shot Gemini varies 5-27% by format (CoT can hurt; option-shuffling flips 59% of predictions). Fine-tuned adapters less sensitive but still domain-tuned.
- **Adapter vs zero-shot tradeoff:** Fine-tuning boosts in-domain, risks OOD brittleness; zero-shot scales via emergence.

**Reviewer expectations:**
- Cohen's d > 0.8 for "large" effect
- Ablate prompts (3-5 variants)
- Calibration curves
- Honest disclosure of OOD degradation

---

## Q3 — Indian BFSI compliance AI market reality (DEEP)

**Contract values 2026 (mid-tier banks/NBFCs/insurers):** ₹2–10 crore/year ($240K–$1.2M USD). Larger numbers tied to TCS/Infosys via NASSCOM ecosystems; RBI sandbox graduates secure sub-₹5 crore initial contracts. **No named deals** with ICICI/HDFC/Axis/Kotak/SBI awarding compliance AI to seed startups.

**Procurement cycle: 9–18 months** (consistent across both auto and deep modes).
- 2-3 mo RFP issuance
- 3-6 mo evaluation/vendor demos
- 2-4 mo board / CRO approval (mandatory under RBI 2024 IT Governance)
- 3-6 mo pilot

**Buying committee:** CTO (feasibility) + CCO (compliance) + CRO (risk) + CISO (cyber) + CDO (data). **CRO or CCO holds final decision authority** — not the CTO. Reference: SBI's Compliance Risk Management Committee involves CXO oversight.

**Deployment requirements (mandatory):**
- On-premise OR air-gapped Indian cloud (DPDP Act 2023, RBI data localization)
- SOC2 Type II + ISO 27001 + RBI Cybersecurity Framework 2024
- **Full explainable AI with audit trails + human-in-loop** ← biggest moat for adapter+attention-trace systems vs frontier API black boxes

**Year-1 revenue (fresh seed, no banking history):** ₹1–3 crore ($120K-$360K). Via 1-2 mid-tier NBFC pilots, **assumes sandbox graduation + CTO intros**. Major banks demand 2–3 year track record before contracting.

**Steel-man (why this is harder than founders assume):**
1. RBI Master Direction 2024 mandates auditability → unexplainable AI rejected
2. RBI/SEBI/IRDAI fragmentation = no one product covers all three
3. IT budgets allocate <5% to AI (core banking takes priority)
4. TCS/Infosys/NASSCOM incumbents lock 70%+ spends
5. Sub-3-year vendors face **80%+ audit rejection rate**

---

## What this changes in the YC application

1. **OnFinance section** — replace generic differentiation with 6 specific gaps (sovereign/on-prem, methodology transparency, IRDAI depth, drift handling, PFRDA/FEMA, no underwriting/credit-risk).

2. **Comparison framing** — once IndiaFinBench eval finishes, report all 3 metrics (substring + normalized + token F1) with paired bootstrap CIs. Acknowledge "task-style transfer" as primary remaining confound.

3. **Revenue projection** — anchor at ₹1-3 cr year-1 (₹2-5 cr was optimistic). Frame as: "We will graduate the RBI sandbox with the methodology rigor no incumbent publishes."

4. **Steel-man response** — add explicit acknowledgment of 80% audit rejection rate for sub-3-year vendors. Counter: "Our methodology transparency + open-sourced benchmark + reproducible held-out evaluation is the substitute for missing track record."

5. **Drift handling positioning** — explicit competitive moat over OnFinance (which has none publicly).

---

## What we're seeing live (IndiaFinBench at 36% complete, 117/324 done)

| Task type | n | Substring | Norm | F1 | Gemini Flash baseline |
|---|---|---|---|---|---|
| regulatory_interpretation | 54 | 35.2% | 35.2% | 0.365 | 93.1% |
| temporal_reasoning | 20 | 25.0% | 25.0% | 0.329 | 88.5% |
| numerical_reasoning | 27 | 7.4% | 11.1% | 0.294 | 84.8% |
| contradiction_detection | 16 | 62.5% | 62.5% | 0.014 | 88.7% |
| **Overall** | **117** | **30.8%** | **31.6%** | **0.295** | **89.7%** |

**Honest read:** Synapta's adapter does NOT transfer well to IndiaFinBench's question style. The ~60pp gap vs Gemini Flash on this benchmark is real and disclosable. Per perplexity Q2's "task-style transfer" warning — this is the predicted failure mode of fine-tuned adapters on OOD benchmarks.

**What this means for YC framing:**
- Drop the "Synapta 89.6% ≈ Gemini Flash 89.7%" narrative entirely (perplexity Q2 confirmed it's not legitimate)
- Lead with: "Our adapter delivers +30.9pp lift on its training distribution (McNemar p<10⁻⁴⁸) — methodology that's repeatable per customer dataset. We do NOT claim frontier parity on out-of-distribution academic benchmarks; we do claim a reproducible recipe to lift any base model on any customer's specific document corpus."
- This is actually a *better* pitch — narrower, more honest, more defensible.
