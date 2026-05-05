# [SYNAPTA logo]    SYNAPTA

**Sovereign AI for India's regulated $400M-$1B inference market.**
Frontier-class reasoning on your hardware. No data egress. Audit-defensible.

Contact: [founder@synapta.ai]   ·   Web: [synapta.ai]   ·   Founder: Udit Jain

---

## +31.3 pp lift on document-disjoint held-out RBI/SEBI extractive QA.

### McNemar p < 10⁻⁴³ on n=595 paired examples. Trained in 3h 28min on a single GPU at $1.50/day amortized.

| n=595 paired, held-out (26 PDFs never seen) | Base Nemotron-30B | + Synapta BFSI Adapter |
| ------------------------------------------- | ----------------- | ---------------------- |
| Substring match                             | **58.3 %**        | **89.6 %**             |
| Wilson 95% CI                               | [54.3, 62.2]      | [86.9, 91.8]           |
| Tier 3 (heading-extractive, n=250)          | 52.9 %            | 92.0 %  (+39.1 pp)     |
| Per-regulator: RBI / SEBI                   | 58.0 % / 59.7 %   | 89.8 % / 89.3 %        |

199 questions adapter-only-correct vs 13 base-only-correct (15× ratio). No paraphrase augmentation. No LLM-generated questions. Document-disjoint train/eval split — the methodology a procurement auditor will accept.

---

## What you get

- A LoRA adapter (~1.7 GB) trained on **your** regulatory corpus, deployed on a single on-prem GPU alongside Nemotron-30B 4-bit base.
- The full deterministic pipeline (scrape → extract → chunk → 3-tier extractive QA → 10-check validator → LoRA train → held-out eval), reproducible in under 8 hours from a clean clone.
- Document-disjoint held-out eval methodology with McNemar paired-test reporting — the same framework you can hand to your auditor.
- 60 days of engineering support, integration with your retrieval stack, and adapter retraining on corpus updates.

## What you bring

- Your regulatory + policy corpus (PDFs, internal circulars, board policies — whatever is in scope).
- One GPU box (RTX 5090 32 GB or A100 40 GB) inside your firewall, or one we provision on your behalf.
- A compliance / risk SME for ~2 hours / week during the pilot to validate eval question construction.

---

## Architecture

```
   [Your firewall — no egress]
   Nemotron-30B 4-bit  ──►  Router  ──►  REST API  ──►  Your apps
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Format Guard    Your BFSI LoRA   Other domains
```

Single GPU, single base model, hot-swappable adapters. Hardware budget: ~$20K consumer GPU.

---

## Pricing anchor

**$50,000 pilot, 60 days.** Your data never leaves your environment. You keep the adapter weights even if you do not renew. No lock-in by contract or by design.

## Why this matters now

- **RBI compute-localization circular (2024) + DPDP Act** make frontier APIs (OpenAI, Anthropic, Gemini) legally inaccessible for ~70% of Indian enterprise data.
- **Frontier providers cannot deploy on-prem** — their unit economics depend on cloud routing of single large models. They have a business-model conflict against serving you.
- **Your competitors will move first.** The category window opens now and closes when one mid-tier Indian bank publishes a measured AI compliance result. Be that bank.

---

## Next step

**30-minute call to scope a 60-day pilot on your regulatory corpus.**
[calendar.synapta.ai/cto-pilot]

*Synapta is a YC W26 applicant. Founder previously won an international ECG hardware-and-ML hackathon and met with Microsoft India CTO on 2 May 2026.*
