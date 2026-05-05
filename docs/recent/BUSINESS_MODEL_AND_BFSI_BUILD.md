# Synapta — Business Model and BFSI Adapter Build
*Written May 3 2026 · for Udit's reference*

---

# PART 1 — BUSINESS MODEL

## What we sell

A base large language model (Nemotron-30B in 4-bit) plus customer-trained LoRA adapters, deployed on the customer's own hardware. Each customer gets:

1. **The adapter** trained on their proprietary regulatory + SOP corpus (the IP they own)
2. **The pipeline** that produced it (so they can retrain when regulations change)
3. **Format Guard routing** to swap between adapters at inference time
4. **Eval methodology** so their compliance team can validate outputs against held-out questions

We do *not* sell a hosted API. We do *not* sell access to a multi-tenant model. The whole pitch is sovereignty: your data never leaves your environment.

## Who buys

Indian financial institutions that legally cannot send customer or regulatory data to OpenAI / Anthropic / Google because of:
- RBI's compute localization mandates (2023 Master Direction on Outsourcing IT Services)
- DPDP Act 2023 data residency requirements
- SEBI's IT framework for regulated entities

Realistic addressable market (year 1-3 reach):
- ~10 large private sector banks (HDFC, ICICI, Axis, Kotak, IndusInd, Federal, RBL, IDFC First, AU Small Finance, Bandhan)
- ~15 NBFCs (Bajaj Finserv, Chola, Mahindra Finance, Muthoot, Manappuram, Tata Capital, etc.)
- ~10 insurers (HDFC Life, ICICI Pru, SBI Life, Bajaj Allianz, Tata AIA, etc.)
- ~10 fintech / payment companies (Razorpay, PhonePe, NPCI, etc.)

Total reachable: ~45 institutions in years 1-2. Public sector banks (SBI, BoB, etc.) are slower-cycle; deprioritize for first 24 months.

## Pricing structure

Three-tier model:

| Phase | Product | Price | Duration |
|---|---|---|---|
| **Pilot** | One adapter trained on customer corpus + on-prem deployment | $50K | 60 days |
| **Platform license (annual)** | Format Guard infra + adapter update pipeline + support | $250-400K/yr | recurring |
| **Per-adapter training** | Each new domain adapter (e.g., bfsi_extract → bfsi_recall → internal-SOP) | $50-75K each | one-time |

**Why this structure**:
- $50K pilot is low enough for a CTO to sign off without board approval (typical Indian bank discretionary IT spend)
- Annual license creates recurring revenue (the actual investor metric)
- Per-adapter fee incentivizes customers to commission *more* adapters (each one strengthens the platform lock-in)

**Customer keeps the IP** even if they don't renew. This makes the pilot easy to sign — the customer faces zero IP risk. Renewals are driven by needing the *platform* (Format Guard, eval pipeline, retraining infrastructure), not the adapter weights themselves.

## Revenue projections (conservative)

| Year | Customers | Avg ACV | ARR | Notes |
|---|---|---|---|---|
| Year 1 | 3 pilots | $50K | $150K | Mostly proof, mostly burn |
| Year 2 | 3 pilots → 5 platform | $300K avg | $1.5M | First retentions |
| Year 3 | 8 platform + 3 new pilots | $400K avg | $3.4M | Compounding |

This is conservative. Optimistic version assumes a flagship private sector bank closes in year 1, which 5×s year-2 revenue via reference effect.

## Defensibility (3 layers)

1. **Methodology moat** — document-disjoint held-out eval is hard to fake; competitors who paraphrase-augment can't survive a real customer dataroom evaluation
2. **Data flywheel** — each customer's corpus produces an adapter only they own, but the *pipeline* improves with every customer engagement (better regex templates, better chunking, better validators)
3. **Sovereignty positioning** — frontier APIs (GPT-4o, Claude) cannot deploy on-prem in India. Our 4-bit single-GPU stack is the only thing that fits

## Costs / unit economics

- Compute: ~$50/customer/month (one RTX 5090 or A100 amortized; or customer's own hardware)
- Training: $5-50 per adapter (one-time, deterministic pipeline)
- Customer success: 1 engineer per ~10 customers
- Gross margin at $300K ACV: ~80% after support costs

## Why now

- RBI tightening sovereignty (April 2023 Master Direction)
- Open-source frontier models (Nemotron-30B, Qwen-3.6 27B, Llama-4) closing the gap on small-context tasks
- PEFT-LoRA matured to production-deployable
- Indian banks have been deferring AI compliance investments for 18-24 months waiting for sovereign options

The window for being the first mover is now. By 2027, every IT services firm (TCS, Infosys, Wipro) will have a "compliant LLM" offering — but they'll be reselling commodity wrappers, not adapter platforms.

## Honest risks

1. **Sales cycle in Indian BFSI is 6-18 months.** Series A timing depends heavily on closing 2-3 reference customers fast.
2. **Solo founder is a YC concern.** Need a technical cofounder by year 1.
3. **Frontier models may eventually deploy on-prem.** OpenAI / Anthropic / Google have signaled interest in sovereign deployments. Our window is 18-36 months before this assumption breaks.
4. **Custom adapters require customer corpus access.** Some banks will refuse to share even internal SOPs with us during training, even on-prem. Workaround: customer runs the pipeline themselves with our scripts.

---

# PART 2 — THE BFSI ADAPTER (WHY, WHAT DATA, ACCURACIES)

## Why we trained it

Three reasons:

1. **Prove the customer flywheel can actually work.** Before May 3, the claim "customer brings corpus, we produce a domain adapter that lifts performance" was a slide promise. After May 3, it's a measured outcome on a real regulator corpus. This is the difference between a fundable startup and a deck.

2. **Generate a defensible, citable, measured number for YC and CTO conversations.** The +31.3 pp lift on document-disjoint held-out is the single hardest claim to substantiate in this space. Now we can substantiate it.

3. **Validate the deterministic QA construction pipeline end-to-end.** The pipeline (scrape → extract → chunk → 3-tier QA → validator → train → held-out eval) had never been run on a fresh regulator corpus before. This was the integration test.

## What data we used and why

**Source**: 130 PDFs, 115 MB total, all public domain from gov.in:
- 80 RBI Master Directions from `rbi.org.in/Scripts/BS_ViewMasDirections.aspx`
- 50 SEBI Master Circulars from `sebi.gov.in/sebiweb/home/HomeAction.do?...sid=1&ssid=6`
- (Tried IRDAI; blocked by Azure WAF; deferred to a Playwright-based scrape later)

**Why this data specifically**:
- Master Directions and Master Circulars are the *canonical regulatory text* a compliance officer queries — not interpretive guides, not bank circulars, the actual rules
- Public domain → license-clean for the demo, reproducible by anyone
- Public sources → no IP entanglements with any potential customer (we don't have to negotiate access)
- Both regulators (banking + capital markets) → cross-domain generalization signal

**Data discipline applied**:
- SHA-256 of every source PDF stored in `manifest.jsonl` (provenance audit)
- 26 entire PDFs (20% of total) held out *before* any training data was constructed (document-disjoint, not paraphrase-disjoint)
- Train and eval share zero source documents — so eval scores cannot be inflated by the model memorizing question paraphrases

**Pipeline output**:
| Stage | Input | Output |
|---|---|---|
| Scrape | URLs | 130 PDFs (115 MB) |
| Text extract | PDFs | 8.06M chars, 7,329 sections detected |
| Chunk | text | 4,185 chunks (mean 425 tokens, on numbered section boundaries) |
| QA construction | chunks | 4,477 raw QA pairs |
| Validator (10 checks) | raw QA | 4,373 valid (98.45% pass) |
| v2 cleaner template rerun | chunks | 3,195 raw → 2,931 valid (91.7% pass) |
| Document-disjoint split | valid QA | 2,931 train / 664 eval (v2) |

**3-tier QA construction (no LLM-generated questions — deterministic only):**
- **Tier 1** (native FAQ): 0 examples — RBI MDs don't use Q1:/A1: format
- **Tier 2** (numeric regex): captures Rs amounts, day timelines, percentages, section references
- **Tier 3** (heading-based extractive): paragraph body keyed on numbered section heading

**Why no LLM-generated questions**: using an LLM (e.g., GPT-4 or Nemotron itself) to generate training questions causes self-distillation — the model learns to mimic its own (or another model's) question style, not the regulatory content. We rejected this entirely.

## Training configuration

| Parameter | Value | Reason |
|---|---|---|
| Base model | Nemotron-30B-Nano-A3B | Best open hybrid Mamba-MoE-Attn at the size that fits a single RTX 5090 |
| Quantization | 4-bit NF4 (BitsAndBytesConfig) | Single-GPU deployment requirement |
| LoRA rank | 16 | r=64 (existing math/code adapter rank) overfits at 2,931 examples; r=16 is the sweet spot |
| LoRA alpha | 32 | 2× rank, standard |
| LoRA dropout | 0.05 | Mild regularization |
| Target modules | q, k, v, o, gate, up, down | All attention + MLP; MLP needed for knowledge injection |
| Learning rate | 1e-4 | Standard for LoRA on 30B base |
| Schedule | Cosine, warmup 0.05 | Standard |
| Epochs | 1 | Cleaner v2 data + small set; 2 epochs would overfit |
| Batch | 1 × 16 grad accum (effective 16) | Memory-bound; effective batch via accumulation |
| MAX_LEN | 1024 | Token analysis: 99th percentile = 849; 1024 covers 100% |
| Optimizer | paged_adamw_8bit | Standard QLoRA |
| Precision | bf16 + gradient_checkpointing | Memory efficient |
| max_grad_norm | 0.3 | Prevents outlier-batch gradient spikes |
| Trainable params | 434.6M / 32B (1.36%) | LoRA r=16 across all attn+MLP modules |
| Wall clock | 174 update steps × 70s = 3h 28min | RTX 5290 4-bit; 4-bit blocks Mamba fast-path kernels |
| Output adapter size | 1.74 GB safetensors | Stored at `adapters/nemotron_30b/bfsi_extract/best/` |

## Accuracies (held-out, document-disjoint)

**Headline (n=595 paired questions, eval set never seen during training)**:

| Mode | Substring match | Wilson 95% CI | Token F1 |
|---|---|---|---|
| Base Nemotron-30B (4-bit, no adapter) | **58.3%** | [54.3%, 62.2%] | 0.133 |
| **+BFSI extract adapter** | **89.6%** | [86.9%, 91.8%] | 0.173 |
| **Lift** | **+31.3 pp** | (non-overlapping CI) | +0.040 |

**Statistical significance — McNemar paired test on 595 paired samples**:

|  | Adapter correct | Adapter wrong |
|---|---|---|
| Base correct | 334 | 13 |
| **Base wrong** | **199** | 49 |

- 199 questions adapter fixed (base wrong → adapter right)
- 13 questions adapter broke (base right → adapter wrong)
- 15.3× improvement-to-regression ratio
- Exact binomial McNemar **p = 6.26 × 10⁻⁴⁴** — probability that this lift is chance is essentially zero

**Per-tier breakdown**:

| Tier | Type | n | Base | +BFSI | Lift |
|---|---|---|---|---|---|
| 2 | Numeric facts (Rs amounts, %, days) | 345 | 63.0% | 87.8% | +24.8 pp |
| 3 | Heading-based extractive paragraphs | 250 | 52.9% | 92.0% | **+39.1 pp** |

The Tier 3 lift is largest because base model genuinely has no knowledge of Indian regulations; adapter teaches it. Tier 2 is smaller because base can already pattern-match numbers when the question is straightforward.

**Per-regulator breakdown**:

| Regulator | n | Base | +BFSI | Lift |
|---|---|---|---|---|
| RBI | 342 | 58.0% | 89.8% | +31.8 pp |
| SEBI | 253 | 59.7% | 89.3% | +29.6 pp |

Lift is consistent across both regulators — the adapter learned the *general shape* of Indian financial regulation, not just one corpus.

## What is NOT measured

Be honest about gaps so we don't get caught:

1. **Format Guard mode (3rd eval mode) never ran.** Eval timed out at 4h before reaching it. We have no measurement of "FG with bfsi adapter routing" vs base. Can be re-run; expected to roughly match the bfsi-only number.
2. **Token F1 lift is small (+0.04)** despite huge substring lift. Reason: adapter often quotes the answer plus extra context, hurting strict token overlap. For the production use case (compliance officer reads the answer), substring is the correct metric.
3. **No human evaluation by a compliance professional.** All scoring is automated (substring + F1). To claim "compliance-grade" we'd need a paid Indian compliance professional to grade ~100 outputs.
4. **IRDAI not included** — only RBI and SEBI corpora. Insurance regulation is a third of BFSI.
5. **Numbers measured with simple test questions, not real customer queries.** The Tier 2 templated questions ("What is the amount specified for X?") are repetitive by design. Real compliance officer queries are messier.
6. **Single random seed.** Trained once, evaluated once. No sample-mean over multiple seeds.
7. **Frontier comparison not yet run** — we don't have a measured number for GPT-4o or Claude on these same questions. (Script ready, awaits API keys.)

## Reproducibility

The full pipeline runs in ~8 hours, $5 of compute, on a single RTX 5090. Scripts in `synapta_src/data_pipeline/`:

```
01_scrape_rbi_mds.py 80                 # ~10 min
01c_scrape_sebi_circulars.py            # ~5 min
02_extract_text.py                      # ~1 min (mp Pool 8)
03_chunk_circulars.py                   # ~1 min
04b_build_qa_pairs_v2.py                # ~10s
06_validate_qa.py                       # ~10s
07_train_bfsi_extract.py                # ~3.5h on RTX 5090 4-bit
08_eval_bfsi_extract.py                 # ~3-4h, partial OK
```

All raw PDFs at `data/{rbi,sebi}_corpus/pdfs/`. Manifests with SHA-256 at `data/{rbi,sebi}_corpus/manifest.jsonl`. Train/eval split at `data/rbi_corpus/qa/split_manifest_v2.json` (lists exact PDFs in each set).

## Bottom line

In one day, on a single GPU, we went from "we claim domain adapters work" to "we measured a +31.3 pp lift, p ≈ 10⁻⁴⁴, on a held-out set the model never saw, on a domain (Indian financial regulation) where there is no published baseline." That's the technical foundation. The business model above is what we monetize from it.
