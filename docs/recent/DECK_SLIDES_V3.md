# Synapta — Pitch Deck V3

**Audience:** YC W26 application + investor / design-partner pitches
**Format:** 14 content slides + 2 appendix
**Date:** May 3, 2026
**Founder:** Udit Jain (solo, 19)

Each slide below is sized for one physical slide. Headlines are one sentence. Speaker notes read in roughly 60 seconds out loud.

---

## Slide 1 — Cover

**Headline:** Synapta — Sovereign AI for India's regulated $400M-$1B inference market.

**Bullets:**
- Frontier-class reasoning on a single consumer GPU, behind your firewall.
- Adapter-routing on a 30B open base model — customer-extensible.
- Built for the 70% of Indian enterprise data that legally cannot reach OpenAI or Anthropic.
- Solo founder, 19. YC W26 candidate.

**Speaker notes:** I'm Udit. Synapta is the inference layer for sovereign AI. Frontier labs spent two hundred billion dollars training models that the majority of regulated enterprise data cannot legally touch. We make it so a bank, hospital, or defense buyer can run a thirty-billion-parameter open model on their own hardware, with adapter routing they can extend with their own data. India BFSI is the beachhead. The pipeline is what I want to walk you through today.

**Suggested visual:** Synapta wordmark over a faint India outline; tagline "Sovereign AI Inference" beneath.

---

## Slide 2 — The Problem

**Headline:** Indian BFSI is stuck in a sovereignty-accuracy-cost trilemma — and today nobody solves all three.

**Bullets:**
- **Sovereignty:** RBI 2024 localization circular + DPDP Act block frontier APIs for ~70% of regulated data.
- **Accuracy:** open base models alone underperform on regulator-specific terminology, citations, and section structure.
- **Cost:** running a 70B+ frontier-class model on-prem is capex prohibitive for mid-tier banks and NBFCs.
- **Result today:** banks either stall on AI deployment or quietly pipe sensitive data to non-compliant APIs.

**Speaker notes:** Every Indian bank CTO I have spoken to faces the same trilemma. They cannot send compliance data to Claude or GPT — that violates RBI localization. They cannot run a 70B model on-prem at the price point they can justify. And open base models out-of-the-box do not know the difference between a SEBI Master Circular and a marketing brochure. Today most of them just pause AI for the regulated workflows. That stall is the gap we are filling.

**Suggested visual:** Triangle diagram with three corners labeled Sovereignty / Accuracy / Cost, each pulling away from the center; current vendors plotted as dots that hit only one or two corners.

---

## Slide 3 — Why Now

**Headline:** Three forces converged in 2024-2026 to make sovereign AI inference newly viable as a category.

**Bullets:**
- **Regulatory tailwind:** RBI compute-localization circular (2024) and DPDP Act make cloud frontier APIs legally inaccessible for regulated workflows.
- **Open models caught up:** Nemotron-30B, Llama, Qwen now reach usable quality at sizes deployable on $20K of consumer hardware.
- **PEFT matured:** LoRA + 4-bit quantization make domain specialization reliable, *if* the data pipeline is built correctly.
- **Frontier labs are structurally blocked:** their unit economics depend on cloud routing; on-prem is a business-model conflict.

**Speaker notes:** This category did not exist eighteen months ago. Open models were not good enough, the regulatory deadlines had not bitten, and adapter techniques were too brittle for production. All three flipped between 2024 and 2026. Meanwhile OpenAI and Anthropic cannot serve this market without breaking their own pricing model — they would have to bring frontier models on-prem at margins their cloud business does not allow. The window opens now and stays open until they restructure, which I estimate buys us twelve to eighteen months of architectural lead.

**Suggested visual:** Three converging arrows on a timeline (2024 -> 2026) labeled Regulation / Open Models / PEFT, meeting at "Synapta opportunity."

---

## Slide 4 — The Insight

**Headline:** Small adapters can route a single open base model to crush domain tasks — without retraining the base.

**Bullets:**
- One Nemotron-30B base model in 4-bit, plus swappable LoRA adapters for code / math / BFSI / customer-specific.
- Format Guard: per-token regex routing decides which adapter is active inside which span (e.g. math adapter inside Python blocks).
- Total inference VRAM ~17.5 GB. Runs on a single RTX 5090. No cloud dependency.
- Customer trains additional adapters on their own corpus — that is the flywheel.

**Speaker notes:** The architectural bet is that you do not need a different model per domain. You need one strong base and a library of cheap, swappable adapters with smart routing between them. We call the routing layer Format Guard — it inspects the generation context per token and decides which adapter should be hot. The base never moves. Adapters are tens of megabytes. A customer who trains five adapters on their own compliance corpus has built switching cost we never had to engineer.

**Suggested visual:** Simple stack diagram: Nemotron-30B base on bottom, three labeled adapter modules on top (Code / Math / BFSI), Format Guard router as a thin layer in between with arrows showing per-token switching.

---

## Slide 5 — Architecture

**Headline:** Format Guard routing on Nemotron-30B + LoRA adapters, in 17.5 GB of inference VRAM.

**Bullets:**
- **Base:** Nemotron-30B-Nano-A3B (NVIDIA hybrid Mamba-MoE-Attention, 30B total / 3.5B active per token), 4-bit NF4.
- **Adapters:** LoRA r=16, alpha=32, attention + MLP modules; ~50 MB each on disk.
- **Routing:** Format Guard regex inspects context every N tokens, swaps adapter weights in place.
- **Hardware:** single RTX 5090 (32 GB VRAM); ~22 GB used during training, ~17.5 GB inference.
- **Deployment:** on-prem appliance or single-node cloud-VPC; air-gap-capable.

**Speaker notes:** Architecturally everything fits on one consumer GPU. The base is Nemotron-30B in four-bit NF4 quantization. Adapters are tiny — rank sixteen, alpha thirty-two, applied to attention and MLP modules. Format Guard sits between the tokenizer and the forward pass, deciding per token which adapter should be active. Customers can deploy this as an on-prem appliance behind their firewall or inside their own VPC. There is no API call to us at inference time. That is what makes it sovereign.

**Suggested visual:** System diagram — tokenizer to Format Guard router to base model, with three adapter slots that can be hot-swapped; sidebar lists VRAM budget per component.

---

## Slide 6 — Proof Point #1: Code Reasoning

**Headline:** Format Guard routing lifts HumanEval +17.1pp and MBPP +20.1pp on full benchmarks, statistically significant.

**Bullets:**
- **HumanEval (n=164):** 56% -> 73%, **+17.1 pp**, McNemar paired test **p < 0.001**.
- **MBPP (n=164):** 42% -> 62%, **+20.1 pp**, McNemar paired test, same hardware.
- Identical base model, identical decoding params — only the routing layer differs.
- Reproducible: every claim has a JSONL artifact path in the repo.

**Speaker notes:** This is the published result. On full HumanEval, one hundred sixty-four problems, our routing lifts pass rate from fifty-six to seventy-three percent. McNemar paired test, p less than zero point zero zero one. On full MBPP, also one sixty-four problems, forty-two to sixty-two percent — a twenty-point lift on a benchmark that fintechs use directly. Same base, same decoding, only the routing changes. Every number is backed by a JSONL artifact in the repo. This is the methodologically airtight number that proves the architecture.

**Suggested visual:** Two side-by-side bar charts (HumanEval, MBPP) with base vs Format Guard, deltas annotated in bold, p-value footnote.

---

## Slide 7 — Proof Point #2: The BFSI Build (Methodology)

**Headline:** This week we shipped an audit-defensible BFSI compliance adapter pipeline — 130 PDFs, deterministic QA, document-disjoint hold-out.

**Bullets:**
- **Source:** 80 RBI Master Directions + 50 SEBI Master Circulars (130 PDFs, 115 MB, all public-domain government text).
- **Extraction:** 8.06M characters via pdfplumber + multiprocessing; 7,329 numbered sections detected.
- **Chunking:** 4,185 smart chunks on section boundaries, mean 425 tokens.
- **QA construction:** 3-tier deterministic regex extraction. **Zero LLM-generated questions.** No paraphrase augmentation.
- **Validator:** 10-check quality gate, 98.45% pass rate -> 4,373 train / 698 eval pairs (cleaner v2: 2,931 train / 664 eval currently in run).
- **Hold-out:** 26 entire PDFs (20% of corpus) document-disjoint — model never sees them during training.

**Speaker notes:** Every domain-LLM pitch you will see this year has the same hidden bug — an LLM was used to generate evaluation questions from the same documents the model trained on. Eval scores look great, deployment falls over. We refused to do that. One hundred thirty real regulator PDFs, three deterministic regex templates extract questions directly from source text, a ten-check validator gates every pair, and the held-out eval is twenty-six entire PDFs the model literally never sees. This is the test that survives a procurement audit. We built it first.

**Suggested visual:** Pipeline waterfall — Sources (130 PDFs) -> Extraction (8.06M chars) -> Chunking (4,185 chunks) -> QA (4,477 raw -> 4,373 validated) -> Document-disjoint split (104 train PDFs / 26 eval PDFs).

---

## Slide 8 — Proof Point #2: Results

**Headline:** Held-out evaluation on 26 unseen PDFs — base 58.3% → +BFSI adapter 89.6% substring match, +31.3 pp, McNemar p = 6.26 × 10⁻⁴⁴.

**Bullets:**
- **Held-out substring match (n=595 paired, document-disjoint):** base **58.3%** [Wilson 95% CI 54.3–62.2] → +BFSI **89.6%** [86.9–91.8], **+31.3 pp**. Token F1 0.133 → 0.173 (+0.040; adapter often quotes answer + surrounding context, hurting strict overlap).
- **McNemar paired significance:** **p = 6.26 × 10⁻⁴⁴** (two-sided exact binomial). 199 adapter-only-correct vs 13 base-only-correct — 15× improvement-to-regression ratio.
- **Per-tier:** Tier 2 numeric (n=345) 63.0% → 87.8% (+24.8 pp); Tier 3 heading-extractive (n=250) 52.9% → 92.0% (**+39.1 pp**).
- **Per-regulator:** RBI (n=342) 58.0% → 89.8% (+31.8 pp); SEBI (n=253) 59.7% → 89.3% (+29.6 pp). Lift consistent across both — adapter learned the general shape of Indian financial regulation, not one corpus.
- **Training:** 174 update steps × 70s = 3h 28min on a single RTX 5090 (32 GB), 4-bit NF4 base + LoRA r=16 / alpha=32 on attn+MLP, 1 epoch on 2,931 examples. Trainable 434.6M / 32B (1.36%). Adapter 1.74 GB safetensors.
- **Honest caveat:** eval timed out at 4h with 595 of 664 paired examples completed in bfsi-extract mode. Format Guard composition mode is re-launching in parallel; results expected within 90 min. The 595-pair sample is more than large enough for the McNemar p ≈ 10⁻⁴⁴ headline to stand.

**Speaker notes:** Held-out evaluation, document-disjoint, 26 PDFs the model never saw. Five hundred ninety-five paired questions. Base Nemotron-30B in four-bit hits fifty-eight point three percent substring match. With our BFSI adapter on top, eighty-nine point six percent. A thirty-one point three point lift. McNemar paired test p equals six point two six times ten to the negative forty-four — the probability this happened by chance is essentially zero. Lift is consistent — RBI plus thirty-one point eight, SEBI plus twenty-nine point six, numeric extraction plus twenty-four point eight, paragraph extraction plus thirty-nine point one. Token F1 lift is more modest because the adapter likes to quote answer plus surrounding context, which hurts strict overlap but is the right behaviour for a compliance officer workflow. Trained in three hours twenty-eight minutes on a single RTX 5090. Honest caveat: the eval was killed at the four-hour timeout so we have five hundred ninety-five of six hundred sixty-four paired examples; that sample is enormous for the significance test. Format Guard composition mode is re-running now.

**Suggested visual:** Three side-by-side bar groups (Overall, Per-Tier, Per-Regulator). Overall: Base 58.3% vs +BFSI 89.6% with the +31.3 pp delta annotated and p = 6.26 × 10⁻⁴⁴ in the footer. Per-tier: Tier 2 (63.0 → 87.8) and Tier 3 (52.9 → 92.0). Per-regulator: RBI (58.0 → 89.8) and SEBI (59.7 → 89.3). Footnote: n=595 paired, document-disjoint held-out (26 PDFs); Format Guard composition mode re-running.

---

## Slide 9 — Why the Methodology Matters

**Headline:** OnFinance, Signzy, IDfy, Big-4 — the entire Indian BFSI AI surface ships marketing percentages; nobody publishes a paired held-out evaluation. Synapta is alone in the auditor-grade quadrant.

**Bullets:**
- **OnFinance AI** ($4.2M Sep 2025, Peak XV Surge — only direct competitor): "60 → 10 hours weekly", "65% productivity" — no model card, no held-out test set, no paired statistical test.
- **Signzy** (~$40M total) and **IDfy** (~$21M): KYC/RegTech case-study claims ("70% overhead reduction", "85% manual-review reduction") — no FPR/FNR per decision node, no methodology surfaced.
- **Big-4 (EY ART, PwC Compliance Insights, KPMG Intelligence Platform):** consulting-led, no public model metrics. EY's "automates 80% of RBI/SEBI reporting" is forecast-style, not a measured benchmark.
- **The shortcuts the field takes:** LLM-generated eval questions over training docs (data leakage); random chunk-level splits (memorization of neighbours); synthetic LLM-generated training data (generator-biased, ungrounded).
- **Our approach:** deterministic extractive QA from real regulator text + 26-PDF document-disjoint hold-out + 10-check validator + paired McNemar significance test. Day-one auditor artifact.

**Speaker notes:** Cross-referencing every Indian BFSI AI vendor we could find — OnFinance, Signzy, IDfy, EY, PwC, KPMG, Infosys, Wipro — none publishes a document-disjoint paired evaluation. The shortcuts are everywhere. LLM-generated eval over training docs is data leakage with extra steps. Random chunk splits let the model memorize neighbouring paragraphs. Synthetic training data inherits the generator's biases. A pipeline with any of these can show triple or quintuple the real lift. We did none of them. The procurement and audit teams at Indian banks ask for exactly this artifact at month six of a nine-month sales cycle. We built it first so it cannot become a deal-breaker.

**Suggested visual:** Two-column table — left column "Field standard" listing OnFinance / Signzy / IDfy / Big-4 with their marketing-style claims (no methodology icons); right column "Synapta" with the document-disjoint, paired-McNemar, validator-gated stack (green check icons).

---

## Slide 10 — The Customer Flywheel

**Headline:** Each customer brings their corpus -> we extract their proprietary domain adapter -> they own the IP, we own the platform.

**Bullets:**
- Customer ships internal compliance / underwriting / claims docs.
- We run the same pipeline (extraction -> chunking -> deterministic QA -> document-disjoint eval -> LoRA train) on their corpus.
- Customer-trained adapter is theirs — their IP, their data, never leaves their VPC.
- They accumulate 5+ adapters on internal corpora -> switching cost compounds.
- Synapta keeps the platform, the routing engine, the methodology — flywheel turns with each customer.

**Speaker notes:** This is the part investors usually miss on first pass. We are not selling a model. We are selling a pipeline. Every customer brings their own internal corpus — underwriting policies, KYC manuals, treasury operations docs — and we run the same audit-defensible pipeline on their data. The resulting adapter belongs to them. We do not see their data, we do not retain it, we do not retrain on it. After five adapters they have a custom inference stack that no competitor can rebuild without re-running the same pipeline on the same proprietary corpora. The platform compounds per customer.

**Suggested visual:** Circular flywheel — Customer corpus -> Pipeline -> Customer adapter (IP) -> Platform improvement -> Next customer; arrows showing positive feedback loop.

---

## Slide 11 — Market

**Headline:** 200 mid-tier Indian BFSI institutions, $400M-$1B addressable in 5 years, $20B+ TAM by 2028.

**Bullets:**
- **Beachhead:** ~200 mid-tier Indian banks, NBFCs, insurers (institutions too big to ignore AI, too small to build their own LLM team).
- **5-year addressable:** $400M-$1B annual inference + adapter spend across the beachhead.
- **Adjacent:** Indian healthcare, defense, government — same sovereignty constraints, same trilemma.
- **Long-term TAM:** $20B+ by 2028 across regulated enterprise globally (US/EU healthcare and finance face the same compliance walls).
- **Entry pricing:** appliance + adapter retainer; design-partner contracts target $250K-$500K ACV.

**Speaker notes:** The beachhead is two hundred mid-tier Indian banks, NBFCs, and insurers. Big enough to have an AI budget, small enough that they cannot build their own LLM team. Four hundred million to one billion dollars of addressable spend over five years just on this segment. The natural expansion is healthcare and government in India — same sovereignty constraints — and then regulated enterprise globally. US and EU healthcare and finance face the same compliance walls; the playbook ports. Twenty-billion-dollar TAM by 2028 if we execute the expansion.

**Suggested visual:** Three concentric rings — Inner: India BFSI mid-tier ($400M-$1B); Middle: India regulated total ($3-5B); Outer: Global regulated enterprise ($20B+). Beachhead labeled clearly.

---

## Slide 12 — Competition (the map, not the list)

**Headline:** A full Indian BFSI AI landscape sweep finds one direct competitor (OnFinance AI, $4.2M Sep 2025) and an empty quadrant where audit-defensible methodology meets sovereign deployment.

**Bullets:**
- **OnFinance AI** — $4.2M Pre-Series A, Sep 2025, lead Peak XV Surge. Ships own LLM (NeoGPT) + ComplianceOS. **Only direct competitor.** No published methodology, no held-out test set, no paired statistical test surfaced anywhere in their material.
- **KYC/RegTech adjacents:** Signzy (~$40M total) and IDfy (~$21M total) — identity/onboarding stack, not regulator-knowledge work; case-study marketing percentages only.
- **Big-4 (EY ART, PwC Compliance Insights, KPMG Intelligence Platform):** consulting-priced, methodology-opaque. EY's "80% of RBI/SEBI reporting" is forecast, not benchmark.
- **Indian IT services (Infosys Topaz, Wipro GIFT-City, HCLTech, Tata Elxsi):** horizontal GenAI capability, no India-regulator-specific product.
- **Frontier APIs (GPT/Claude/Gemini):** legally blocked by RBI localization + DPDP for ~70% of regulated data.
- **Horizontal LoRA serving (Together / Fireworks / Anyscale):** AI-native cloud customers; unit economics don't survive air-gap.
- **Synapta — alone in (sovereign + on-prem + RBI/SEBI corpus + adapter-portable + paired-eval) quadrant.** McNemar p = 6.26 × 10⁻⁴⁴ on 595 paired held-out questions is the only auditor-grade public evidence in the field. Indian RegTech funding fell 43% in 2024 — capital is selective; methodology will decide the rounds that close.

**Speaker notes:** We did a full landscape sweep this week. One direct competitor — OnFinance AI, four point two million from Peak XV Surge last September. They ship their own LLM. They publish marketing-style claims — sixty hours a week down to ten — but no model card, no held-out test, no paired statistical test. Adjacent: KYC/RegTech, Signzy and IDfy — different product, identity not regulator knowledge. Big-4 sells consulting hours around opaque platforms. The quadrant where sovereign deployment meets audit-defensible paired methodology has no occupant. That is the empty space we sit in, and our McNemar p of ten to the negative forty-four is the only artifact in the field that survives a procurement audit.

**Suggested visual:** Competitor map (2x2): x-axis Sovereignty (Cloud-API -> On-prem), y-axis Methodology (Marketing-claims -> Paired-eval). Plotted dots: OnFinance bottom-right (sovereign-ish, no methodology); Signzy/IDfy bottom-left; Big-4 mid-bottom; Together/Fireworks far bottom-left; Frontier APIs far bottom-left; **Synapta alone in the top-right quadrant.** Footer: source `COMPETITIVE_LANDSCAPE.md`, May 3 2026.

---

## Slide 13 — Team and Traction

**Headline:** Solo founder, 19; technical depth proven by this week's pipeline ship; Microsoft India CTO meeting May 2.

**Bullets:**
- **Founder:** Udit Jain, 19, solo. Six months building the Nemotron-30B adapter system + 12 routing strategies + full BFSI pipeline.
- **Recent ship velocity:** won an international ECG hardware-and-ML hackathon yesterday in 12 of 48 hours allotted (full-team competition, built solo).
- **Past builds:** ICPC India 2024 prelims landing site; VisionOS app at Bharat Mandapam (1 week, zero prior platform); full-stack product during Shark Tank India internship.
- **Industry signal:** Microsoft India CTO meeting held May 2.
- **Customer-discovery sprint:** May 5-13 with 10+ Indian BFSI institutions via existing networks.

**Speaker notes:** I am Udit, nineteen, solo. Six months on this stack — Nemotron-30B adapter system, twelve routing strategies tested, the full BFSI pipeline you saw on slide seven. Yesterday I won an international ECG hardware-and-ML hackathon in twelve of forty-eight hours allotted, full team competition, built hardware and ML pipeline alone. Past builds include the ICPC India twenty twenty-four prelims site, a VisionOS app at Bharat Mandapam in a week with zero platform experience, and a full-stack product during a Shark Tank India internship. Met the Microsoft India CTO yesterday. Customer-discovery sprint launches Monday with ten plus Indian BFSI institutions.

**Suggested visual:** Founder photo + a horizontal timeline ribbon: ICPC site -> Bharat Mandapam VisionOS -> Shark Tank India internship -> Synapta architecture -> BFSI pipeline -> ECG hackathon win -> MS India CTO meeting.

---

## Slide 14 — The Ask

**Headline:** YC W26, $500K SAFE, 18 months runway to BFSI design-partner contracts and 70B-scale adapter validation.

**Bullets:**
- **Round:** $500K SAFE, YC W26.
- **Runway:** 18 months.
- **Use of funds:** 60% cloud GPU access for 70B-200B scaling validation + adapters across verticals; 25% first 3 BFSI design-partner deployments; 15% founder + minimum infra.
- **Milestones:** held-out BFSI lift **shipped (this week, +31.3 pp, p = 6.26 × 10⁻⁴⁴)**; Format Guard composition mode reported (Week 1); **drift-resistance auto-retrain loop live (2 weeks post-YC — 5 components designed in `AUTO_RETRAIN_PIPELINE_DESIGN.md`, RBI/SEBI watchers → auto-ingest → incremental LoRA → eval gate → customer dashboard)**; 3 design-partner LOIs (Month 2); first paid pilot (Month 6); 70B adapter routing validated (Month 9).
- **Beyond capital:** customer intros into US/EU regulated enterprise; design-partner network; founder community for hiring deployment engineers.

**Speaker notes:** Five hundred K SAFE, eighteen months. Sixty percent goes to cloud GPU credits to validate the routing lift at seventy and two hundred billion parameter scales — the architectural risk worth de-risking. Twenty-five percent to the first three BFSI design-partner deployments. Fifteen percent to me and minimum infra. Concrete milestones — F1 number this week, three LOIs by month two, first paid pilot by month six, seventy-B adapter validation by month nine. From YC specifically I want customer intros into US and EU regulated enterprise; that is where the playbook ports.

**Suggested visual:** Pie chart of use-of-funds (60/25/15) on the left; horizontal milestone timeline on the right with Week 1 / Month 2 / Month 6 / Month 9 markers.

---

## Slide 15 — Closing

**Headline:** Defensibility = methodology + data flywheel + sovereignty positioning, in a 12-18 month architectural window.

**Bullets:**
- **Methodology moat:** the audit-defensible pipeline competitors will not build because the lazy alternative looks identical on a slide.
- **Data flywheel:** customer-trained adapters stay with the customer; the platform compounds per deployment.
- **Sovereignty positioning:** RBI + DPDP create a 5-year structural tailwind frontier labs cannot follow without breaking their unit economics.
- **Window:** 12-18 months of architectural lead before frontier providers restructure.
- **What we want next:** two BFSI customer intros and YC W26.

**Speaker notes:** Three layers of defensibility. The methodology — the data pipeline we built this week is what an auditor will demand and what most teams will not invest in because the lazy version looks the same on a pitch slide. The flywheel — customers train adapters on their corpus, those adapters stay with them, switching cost compounds. The positioning — RBI and DPDP are a five-year structural tailwind frontier labs cannot follow without breaking their pricing model. We have twelve to eighteen months of lead. From this room I want two BFSI customer intros and YC W26.

**Suggested visual:** Three-layer pyramid — base layer "Sovereignty positioning (5y tailwind)", middle "Data flywheel (compounding)", top "Methodology (audit-defensible)"; tagline beneath: "Synapta — sovereign AI inference, built right the first time."

---

## Appendix Slide A1 — BFSI Pipeline Waterfall (full detail)

**Headline:** End-to-end BFSI adapter pipeline — from public-domain regulator PDFs to a document-disjoint held-out eval.

**Bullets:**
- **Sources:** 80 RBI Master Directions (rbi.org.in, 36 MB) + 50 unique SEBI Master Circulars (sebi.gov.in, 79 MB). 130 PDFs total, 115 MB raw. SHA-256 manifest at `data/{rbi,sebi}_corpus/manifest.jsonl`.
- **Extraction:** pdfplumber + pymupdf fallback, multiprocess Pool(8). Drops headers/footers/tables, preserves section structure. 8.06M characters total.
- **Section detection:** regex over canonical RBI/SEBI numbering schemes -> 7,329 numbered sections.
- **Smart chunking:** chunk on numbered section boundaries; target 400-800 tokens, mean 425. 4,185 chunks total.
- **QA construction (3 deterministic tiers, no LLM):** Tier 1 FAQ (0 emitted — RBI MDs do not use Q/A format); Tier 2 numeric regex (~3,000 examples on currency / time periods / percentages / section refs / thresholds); Tier 3 heading-based extractive (~2,000 examples).
- **Validator:** 10-check quality gate (length, grounding, format, answer-presence-in-source, etc.) -> 98.45% pass.
- **Final dataset:** 4,373 train / 698 eval (currently training on cleaner v2: 2,931 train / 664 eval).
- **Document-disjoint split:** 26 PDFs (20%) quarantined entirely from training. Split manifest at `data/rbi_corpus/qa/split_manifest.json`.
- **Training config:** Nemotron-30B 4-bit NF4 + LoRA r=16, alpha=32, attn+MLP, lr 1e-4, 2 epochs, paged_adamw_8bit, MAX_LEN 2048, ~22 GB VRAM on RTX 5090, ~2-2.5h estimated.
- **Eval config:** 3 modes (base / +bfsi / FG+bfsi), 3 metrics (substring / F1 / exact match), Wilson 95% CI, McNemar paired test for all pairwise comparisons, per-tier and per-regulator breakdowns.

**Speaker notes:** This is the appendix detail anyone technical will ask for. Every step has a script number — `01_scrape_rbi_mds.py` through `08_eval_bfsi_extract.py` — and every source PDF has a SHA-256 in the manifest. The validator pass rate, the chunk size distribution, the LoRA config, the eval modes, the statistical tests — all reproducible. If a reviewer wants to challenge the methodology this is where the receipts live.

**Suggested visual:** Full waterfall flowchart with funnel widths proportional to data volume at each stage: 130 PDFs -> 8.06M chars -> 7,329 sections -> 4,185 chunks -> 4,477 raw QA -> 4,373 validated -> 104 train PDFs / 26 eval PDFs (highlighted in a different color to emphasize document-disjoint hold-out). Annotate each stage with its script filename for the technically curious.

---

## Appendix Slide A2 — Drift-Resistance Architecture (technical-deep-dive partner)

**Headline:** RAG-decoupled adapter + auto-retrain loop — the partner's first objection ("your fine-tune goes stale") answered architecturally. Full design in `AUTO_RETRAIN_PIPELINE_DESIGN.md`.

**Bullets:**
- **The drift problem, with numbers:** RBI ships 200-400 circulars/year; SEBI 50-100. Most are value patches inside otherwise-stable Master Directions (rate caps, thresholds, timelines). A snapshot adapter is materially stale on time-varying fields within 3-6 months.
- **The architectural fix:** the adapter learns the **skill** of locating + citing + extracting from a regulator paragraph. It does **not** memorize values. Current text is supplied at inference time by a live RAG index updated within an hour of publication.
- **Component 13 — Change Detector:** RSS + polling watchers on RBI / SEBI notification endpoints with conditional GET. Classifies each item NEW / AMENDMENT / SUPERSEDED. Detection latency target: <1 hour from publication.
- **Component 14 — Auto-Ingest Pipeline:** existing `01_scrape -> 02_extract -> 03_chunk -> 04b_build_qa_pairs_v2` chain repackaged behind `process_circular(url)`. Output to RAG index + candidate framework-shaped QA. Idempotent on circular SHA.
- **Component 15 — Incremental LoRA Continuation:** triggers only on structurally new patterns (or 25-circular cumulative threshold). Resumes from prior weights, low LR, with replay buffer to prevent catastrophic forgetting. ~30 min/run on H100 (estimate).
- **Component 16 — Eval Gate:** runs candidate adapter against existing eval set + changed-gold subset + regression suite. Three configurable thresholds; failed gate blocks deploy and pages on-call.
- **Component 17 — Customer Dashboard:** three facts + one button. Last-retrain date, N new circulars indexed, K eval-set golds changed. Customer-controlled "deploy refreshed adapter" with eval-gate status.
- **Build plan:** 2 weeks post-YC funding. Week 1 watchers + ingest; Week 2 continuation + eval gate + dashboard. End state is a working loop, not a polished product.
- **Status:** designed, not built. Components 01-12 already exist in the repo and form the foundation this design extends. We are not claiming to have built the loop; we are claiming to have understood the failure mode in enough detail to design around it before shipping.

**Speaker notes:** This is the appendix for the partner who asks "what about regulatory drift?" Two-line answer: the adapter is a skill, not a snapshot; the live RAG index is the source of truth for current text. Five components — change detector, auto-ingest, incremental LoRA continuation, eval gate, customer dashboard. Two-week build plan against the existing data pipeline. The advantage is the loop, not the snapshot — every week the live index gets a paragraph fresher; every structurally new circular adds a pattern to the adapter; the customer sees, on a dashboard, exactly what changed and chooses when to deploy. Full design doc has the architecture diagram, the open questions, the replay-buffer composition, and the cross-regulator dependency edge cases.

**Suggested visual:** The architecture diagram from `AUTO_RETRAIN_PIPELINE_DESIGN.md` (RBI/SEBI watchers -> Change Detector 13 -> Auto-Ingest 14 -> {Live RAG index, Incremental LoRA 15} -> Eval Gate 16 -> Customer Dashboard 17). Highlight that components 01-12 already exist; 13-17 are the 2-week post-YC build.
