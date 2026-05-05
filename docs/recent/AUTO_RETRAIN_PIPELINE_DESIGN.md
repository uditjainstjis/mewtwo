# Auto-Retrain Pipeline: Design for Regulatory Drift

**Status:** Design proposal. Components 13-17 below are unbuilt. Components 01-12 (`synapta_src/data_pipeline/01_*` through `12_*`) already exist in the repo and form the foundation this design extends.

---

## TL;DR

RBI publishes 200-400 circulars per year. SEBI publishes 50-100. Each can amend a specific value (a rate cap, a threshold, a timeline) inside an otherwise stable Master Direction. A LoRA adapter trained on a snapshot today is materially stale on time-varying fields within 3-6 months. If a customer asks "what is the current cap on X" and we answer from a six-month-old snapshot, we are wrong in production. That is the existential threat to any "fine-tune once, sell forever" pitch.

The fix is architectural. The adapter does not memorize values. It learns the *skill* of reading regulatory text - locating the relevant paragraph, citing it, extracting the value present in that paragraph. Current text is supplied at inference time by a live document feed (RAG) that is updated within an hour of publication. A watcher detects new circulars, an auto-pipeline ingests and chunks them, and an incremental LoRA continuation run updates the skill only when new structural patterns appear. Customers see, on a dashboard, exactly what changed since their adapter was last refreshed and choose when to deploy.

---

## The Drift Problem, with Numbers

RBI's notification feed (`rbi.org.in/Scripts/NotificationUser.aspx`) historically averages 200-400 circulars/year across departments. The Master Direction index (`BS_ViewMasDirections.aspx`) is re-consolidated as MDs are amended. SEBI's notification stream (`sebi.gov.in/sebiweb/home/HomeAction.do`) runs 50-100 circulars/year, clustered around disclosure cycles.

Most circulars do not introduce new regulated activities. They patch values inside an existing framework: a tenor extension, a reporting threshold change, a revised LCR calibration, a fee cap. This is precisely the failure mode for a static fine-tune. The framework ("how is X structured?") is stable for years. The values ("what is the cap on X today?") drift on a weekly cadence.

We do not have measured production drift numbers - we have no production deployment. The 3-6 month window is an estimate from sampling amendment frequency on tracked MDs, to be measured once the watcher is live.

---

## Architecture

```
+------------------+    +------------------+    +----------------------+
|  RBI watcher     |    |  SEBI watcher    |    |  Manual upload       |
|  (RSS / poll)    |    |  (RSS / poll)    |    |  (customer circular) |
+--------+---------+    +--------+---------+    +----------+-----------+
         |                       |                          |
         +-----------+-----------+--------------------------+
                     |
                     v
         +---------------------------+
         |  13. Change Detector      |
         |  - new vs amend vs        |
         |    superseded             |
         +-------------+-------------+
                       |
                       v
         +---------------------------+         +------------------------+
         |  14. Auto-Ingest Pipeline | ------> |  Live RAG index        |
         |  (PDF -> text -> chunk    |         |  (current text, served |
         |   -> framework-shaped QA) |         |   to adapter at infer) |
         +-------------+-------------+         +------------------------+
                       |
                       v
         +---------------------------+
         |  15. Incremental LoRA     |
         |  continuation training    |
         |  (resumes from prior      |
         |  weights, ~30 min GPU est)|
         +-------------+-------------+
                       |
                       v
         +---------------------------+
         |  16. Eval Gate            |
         |  - regression suite       |
         |  - changed-gold detection |
         +-------------+-------------+
                       |
                       v
         +---------------------------+
         |  17. Customer Dashboard   |
         |  + deploy controller      |
         +---------------------------+
```

The adapter is the constant. The text it reads is the variable. The pipeline keeps the variable fresh and only retrains the adapter when something *structurally* new appears (a new disclosure form, a new regulated activity).

---

## Component-by-Component

### Component 13: Change Detector

A small service that polls the RBI and SEBI notification endpoints. RBI's page exposes per-section listings; SEBI's is an ASP.NET POST form. Both have historically offered section-level RSS - we'll use RSS where reliable and fall back to polling the index pages with conditional GET (`If-Modified-Since`, ETag) on a 30-60 min cadence. Target detection latency: under 1 hour from publication.

Each new item is classified:

- **NEW** - unrelated circular on a fresh topic. Goes into ingestion.
- **AMENDMENT** - modifies values inside a tracked MD. Diffed against the prior version; the diff drives an "eval gold update" candidate list.
- **SUPERSEDED** - replaces a prior circular outright. Prior is marked retired in the RAG index.

Classification is rule-based first (regex on title for "amendment to", "supersession of", "withdrawal of"), with an LLM call on ambiguous cases. We err on the side of "treat as amendment, surface to a human."

### Component 14: Auto-Ingest Pipeline

The existing `01_scrape -> 02_extract -> 03_chunk -> 04b_build_qa_pairs_v2` chain, repackaged to run on a single-circular input rather than a full corpus refresh. Output goes two places: the RAG index (chunks + metadata, served at inference) and a candidate training-data file (framework-shaped QA pairs only - see below). Idempotent on circular SHA.

### Component 15: Incremental LoRA Continuation

Most days, no continuation runs - only the RAG index updates. Continuation triggers when:

1. A NEW circular introduces a structural pattern absent from the current training distribution (new section template, new disclosure form, new citation style), OR
2. Cumulative un-ingested NEW circulars since the last run exceeds a configurable threshold (default 25).

Continuation resumes from prior adapter weights at a low learning rate, on a small batch composed of (a) new framework-shaped QA pairs and (b) a replay sample from the existing training set to prevent catastrophic forgetting. Estimated wall time on a single H100: ~30 min per run - estimate, not measurement; we'll calibrate after the first three runs.

### Component 16: Eval Gate

Runs the candidate adapter against (a) the existing eval set, (b) a "changed-gold" subset where the gold answer is known to have moved due to recent amendments, and (c) a regression suite of historically-failing questions we have fixed. Three thresholds, configurable per customer:

- Overall exact-match must not regress more than X points.
- Changed-gold accuracy must improve by at least Y points.
- No regression-suite item may flip from pass to fail.

A failed gate blocks deployment and pages the on-call. A passing gate auto-deploys (if opted in) or produces a deploy-ready artifact.

### Component 17: Customer Dashboard

One screen, three facts:

1. Your adapter was last retrained on `<date>`.
2. `<N>` new RBI circulars and `<M>` new SEBI circulars have been indexed since then. (Click to see titles.)
3. `<K>` questions in your eval set have a changed gold answer due to those circulars. (Click to see diffs.)

A single button: "Deploy refreshed adapter (eval gate: PASS)." The customer is in the loop on every change. They never wake up to a silently-updated model.

---

## Training Data Philosophy: Frameworks Beat Values

Inside `data/rbi_corpus/qa/train_clean.jsonl` (2,931 rows) the data already splits cleanly into two tiers:

- **Tier 2 (1,925 rows) - value-shaped.** Example: *"What monetary value is prescribed for Lending for acquiring shares under the ESOP under paragraph 3.2?"* Gold: `INR 20 lakh`. If the RBI raises that ceiling tomorrow, this row's gold is wrong.
- **Tier 3 (1,006 rows) - framework-shaped.** Example: *"What is provided under paragraph 5... concerning INR loans by Indian body corporate to its NRI/PIO employees?"* Gold: a paragraph-quoted summary of the rule's *structure*. Unchanged across most amendments.

For v1 we built both. For v2 the dataset should invert the tier-3 to tier-2 ratio and add a tier focused purely on extraction-given-context: the model is given the paragraph and asked to extract the value, never to recall it from memory. The skill becomes "read the paragraph in front of you, locate the value, cite it." That skill survives any number of value changes.

---

## Two-Week Build Plan (post-YC)

**Week 1 - watchers and ingestion.**

- Day 1-2: Component 13 RBI watcher. RSS first, polling fallback. Queues every detected item.
- Day 3: Component 13 SEBI watcher, same shape.
- Day 4-5: Component 14 single-circular runner. Wraps existing scripts behind `process_circular(url)`.
- Day 6-7: RAG index update path. End-to-end test: drop in a known recent circular, confirm the answer changes within 10 min.

**Week 2 - training, eval, dashboard.**

- Day 8-9: Component 15 continuation script. Replay buffer, small-batch fine-tune, checkpoint per run.
- Day 10-11: Component 16 eval gate. Wrap `08_eval_bfsi_extract` in a pass/fail decision with the three thresholds.
- Day 12-13: Component 17 dashboard. Deliberately ugly. Three facts and one button.
- Day 14: Dogfood on the last 30 days of real RBI/SEBI circulars. Report drift rate, gate pass rate, missed amendments.

End of week 2 is not a polished product. It is a working loop: a real circular on the live RBI site triggers a real adapter update under our control within an hour.

---

## Open Questions (Honest)

1. **Amend vs supersede classification.** Some RBI notifications quietly modify an MD paragraph without "amendment" or "supersession" in the title. Frequency unknown; we suspect 10-20% of amendments. If high, we need per-MD diff-aware ingest, not just title parsing.
2. **Replay buffer composition.** How much prior training data to mix in to prevent catastrophic forgetting on framework-shaped questions is unknown. Default 1:1 is a guess; calibrate over the first half-dozen runs.
3. **Cross-regulator dependencies.** SEBI rules cite RBI definitions and vice versa. When a cited definition changes, the citing rule's *meaning* changes even though its text didn't. No design yet beyond "flag and surface to human reviewer."
4. **Customer-specific adapters at scale.** With 50 customers each on a customized adapter, do we run 50 continuation jobs per batch, or one base-model continuation that all customer LoRAs sit on top of? The latter assumes a layered LoRA stack we haven't validated.
5. **Eval gate gaming.** The changed-gold subset is generated by the same pipeline that trains the adapter - real risk of teaching to the test. We need an independent eval source, probably a small expert-curated set the training pipeline never sees.

---

## How This Turns the YC Partner's Objection into a Positioning Advantage

The objection: "your fine-tune goes stale, customers will see wrong answers, your moat evaporates." It assumes the adapter is the source of truth. In our architecture it isn't. The adapter is a skill. The source of truth is the regulator's website, mirrored into our RAG index within an hour of publication. Every answer is grounded in text current as of right now.

The advantage is the loop, not the snapshot. A competitor selling a fine-tune of GPT-4-on-RBI-corpus has a product that decays. We have a product that gets *more* accurate every week: each circular adds a paragraph to the live index and, when structurally new, another framework-shaped pattern to the adapter. The dashboard makes the loop visible - customers see what we've ingested, what changed, and they choose when to deploy. That is the customer flywheel claim, made concrete and defensible against the staleness objection.

We are not claiming to have built this. We are claiming to have understood the failure mode in enough detail to design around it before shipping. The two-week plan turns the design into a working loop.
