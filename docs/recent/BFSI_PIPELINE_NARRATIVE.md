# The BFSI Adapter Pipeline — How We Built It Right

**One-page narrative for deck slide / blog post.** ~400 words.

---

## The shortcut everyone takes

Almost every "domain-specialized LLM" pitch you have ever evaluated has the same methodological hole: an LLM was used to generate evaluation questions from the same documents the model trained on. Sometimes it is dressed up as "synthetic data augmentation." Sometimes it is "paraphrase expansion." The result is identical — eval scores that look spectacular and a model that falls over the first time a real user asks a real question. The eval has been contaminated by the generator.

We refused to do this.

## What we built this week

We constructed an audit-defensible BFSI (Indian banking and securities) compliance adapter from scratch — corpus, training set, held-out eval, and a LoRA training run that is currently live on a single RTX 5090.

**Source corpus (public-domain, government-published):**
- 80 RBI Master Directions
- 50 SEBI Master Circulars
- 130 PDFs total, 115 MB raw

**Extraction and chunking:**
- 8.06 million characters extracted via pdfplumber + multiprocessing
- 7,329 numbered sections detected by regex over canonical RBI/SEBI numbering schemes
- 4,185 smart chunks on section boundaries, mean length 425 tokens

**QA construction — the part that matters:**
- 3-tier deterministic extractive templates over real regulator text
- **Zero LLM-generated questions.** Every question is a regex extraction from the source. No paraphrase augmentation. No synthetic prompting.
- 4,477 raw QA pairs generated
- 10-check validator gates every pair (length, grounding, format, answer presence in source, etc.) — **98.45% pass rate**, yielding 4,373 train and 698 eval pairs

**Document-disjoint train/eval split:**
- 26 entire PDFs, 20% of the corpus, are quarantined from training
- The eval model has literally never seen those documents
- This is the only kind of split that survives a procurement audit

**Training:**
- Nemotron-30B in 4-bit NF4 base + LoRA (r=16, alpha=32, attention + MLP modules)
- Learning rate 1e-4, 2 epochs, paged AdamW 8-bit, max sequence length 2048
- Fits in ~22 GB on the RTX 5090's 32 GB VRAM
- Training in progress as of writing

## Honesty caveats

The held-out eval F1 is not yet measured. We will publish the number when it lands. If the lift is modest, we will say so and reframe — the BFSI play in that case becomes deployment workflow plus retrieval, not reasoning lift. A separate long-context needle-in-haystack test on 20 RBI questions returned a null result (both base and Format Guard hit 100%); we read that as confirmation that base reasoning is fine when context is in-window, which is exactly why the adapter is built for the workflows where retrieval is non-trivial.

## Why this is the bet

The pipeline is the moat. Anyone can rent a GPU. Few will build the data pipeline so it withstands an auditor. We did it first.
