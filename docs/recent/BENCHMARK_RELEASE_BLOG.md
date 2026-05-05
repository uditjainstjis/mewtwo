# Releasing the first auditor-grade Indian BFSI extractive QA benchmark

Today we are open-sourcing the **Synapta Indian BFSI Benchmark v1**: 60
hand-curated extractive question-answering items drawn from Reserve
Bank of India (RBI) Master Directions and Securities and Exchange
Board of India (SEBI) Master Circulars, with a deterministic reference
scorer.

It is released under **CC-BY-SA-4.0** and is available now on
HuggingFace Datasets and Kaggle, with the construction code on GitHub.

- HuggingFace: <https://huggingface.co/datasets/synapta/indian-bfsi-bench-v1>
  (placeholder URL)
- Kaggle: <https://www.kaggle.com/datasets/synapta/synapta-indian-bfsi-benchmark-v1>
  (placeholder URL)
- GitHub: <https://github.com/synapta/indian-bfsi-bench> (placeholder)

## Why nobody had done this

There is, at the time of this release, **no public extractive-QA
benchmark targeted at Indian financial regulation**. The major
contenders address adjacent problems:

- **LegalBench** evaluates legal reasoning but draws almost entirely
  from U.S. case law and contract clauses.
- **FinanceBench / FinQA / TAT-QA** target U.S. 10-K financial
  statements and tabular reasoning.
- **SQuAD-family** benchmarks have neither the regulatory genre nor
  the Indian-English drafting conventions: `lakh` and `crore`, `Rs.`
  and `INR`, paragraph-numbered Master Directions, embedded section
  cross-references.

For teams building NLP for Indian banks, insurers, fintechs, NBFCs
and the regulators themselves, this gap means model selection is
usually anecdotal. We hit this wall internally when trying to track
regression on a domain-adapted model and realised the right move was
to publish the instrument we built for ourselves.

## What we did differently

Three deliberate choices distinguish v1 from a generic web-scraped QA
set.

**1. Document-disjoint held-out split.** All 60 questions come from
PDFs that never appear in any training partition of the underlying
RBI/SEBI corpus. The disjointness is an `assert` in
`build_benchmark.py`, not a promise. If you train on the public
corpus and report v1 numbers without re-asserting disjointness, your
result is meaningless.

**2. Deterministic templates, not LLM generation.** Every candidate
question and every gold answer was produced by deterministic
pattern-matching templates over headings, numeric spans, and
regulation references. **No language model authored a question or an
answer.** This matters: LLM-generated benchmarks routinely smuggle in
the distributional fingerprint of the generator and reward downstream
systems that share that fingerprint. We did not want to publish a
benchmark whose ground truth was secretly "what GPT-4 thinks the
answer is."

**3. A strict hand-validated quality gate.** Candidates were filtered
on context length, anaphora, answer shape (tier-2 must contain a
digit, currency or section reference under 80 chars; tier-3 must be
under 250 chars and not start with a stopword), and trailing-phrase
length. Then a stratified greedy procedure picked 60 questions under
simultaneous regulator-balance, tier-balance, per-PDF-cap, and
topic-diversity constraints, breaking ties by sorted `qa_id` for full
reproducibility.

Each retained record carries a difficulty tag, a topic tag,
0-3 alternative-answer normalisation variants
(`Rs. 21` / `Rs.21` / `INR 21` / `1 lakh` for `0.1 million`), and a
fixed `scoring_method` so cross-system comparisons are well-defined.

## Scoring you can trust

`scoring.py` ships three deterministic functions - exact match,
substring match, token-F1 - and a CLI that reports the primary score
with a Wilson 95% confidence interval. With two `--predictions`
arguments it also runs a paired McNemar test (statsmodels first,
scipy.binomtest fallback, pure-Python fallback if neither is
installed). Significance is something benchmark cards usually skip;
we're treating it as table stakes.

## Honest about coverage

v1 reflects the held-out document set we had at the time. The topic
histogram skews **governance-heavy**: 22 of 60 items are tagged
`governance`, 12 `mutual_fund`, 9 `foreign_exchange`. KYC, AML / PMLA,
capital adequacy, ATM/UPI, NBFC supervision, outsourcing, related-
party transactions and stress-testing are **under-represented or
absent** in v1, because no Master Direction on those subjects landed
in the held-out split. This is an artefact of the partition, not of
intent.

If your application is sensitive to those topics, treat v1 as a
**floor**, not a ceiling: a model that does badly on v1 will likely do
badly on the omitted topics too, but the converse does not hold. v2
will broaden the corpus to fill these gaps.

## Why CC-BY-SA-4.0, not "gated"

An earlier draft of this benchmark was distributed under a gated
named-evaluator agreement specifically to keep the question text out
of pretraining corpora. We changed our minds. Gated benchmarks are
useful for one team for a few months; they don't move the field. The
trade-off we now accept is that v1 will eventually appear in web
crawls, and we will refresh from undisclosed documents in v2 to keep
the family useful.

CC-BY-SA-4.0 (rather than Apache-2.0) was chosen because the artefact
is data, not code, and the ShareAlike clause keeps community
extensions and translations open as well.

## Baselines: deliberately empty

The baselines table in the dataset card ships **without numbers**.
We will not seed it with internal-only results; we want the first row
to come from a public, reproducible run. Synapta will publish ours
shortly with the exact prompt and harness.

## Call for contributions

If you can help, three things would move the field forward:

1. **Run a baseline.** Pick any frontier or open-weights model, run it
   over `(context, question)` pairs, score with `scoring.py`, and open
   a PR adding a row to the baseline table with your prompt and harness.
2. **File errata.** If a gold answer is wrong, ambiguous, or
   superseded by a regulator amendment, open an issue with the
   `benchmark_id` and the corrected text.
3. **Send v2 candidates.** If you have a Master Direction or Master
   Circular that v1 missed, send the source PDF (or a stable RBI/SEBI
   URL) plus 1-3 candidate `(context, question, gold_answer)` triples
   in the schema documented in the README.

PRs welcome on the GitHub repo. We will review on a rolling basis and
credit contributors in the dataset card and in v2's release notes.
