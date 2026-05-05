# Bounded Multi-Adapter Composition on Apple Silicon: A Negative-to-Mixed Results Study with a 1.5B Base

## Abstract

We report a multi-phase empirical study of prompt-level multi-adapter
composition on a 1.5B parameter base model running on Apple Silicon. The
working system, Synapta, attaches up to twenty domain LoRA experts to
Qwen2.5-1.5B-Instruct (4-bit MLX) and composes them at inference time with
two clamped activation-mixing variants (per-adapter weight-cap and per-layer
norm-ratio) under several routing schemes. Across three pre-registered
hypothesis sets we find that bounded $K{=}2$ composition does **not** beat
single-adapter routing on a single-domain benchmark (v1: $\Delta_\text{SIM}
= -0.011$, $n{=}100$); on a multi-domain benchmark designed to favour
composition (v2 MD, $n{=}40$) the gain is directionally positive but
sub-threshold ($+0.0171$ vs a pre-registered bar of $+0.03$). A clamp
ablation shows the per-layer norm-ratio variant is empirically identical to
weight-cap ($\Delta = -0.0003$), revealing the failure as a property of the
base/adapter geometry rather than the clamp implementation. An oracle vs
real-router gap analysis bounds the achievable headroom at $+0.0206$, of
which a CoT top-2 router recovers $\sim 26\%$. Replacing the CoT router with
an SFT classifier raises routing accuracy from $48.7\%$ to $85\%$, while a
DPO post-training step regresses it to $42\%$. An externally authored,
blindly judged 30-question rubric does not support the prior internal
"Qwen Synapta beats Mistral-7B" headline (Mistral wins 23-26 of 30 across
three Qwen variants). Finally, a TCAR collaborative-inference loop
(parallel-branch + refiner) approaches Mistral-7B on semantic similarity
($0.6900$ vs $0.6907$) at roughly one-quarter the VRAM, but loses on token
F1 ($-0.0205$) and pays a $\sim 2.3\times$ latency penalty. We frame the
work as an honest mixed-and-negative-results paper about when composition
helps, what evaluation infrastructure is required to detect it, and how
quickly internal benchmarks over-rank methods that an external blind judge
disagrees with.

---

## 1. Introduction

Multi-adapter composition on a small base model is an attractive deployment
story for Apple Silicon: customer-owned hardware, no cloud round-trip, and
unified memory architectures that can hold a base model plus a non-trivial
set of low-rank adapters in VRAM. The intuition is straightforward. Train
a pool of single-domain LoRA experts, route to two of them at inference
time, mix their contributions linearly with a bounded clamp, and obtain a
composed model that handles cross-domain queries that no single adapter
covers cleanly. If this works, a 1.5B base on a laptop GPU can in principle
substitute for a single 7B generalist on the multi-domain queries that
matter for an end-user assistant.

The empirical question is whether it *actually* works. This paper reports
what happens when one tries: a sequence of pre-registered experiments on
Qwen2.5-1.5B-Instruct (4-bit MLX) with twenty domain LoRA experts. The
study went through three phases:

1. **v1: Bounded prompt-level composition.** A first benchmark of $n{=}100$
   single-domain questions, four methods (Baseline, SingleAdapter,
   AdaptiveClamp $K{=}2$, UnclampedMix), and a pre-registered headline
   hypothesis $\Delta_\text{SIM}(\text{AC} - \text{SA}) > +0.05$. The
   headline failed: AC scored $-0.011$ similarity below the
   single-adapter baseline.

2. **v2: Multi-domain composition with a clamp ablation and a routing-gap
   study.** A second benchmark with an explicit multi-domain split
   ($n{=}40$ MD), a per-layer norm-ratio clamp variant, and an oracle vs
   real-router comparison. The MD compositional gain came in at $+0.0171$,
   directionally positive but well below the pre-registered $+0.03$ bar.
   The norm-ratio clamp turned out to be operationally identical to the
   simpler weight-cap, and the real CoT top-2 router recovered only
   $\sim 26\%$ of the oracle's small ceiling.

3. **External evaluation, router upgrade, and TCAR.** A 9-technique
   injection ablation on an externally authored 40-question MD benchmark
   ranked methods differently from the internal benchmark, motivating a
   pivot to external blind judging. A 30-item stratified blind rubric
   directly disagreed with the prior "Qwen Synapta beats Mistral" claim
   (Mistral wins 23-26 of 30). To upgrade the router we trained a small
   classifier-style SFT router on 5000 synthetic routing examples, taking
   exact-match routing accuracy from $48.7\%$ (CoT) to $85\%$; a
   subsequent DPO post-training step regressed it to $42\%$. Finally we
   replaced activation-space weight blending with TCAR, a
   parallel-branch + refiner collaborative-inference loop. TCAR with the
   DPO router approaches Mistral-7B on semantic similarity but loses on
   token F1 and pays a $\sim 2.3\times$ latency penalty.

The contribution of this paper is therefore not a method that wins on a
leaderboard. It is a structured negative-to-mixed result, paired with an
evaluation methodology that we believe is essential for distinguishing
"composition helps" from "the internal benchmark is rewarding the wrong
thing." Specifically we make four claims:

- **C1.** Bounded prompt-level $K{=}2$ composition on a 1.5B base with up
  to twenty single-domain LoRA experts does not clear a pre-registered
  $+0.03$ similarity bar on multi-domain queries, even with oracle
  routing. The bottleneck is the underlying base/adapter geometry, not
  the clamp choice or the router.
- **C2.** A per-layer activation norm-ratio clamp is empirically
  indistinguishable from a per-adapter weight-cap on this configuration
  ($\Delta = -0.0003$). The clamp's only earned-keep on this base is
  preventing the $\sim 8\%$ catastrophic collapse seen in v1
  UnclampedMix.
- **C3.** A small SFT classifier router substantially outperforms both
  CoT-generative routing ($85\%$ vs $48.7\%$ exact-match) and a DPO
  post-trained variant ($85\%$ vs $42\%$). Pairwise-preference DPO
  objectives can degrade classification quality even when they optimise
  the stated objective.
- **C4.** Internal-similarity rankings over-rank Qwen Synapta methods
  that an externally authored blind rubric judge disagrees with. The
  external benchmark and TCAR collaborative system together yield a
  near-match to Mistral-7B on similarity, at roughly one-quarter the
  VRAM, but with a token F1 loss of $-0.0205$ and a $\sim 2.3\times$
  latency penalty. "Synapta beats Mistral" is not supported by the
  external evidence.

The rest of the paper develops these claims. We are deliberate about what
we are *not* claiming. We do not claim that composition beats single-
adapter routing on multi-domain queries; we do not claim that the
norm-ratio clamp is essential; and we do not claim that the small Synapta
stack dominates a 7B baseline on quality. We treat the v2 sub-threshold
result as a *failure* of the pre-registered hypothesis, not a victory, and
we treat the TCAR similarity match as a system-level near-tie that comes
with explicit F1 and latency tradeoffs.

## 2. Background and Related Work

**LoRA adapters and composition.** Low-rank adaptation introduces a small
trainable additive update $\Delta W = AB^\top$ to a frozen base weight
$W$ at low parameter cost. A natural composition idea is to attach
multiple such updates simultaneously, summing their contributions, with
some routing scheme deciding which adapters are active per query. This
study works at this prompt-level granularity: two adapters are selected
once per query and applied for the entire generation.

**Activation-space mixing and clamps.** A naive sum of adapter
contributions can drive the activations of the base model out of
distribution, particularly when several adapters are simultaneously
active or when their contributions are in the same direction. A clamp
limits the magnitude of the adapter update relative to the base
activation. We compare two variants:

- **Per-adapter weight-cap:** scale the adapter contribution by a fixed
  scalar $c \le 1$ before summing. This is the simpler variant and the
  default in v1/v2.
- **Per-layer activation norm-ratio:** at each layer $\ell$, scale the
  adapter contribution $m_\ell$ by
  $\gamma_\ell = \min(1,\, c \cdot \|z_\ell\|_2 / \|m_\ell\|_2)$, where
  $z_\ell$ is the base-model activation. This is the variant introduced
  in v2 (Section 3.2) to test whether the clamp's behaviour is
  implementation-bound.

**Routing on small bases.** In our setting the router selects up to two
adapters per query (top-$K$ with $K \le 2$). We evaluate four router
families: a generative CoT router that emits a domain list as text
("first domain: ...; second domain: ..."); an embedding-centroid router
that scores adapters by query-to-centroid cosine; a small SFT classifier
trained on 5000 synthetic routing examples; and a DPO post-trained
variant of the SFT classifier. The base model used inside the router is
the same Qwen-1.5B stub, fine-tuned in a parameter-efficient way.

**Collaborative inference (TCAR).** Activation-space weight blending
mixes contributions inside the model. An alternative is to keep the
adapters separate at inference time, run two single-adapter forward
passes in parallel, and use a third forward pass (the *refiner*) to
synthesise the two candidate answers. We refer to this configuration as
TCAR (two-candidate adapter refiner). TCAR pays in latency (three
sequential forward passes per query in our implementation) but avoids
the activation-space clamp question entirely.

**Why this study is needed.** The prevailing internal narrative around
small-base multi-adapter composition is that activation-space mixing,
plus a router, *will* get a 1.5B base close to a 7B baseline on
multi-domain queries. The work in this paper started from that
assumption and ended at a different conclusion. We document the path
explicitly because the methodological lessons (pre-registered bars, the
oracle/real router gap, internal vs external benchmarks, blind rubric
judging) generalise beyond the specific Synapta stack. They are how we
ended up with quantitative evidence that contradicts the original
narrative; without them, the internal benchmark would still be telling
us what we wanted to hear.

## 3. Methods / System Design

### 3.1 Synapta architecture

The Synapta stack is a Qwen2.5-1.5B-Instruct base in 4-bit MLX format,
running on Apple Silicon (M-series, MLX runtime), with up to twenty
domain LoRA experts trained on synthetically templated single-domain
data. Domains include `MATHEMATICS`, `MEDICAL_DIAGNOSIS`, `PHILOSOPHY`,
`SANSKRIT_LINGUISTICS`, `MARITIME_LAW`, and others. At inference time a
router selects up to two adapters per query, and a composition method
combines their contributions according to one of the schemes in
Section 3.2.

The system is designed for a single-machine, customer-owned deployment
profile. Base model VRAM is $\approx 1.1$ GB (Qwen-1.5B 4-bit); each
adapter is $\sim 10$-$50$ MB (with sharing of frozen layers); two
adapters plus base fit comfortably on a consumer Apple Silicon GPU.
Mistral-7B-Instruct is used as a same-runtime baseline at $\approx 4.4$
GB and a measured mean latency of $10.65$ s on the external 100-item
benchmark.

### 3.2 v1 vs v2 benchmarks

**v1 (single-domain, $n{=}100$).** The first internal benchmark
contained 100 questions across 20 distinct domains, one domain per
question, synthetically templated. Methods compared at v1 were
Baseline, SingleAdapter (CoT routing, $K{=}1$, clamp $c{=}0.5$),
AdaptiveClamp (CoT, $K{=}2$, weight-cap $c{=}0.5$), and UnclampedMix
($K{=}2$, $c{=}999$). The pre-registered headline hypothesis was
$\Delta_\text{SIM}(\text{AC} - \text{SA}) > +0.05$.

**v2 (single + multi-domain, $n{=}100 + 40$).** The v2 benchmark
addresses the central methodological critique of v1: that single-domain
synthetic templates do not in fact stress composition. v2 has two
splits. The SD split contains 100 single-domain questions; the MD split
contains 40 questions whose construction explicitly requires two
distinct adapters (e.g.\ Sanskrit Linguistics + Ancient History). MD
labels include a required-adapter set, enabling oracle routing as a
ceiling. Pre-registered hypotheses span both splits and include both
similarity and PPL bounds. The H2 (MD compositional gain) bar is
$\Delta_\text{SIM} > +0.03$.

### 3.3 Clamp variants

Two clamps are compared:

- **Weight-cap.** Fixed scalar $c$ applied to the entire adapter
  contribution before summing into the base.
- **Norm-ratio.** At each layer, $\gamma_\ell = \min(1, c \cdot
  \|z_\ell\|_2 / \|m_\ell\|_2)$ scales the per-layer adapter
  contribution $m_\ell$ by the ratio of base activation norm to
  adapter activation norm. The intuition is that the adapter should
  never dominate the base activation by more than a factor of $1/c$.

The clamp ablation in Section 4.3 swaps weight-cap for norm-ratio
on the same v2 MD benchmark with oracle routing held fixed, isolating
the clamp variable.

### 3.4 Routing schemes

We evaluate four routers on the same $n{=}100$ holdout drawn from
`router_synthetic_routing_5000_valid_holdout.json`:

- **CoT (generative).** A Qwen-1.5B stub asked, in a short prompt, to
  list the two domains required for the query. Used in v1 and the v2
  real-router run.
- **Embedding (centroid).** Each adapter has a precomputed centroid in
  sentence-transformer space; the query is embedded once and assigned
  to the top-2 nearest centroids.
- **SFT classifier.** The same Qwen-1.5B stub fine-tuned with
  assistant-only loss masking on 5000 synthetic routing examples
  (`router_synthetic_routing_5000.json`) on Apple Silicon (MPS).
- **DPO post-trained.** A DPO post-training step on
  `router_reasoning_dpo_5000.jsonl`, response-only masked log-probs.

Routing-classifier metrics are exact-match accuracy on the required
domain set, partial-overlap accuracy (any-overlap), mean overlap F1, and
mean latency. The downstream effect on TCAR is measured separately on
the 100-item external benchmark (Section 3.6).

### 3.5 External MD benchmark and blind judging

After the v2 study, an externally authored 100-item multi-domain
benchmark (`multidomain_eval_claude_external_v2_100.json`) was
introduced as the headline external evaluation. The external set is
not internally authored: gold answers are externally provided rubric
responses, and the question style differs from v1/v2 internal templates.

A second sub-experiment runs a 30-item stratified blind pairwise
comparison: a Qwen variant and Mistral-7B each generate an answer; an
external rubric judge scores them blind to identity. We report Qwen
wins / Mistral wins / Ties for each of three Qwen variants
(`weighted_merge`, `late_layer_injection`, `sequential_reverse`) on
the same 30-item stratified subset.

### 3.6 TCAR collaborative inference

TCAR (two-candidate adapter refiner) has three forward passes per
query:

1. **Branch A.** The base model with adapter $a_1$ generates a candidate.
2. **Branch B.** The base model with adapter $a_2$ generates a candidate.
3. **Refiner.** A third forward pass takes the question and both
   candidates and emits a synthesised answer.

The router (SFT or DPO) selects $a_1$ and $a_2$. Branches and refiner
share the same Qwen-1.5B 4-bit base. The system is compared head-to-head
with Mistral-7B-Instruct on the 100-item external MD benchmark using
semantic similarity (sentence-transformers), token F1, exact match, and
mean latency.

### 3.7 9-technique injection ablation

Before TCAR was introduced, a 9-technique injection ablation tested
which prompt-level activation-space mixing strategy maximised internal
similarity. The nine techniques include `weighted_merge`,
`late_layer_injection`, `late_last_quarter`, `early_third_only`,
`sequential_token_segments`, `sequential_reverse`, `oracle_single_d1`,
`oracle_single_d2`, and `merge_high_clamp`; the same internal benchmark
methodology was repeated on a 40-question externally constructed MD
benchmark (`multidomain_eval_v2.json`, 360 inferences = 9 methods $\times$
40 questions). This ablation is the empirical basis for the move from
internal-similarity ranking to external blind judging in Section 3.5.

## 4. Experiments

We organise experiments along the chronological arc of the project, so
that each experiment's negative or sub-threshold result motivates the
next.

### 4.1 v1 prompt-level composition

The v1 experiment runs four methods on a single benchmark of 100
questions. Inferences total 400 (4 methods $\times$ 100). The
pre-registered hypotheses are H1 (compositional gain
$\Delta_\text{SIM} > +0.05$), H2 (PPL preservation), H3 (catastrophic
collapse on UnclampedMix). H1 is the headline.

### 4.2 v2 multi-domain benchmark

v2 runs the same four methods on 140 questions (100 SD + 40 MD).
Inferences total 560. The MD split uses oracle routing (required-
adapter labels) so that the routing variable is held fixed. The
pre-registered hypotheses include H1 (SD non-inferiority), H2 (MD
compositional gain $> +0.03$), H3 (PPL preservation), H4 (latency
overhead $\le 15\%$), H5 (clamped strictly $>$ unclamped on MD).

### 4.3 Clamp ablation: norm-ratio vs weight-cap

The clamp ablation runs three methods on the v2 MD split (40 questions,
120 inferences = 3 methods $\times$ 40): SingleAdapter (weight-cap
$c{=}0.5$, $K{=}1$), AC-v2-WeightCap ($c{=}0.5$, $K{=}2$), and
AC-v2-NormRatio (norm-ratio $c{=}0.5$, $K{=}2$). Oracle routing is
fixed across the three methods. The pre-registered hypothesis is
$\Delta_\text{SIM}(\text{NormRatio} - \text{WeightCap}) > 0$.

### 4.4 Routing gap: oracle vs real CoT top-2

The routing-gap study fixes the clamp at norm-ratio and varies the
router. The three methods are SingleAdapter (CoT, $K{=}1$),
AC-v2-Norm-RealRouter (CoT top-2, $K \le 2$), and AC-v2-Norm-Oracle
(oracle, $K{=}2$). Same 40-question v2 MD split, 120 inferences. We
report achieved $K$ (the routing system can return $K=1$ when only one
adapter scores above threshold) and decompose the gain into oracle
headroom and realised gain.

### 4.5 Router SFT and DPO

The SFT router is trained on 5000 synthetic routing examples with
assistant-only loss masking on Apple Silicon MPS, then evaluated on the
$n{=}100$ holdout for routing exact-match, partial-overlap, mean F1,
and mean latency. The DPO post-trained variant uses
`router_reasoning_dpo_5000.jsonl` (preference pairs) with response-only
masked log-probs; it is evaluated on the same holdout.

A 10-item TCAR pilot is run with each router so that the downstream
effect on a real generation pipeline can be observed. The full 100-item
TCAR comparison (Section 4.7) uses the DPO router after the
classification regression is observed; its inclusion is a deliberate
test of whether the routing-classifier regression carries over to
downstream answer quality.

### 4.6 9-technique injection ablation and external blind judge

The 9-technique ablation runs 360 inferences (9 methods $\times$ 40
external MD questions). Each inference produces semantic similarity,
token F1, exact match, and latency. Aggregates across the nine methods
are reported in Section 5.6.

The 30-item stratified blind judge runs three Qwen variants
(`weighted_merge`, `late_layer_injection`, `sequential_reverse`)
against Mistral-7B. Outcomes are recorded as Qwen wins, Mistral wins,
or Ties.

### 4.7 TCAR collaborative inference vs Mistral-7B

The final 100-item comparison runs four systems on the external MD
benchmark: TCAR + DPO router, Mistral-7B baseline, and the two best
prior Qwen static methods (`sequential_reverse`, `late_layer_injection`)
as system-level reference points. Metrics are semantic similarity,
token F1, exact match, and mean latency. A 10-item stratified pilot is
also reported.

## 5. Results

We organise results in the same order as Section 4 and present numbers
exactly as recorded in the corresponding primary artefacts.

### 5.1 v1: H1 fails

| Method | Sim | PPL | Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6196 | 64.5 | 2.803 |
| SingleAdapter | 0.6223 | 60.9 | 2.695 |
| AdaptiveClamp ($K{=}2$) | 0.6106 | 58.0 | 2.672 |
| UnclampedMix ($K{=}2$) | 0.5573 | 51.2 | 2.511 |

The pre-registered headline H1 ($\Delta_\text{SIM}(\text{AC} -
\text{SA}) > +0.05$) **fails**: the measured delta is $-0.011$,
i.e.\ AdaptiveClamp underperforms SingleAdapter on this single-domain
benchmark. The PPL hypothesis (H2: $\text{PPL}(\text{AC}) \le
\text{PPL}(\text{SA})$) passes ($58.0 \le 60.9$), and so does the
latency overhead bound ($-0.7\%$ relative to SA). The catastrophic
collapse hypothesis (H3) passes qualitatively: the UnclampedMix run
shows $\sim 8\%$ of prompts with similarity below $0.1$, the only
result in the v1 experiment that earns the clamp its keep.

A per-domain breakdown is mixed: AC wins on `MEDICAL_DIAGNOSIS`
($+0.030$) and `MATHEMATICS` ($+0.044$) but loses on `MARITIME_LAW`
($-0.145$) and `SANSKRIT_LINGUISTICS` ($-0.035$). The pattern is
consistent with a redundant-adapter-injects-noise interpretation: on
single-domain queries, the second adapter is at best irrelevant and
at worst a net negative.

The v1 result motivated the v2 benchmark, which explicitly includes a
multi-domain split.

### 5.2 v2: H2 sub-threshold positive (FAIL)

**SD split ($n{=}100$).**

| Method | Sim | PPL | Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6090 | 64.5 | 3.700 |
| SingleAdapter | 0.6064 | 60.9 | 3.571 |
| AdaptiveClamp-v2 | 0.6058 | 57.9 | 3.657 |
| UnclampedMix-v2 | 0.6041 | 52.3 | 3.623 |

**MD split ($n{=}40$).**

| Method | Sim | PPL | Latency (s) |
|---|---:|---:|---:|
| Baseline | 0.6473 | 12.7 | 4.059 |
| SingleAdapter | 0.6334 | 12.7 | 4.057 |
| AdaptiveClamp-v2 | 0.6505 | 12.6 | 4.090 |
| UnclampedMix-v2 | 0.6505 | 12.6 | 4.100 |

The pre-registered hypotheses pass on every axis except the headline:

- H1 (SD non-inferiority, $\ge -0.005$): pass at $-0.0006$.
- H2 (MD compositional gain, $> +0.03$): **FAIL** at $+0.0171$
  (directionally positive, sub-threshold).
- H3 (PPL preservation): pass on both splits.
- H4 (latency overhead $\le 15\%$): pass at $+1.9\%$.
- H5 (clamped strictly $>$ unclamped on MD): **FAIL** at $0.0000$ —
  AdaptiveClamp-v2 and UnclampedMix-v2 produce identical similarity to
  four decimal places.

The honest framing is that v2 does not clear the pre-registered
$+0.03$ bar on its headline hypothesis, and that the H5 failure
indicates the clamp is not doing what it was designed to do on this
configuration. The next two sections diagnose the H5 failure and
quantify the achievable headroom.

### 5.3 Clamp ablation: norm-ratio is operationally inactive

| Method | Clamp | Sim | PPL | Latency (s) | Mean $K$ |
|---|---|---:|---:|---:|---:|
| SingleAdapter | weight_cap | 0.6334 | 12.7 | 4.008 | 1.00 |
| AC-v2-WeightCap | weight_cap | 0.6505 | 12.6 | 4.055 | 2.00 |
| AC-v2-NormRatio | norm_ratio | 0.6502 | 12.6 | 4.221 | 2.00 |

The deltas are:

- $\Delta_\text{SIM}(\text{WeightCap} - \text{SA}) = +0.0171$ (matches v2)
- $\Delta_\text{SIM}(\text{NormRatio} - \text{SA}) = +0.0168$
- $\Delta_\text{SIM}(\text{NormRatio} - \text{WeightCap}) = -0.0003$

The norm-ratio clamp produces empirically the same similarity as
weight-cap on this base+adapter configuration; the difference
($-0.0003$) is well within sampling noise on $n{=}40$. Per-layer
inspection confirms why: the un-clamped adapter activation $\|m_\ell\|$
is already small relative to the base-model activation $\|z_\ell\|$,
so the norm-ratio scalar $\gamma_\ell$ evaluates to $1.0$ at almost
all layers and the additive composition is unchanged. The H5 failure
is therefore a property of the Qwen-1.5B + 20-expert geometry, not an
implementation artefact: there is no clamp variant that, holding
everything else fixed, distinguishes itself from no clamp. We
emphasise that this is a property of *this* configuration; on larger
bases or higher-rank adapters, the un-clamped adapter contribution
could plausibly grow large enough relative to the base activation that
the clamp becomes a non-trivial intervention.

### 5.4 Routing gap: small ceiling, $\sim 26\%$ recovered

| Method | Routing | Sim | PPL | Latency (s) | Mean $K$ |
|---|---|---:|---:|---:|---:|
| SingleAdapter | CoT ($K{=}1$) | 0.6296 | 12.8 | 4.178 | 1.00 |
| AC-v2-Norm-RealRouter | CoT top-2 | 0.6350 | 12.7 | 4.167 | 1.75 |
| AC-v2-Norm-Oracle | Oracle | 0.6502 | 12.6 | 4.211 | 2.00 |

The decomposition is:

- **Oracle headroom** $= +0.0206$ (Oracle vs SA).
- **Realised gain** $= +0.0054$ (RealRouter vs SA).
- **Routing gap** $= -0.0152$ (Oracle vs RealRouter).
- **Headroom recovered by router** $\approx 26\%$.

Two findings stand out. First, the *ceiling* is small: even with
oracle routing the multi-adapter composition gain on MD is $+0.0206$,
an order of magnitude below the v2 pre-registered $+0.03$ bar and well
below the field-standard $+0.05$ bar from v1. A better router cannot
rescue an architecture whose oracle ceiling is below threshold.
Second, the realised CoT top-2 router recovers only about a quarter of
that small ceiling, motivating the SFT/DPO router upgrade in
Section 5.5. But to be precise: even a perfect router on this
configuration would deliver $+0.0206$, not $+0.05$. Routing is a
bottleneck, but it is not the primary failure mode.

### 5.5 SFT vs DPO router

**Router classification accuracy on $n{=}100$ holdout.**

| Router | Exact | Partial overlap | Mean F1 | Latency (s) |
|---|---:|---:|---:|---:|
| CoT (generative) | $\approx 48.7\%$ | — | — | — |
| Embedding (centroid) | $78.7\%$ | — | — | — |
| **SFT** | $\mathbf{85.0\%}$ | $1.00$ | $\mathbf{0.945}$ | $1.079$ |
| DPO | $42.0\%$ | $0.75$ | $0.6333$ | $1.697$ |

The SFT router is the clear winner on classification: $85\%$
exact-match, $100\%$ partial-overlap, and a mean F1 of $0.945$. CoT
generative routing performs near random on this multi-label task, and
the embedding-centroid router sits in between. The DPO post-training
step *regresses* the router from $85\%$ to $42\%$ exact-match, despite
optimising the pairwise preference objective successfully. The lesson
is concrete: pairwise-preference DPO objectives can degrade
classification quality even when they optimise the stated objective.

A 10-item TCAR pilot does show a mild downstream improvement for the
DPO router (sim $0.7032$ vs $0.6902$, F1 $0.3046$ vs $0.2874$ on
$n{=}10$), which is the reason we still ran the full 100-item TCAR
comparison with DPO. We treat that 10-item bump as a small-sample
nuance, not as a defence of DPO; the 100-item result (Section 5.7)
must carry the weight.

### 5.6 9-technique injection ablation

On the externally authored 40-question MD benchmark (360 inferences =
9 methods $\times$ 40 questions), the best Qwen static methods cluster
at:

| Method | Sim | F1 | Latency (s) |
|---|---:|---:|---:|
| `sequential_reverse` | 0.6623 | 0.2734 | 4.605 |
| `weighted_merge` | 0.6592 | 0.2719 | 4.263 |
| `late_layer_injection` | 0.6594 | 0.2715 | 3.890 |
| Mistral-7B (baseline) | $\mathbf{0.6907}$ | $\mathbf{0.2917}$ | $10.654$ |

No Qwen technique beats Mistral on the external benchmark on either
similarity or F1. The internal vs external rankings disagree: the
internal benchmark ranked `sequential_reverse` highest, while the
external benchmark places `weighted_merge` and `late_layer_injection`
on roughly the same rung as `sequential_reverse`. The takeaway is not
that one technique is best; it is that the choice of injection
technique matters less than the underlying base/adapter geometry on
this 1.5B + 20-expert configuration. This finding, more than any
single number, is what motivated the move from internal-similarity
ranking to external blind judging.

The 30-item stratified blind judge shows:

| Qwen method | Qwen wins | Mistral wins | Ties |
|---|---:|---:|---:|
| `weighted_merge` | $6$ | $23$ | $1$ |
| `late_layer_injection` | $4$ | $26$ | $0$ |
| `sequential_reverse` | $4$ | $25$ | $1$ |

All three Qwen methods lose blind comparisons by margins of
$\sim 4$-$7$ Qwen wins to $\sim 23$-$26$ Mistral wins. The "Qwen
Synapta beats Mistral" headline that the internal benchmark suggested
is **not supported** by the external blind judge.

### 5.7 TCAR collaborative inference vs Mistral

The final 100-item external comparison:

| System | Sim | F1 | Exact match | Latency (s) |
|---|---:|---:|---:|---:|
| TCAR + DPO router | $0.6900$ | $0.2712$ | $0.0000$ | $24.198$ |
| Mistral-7B baseline | $\mathbf{0.6907}$ | $\mathbf{0.2917}$ | $0.0000$ | $10.654$ |
| `sequential_reverse` (Qwen) | $0.6623$ | $0.2734$ | $0.0000$ | $4.605$ |
| `late_layer_injection` (Qwen) | $0.6594$ | $0.2715$ | $0.0000$ | $3.890$ |

TCAR with the DPO router is the closest a Qwen-1.5B Synapta variant
gets to Mistral-7B on this external benchmark. On semantic similarity
the gap is $-0.0007$ (a near-tie within sampling noise). However:

- **F1 loss.** TCAR loses $-0.0205$ on token F1 against Mistral
  ($0.2712$ vs $0.2917$). The earlier "Synapta beats Mistral"
  narrative is not supported on F1 at $n{=}100$.
- **Latency penalty.** TCAR mean latency is $24.198$ s vs Mistral's
  $10.654$ s, a $\sim 2.3\times$ penalty. TCAR's three sequential
  forward passes (two branches plus a refiner) account for most of
  this overhead.
- **VRAM.** TCAR uses the same Qwen-1.5B 4-bit base ($\approx 1.1$ GB)
  as the rest of the Synapta stack, against Mistral-7B at
  $\approx 4.4$ GB. The VRAM advantage is real, at roughly one-quarter
  the Mistral footprint, but it is paid for in latency, not earned in
  quality.

The 10-item stratified pilot tells the same story
(`tcar_collaborative` $0.6797$ sim, $0.2682$ F1 vs Mistral $0.7067$
sim, $0.2971$ F1).

The honest summary of the TCAR result is that a 1.5B Apple Silicon
stack with parallel-branch + refiner can *approach* Mistral-7B on
similarity at lower VRAM, while losing on token F1 and paying a
$\sim 2.3\times$ latency penalty. We cannot characterise this as
beating Mistral on quality.

## 6. Limitations and Negative Findings

This is the section that anchors the paper's framing. We list the
limitations explicitly so that no quoted result detaches from its
caveat.

**L1. v1 H1 is a clean failure on the headline hypothesis.** Bounded
$K{=}2$ prompt-level composition does not beat single-adapter routing
on the v1 single-domain benchmark; the measured $\Delta_\text{SIM}$
is $-0.011$, against a pre-registered bar of $> +0.05$. The v1
benchmark is itself imperfect (single-domain synthetic templates), but
the v2 benchmark designed in response did not rescue the headline
either.

**L2. v2 H2 is sub-threshold, not a pass.** The MD compositional gain
is $+0.0171$, directionally positive but well below the pre-registered
$+0.03$ bar, and below the field-standard $+0.05$ bar from v1. We
treat the v2 result as a failure of the pre-registered hypothesis, not
as a "modest positive". A reader who reads only the number
$+0.0171$ without the threshold reads the wrong story.

**L3. The clamp is operationally inactive on this configuration.**
The norm-ratio variant differs from weight-cap by $-0.0003$ on $n{=}40$,
within sampling noise. The H5 failure (clamped $\equiv$ unclamped on
the v2 MD split) is a property of the Qwen-1.5B + 20-expert geometry,
not an implementation artefact. The clamp's only earned-keep is
preventing the $\sim 8\%$ catastrophic collapse seen in v1 UnclampedMix.
We do not claim that the norm-ratio clamp is essential or superior in
general, only that on this configuration it is empirically
indistinguishable from weight-cap.

**L4. The compositional ceiling is small.** Oracle routing on the v2
MD split delivers $+0.0206$ similarity over single-adapter routing.
This ceiling is below the v2 pre-registered $+0.03$ bar, which means
that no router upgrade on this base+adapter configuration can clear
the original headline hypothesis. The bottleneck is the underlying
base/adapter geometry, not the router.

**L5. DPO degraded the router as a classifier.** The DPO post-trained
variant scored $42\%$ exact-match on the routing holdout, against
$85\%$ for the SFT-only variant. We do not generalise to "DPO is
harmful for routing"; we report a specific case where pairwise
preference fine-tuning regressed a classification objective. The
10-item TCAR pilot showed a mild downstream bump for DPO, which we
treat as small-sample noise.

**L6. The "Qwen Synapta beats Mistral" headline does not survive
external evaluation.** All three Qwen static variants lose the 30-item
blind rubric comparison by margins of $\sim 4$-$7$ wins to
$\sim 23$-$26$ losses. On the 100-item external MD benchmark, the
TCAR + DPO Synapta system ties Mistral on similarity ($-0.0007$),
loses on F1 ($-0.0205$), and pays a $\sim 2.3\times$ latency penalty.
These three numbers belong together in any quote.

**L7. Internal benchmarks over-rank Qwen methods.** The 9-technique
injection ablation produced an internal vs external ranking
disagreement: internal-similarity rewarded `sequential_reverse`, while
the external benchmark places `weighted_merge` and `late_layer_injection`
on the same rung as `sequential_reverse` and ranks none of them at
parity with Mistral. We retire semantic similarity as a sole
headline metric for this study and treat the internal benchmark as an
ablation/dev set only.

**L8. Single base, single adapter pool, single hardware target.** All
results are on Qwen2.5-1.5B-Instruct (4-bit MLX) with up to twenty
single-domain LoRA experts on Apple Silicon. We do not claim that the
same negative-to-mixed pattern would hold at 7B or 13B base, with
larger LoRA rank, or with a fundamentally different adapter design
(e.g.\ orthogonal experts, mixture-of-experts gating). The
configuration-specific phrase "the bottleneck is geometry, not the
router" applies to *this* base+adapter set.

**L9. Reproducibility on Apple Silicon depends on missing artefacts.**
A subset of the twenty domain LoRA safetensors used in the v1/v2 runs
is not currently restored to a single canonical location. End-to-end
reproduction on a fresh Apple Silicon machine is gated on this
restoration. Aggregate result files (the JSONL/JSON artefacts cited in
the experiment cards) are present and form the basis of every number
in this paper.

**L10. Synthetic routing labels.** The SFT and DPO routers are trained
on $5000$ synthetic routing examples. Routing accuracy on real
externally authored multi-domain queries is not separately evaluated;
the only externally authored routing test is the indirect one of the
TCAR + DPO router's downstream performance on the 100-item external
MD benchmark.

**L11. Statistical regime.** $n{=}40$ on the v2 MD split and $n{=}30$
on the blind judge are small samples by ML benchmarking standards.
We make no claim about statistical significance for the v2 MD result
beyond the direction; the $-0.0003$ norm-ratio vs weight-cap delta is
within sampling noise. The blind-judge margins ($23$-$26$ Mistral wins
out of $30$) are large enough to be informative even at $n{=}30$ but
should be replicated at larger $n$ before serving as a load-bearing
external claim.

## 7. Discussion and Future Work

We started this study expecting that a 1.5B base on Apple Silicon
plus twenty domain experts plus a competent router and clamp would
substitute for a single 7B generalist on multi-domain queries. By the
end of the study we had a different, more honest, picture.

**What composition does on this base.** Bounded $K{=}2$ prompt-level
composition with single-domain LoRA experts does not clear a
pre-registered $+0.03$ similarity bar on this 1.5B base, and the
oracle ceiling itself sits at $+0.0206$. The H5 failure (clamped
$\equiv$ unclamped on MD) and the operational inactivity of the
norm-ratio clamp tell a consistent story: the adapter contributions
are small enough relative to the base activation that they are not
strongly perturbing the base, which means there is little to clamp,
*and* little to compose. The architecture's ceiling is small because
its degrees of freedom are small. Any honest paper on small-base
composition has to reckon with this geometry-first picture.

**What evaluation does on this base.** The most consequential single
finding of this study, in our view, is methodological: internal
similarity benchmarks rewarded methods that the external blind judge
disagreed with. The 30-item rubric reverses the previously favoured
internal narrative. The 9-technique ablation went from "sequential
routing is best" on the internal benchmark to "no Qwen technique
beats Mistral and the rankings shuffle" on the external benchmark.
The lesson is portable beyond the specific Synapta stack: a
small-base composition study should be evaluated on externally
authored prompts and at least partially blind-judged, not on
internally constructed templates against the system's own embedding
metric. We retire semantic similarity as a sole headline metric and
recommend others working on this problem do the same.

**What routers do on this base.** The SFT classifier router is the
only router we trained that meaningfully outperforms generative CoT
routing on multi-label routing accuracy ($85\%$ vs $48.7\%$
exact-match), and even the embedding-centroid router beats CoT
($78.7\%$). DPO, applied as a post-training step, regressed the SFT
router to $42\%$ exact-match. The general question of when DPO helps
or hurts a routing classifier is open; the specific finding here is
that pairwise preference DPO can degrade classification quality even
when it optimises the stated objective. Future work should evaluate
whether classification-aware DPO variants, or a DPO step staged after
a classification head rather than the same generative LM head, can
recover the SFT baseline.

**What TCAR does on this base.** TCAR is the most "publishable" piece
of system design in the study, but only when its tradeoffs are quoted
together. With the DPO router, TCAR matches Mistral-7B on semantic
similarity ($0.6900$ vs $0.6907$) at roughly one-quarter the VRAM,
while losing on token F1 ($-0.0205$) and paying a $\sim 2.3\times$
latency penalty. As a system design lesson it offers a concrete
reformulation of the composition question: instead of mixing in
activation space, run two single-adapter forward passes and let a
third pass refine. This avoids the activation-space clamp question
entirely. It does not avoid the latency cost. TCAR is the right
direction to push if Apple Silicon multi-adapter composition is to
be useful as a deployment story; it is not yet a quality win against
a 7B baseline.

**Future work.** Several directions follow from the limitations:

1. **Larger base or higher-rank adapters.** The clamp inactivity
   suggests that on a 7B+ base, or with rank-128+ adapters, the
   activation-space contribution may be large enough to make the
   norm-ratio clamp non-trivial. The $\Delta = -0.0003$ result is a
   property of *this* configuration, not an architectural verdict.
2. **Externally authored, blindly judged benchmarks.** The 30-item
   rubric should be scaled and the judge pool diversified. A larger
   $n$ external blind judge would strengthen or refute the
   "Mistral beats Synapta" finding at higher confidence.
3. **Routing under classification-aware preference learning.** The
   DPO regression on the routing classifier is a clean negative
   result; whether it can be rescued with a classification-aware
   preference objective is open.
4. **TCAR latency reduction.** TCAR's $2.3\times$ latency penalty is
   the most visible barrier to deployment. Concurrent branch
   execution on UMA hardware, refiner caching, and partial-prefill
   sharing across branches are all candidates.
5. **End-to-end Apple Silicon reproducibility.** Restoration of the
   full twenty-expert safetensor set to a single canonical location
   is a prerequisite for external reproduction of v1/v2 numbers; we
   flag this as a near-term housekeeping item rather than a research
   open problem.

**Closing.** We came to this work hoping to show that prompt-level
multi-adapter composition is a deployment-grade substitute for a
larger generalist on Apple Silicon. The data we collected does not
support that claim. What it does support is a more careful picture
of when and why composition helps: a small-ceiling improvement that
oracle routing barely clears, an inactive clamp on this geometry, a
router whose ceiling exceeds the architecture's compositional
ceiling, and a collaborative-inference loop that approaches but does
not beat a 7B baseline. Reporting this honestly is the contribution.
The negative-results half of the contribution is, we argue, more
useful to the field than yet another internally favoured headline
that an external rubric judge would not have agreed with.

---

### Appendix: number-to-source map

For audit purposes, every quantitative claim in this paper traces to
one of the following primary artefacts (one row per claim cluster):

- v1 aggregate similarity / PPL / latency: `results/decision_summary.md`.
- v2 SD / MD aggregate: `results/v2_decision_summary.md`,
  `results/v2_both_raw.jsonl`.
- Clamp ablation: `results/v2_clamp_ablation_summary.md`,
  `results/v2_md_clamp_ablation.jsonl`.
- Routing gap: `results/v2_routing_gap_summary.md`,
  `results/v2_md_routing_ablation.jsonl`.
- SFT/DPO router: `results/router_accuracy_sft_5000_valid_holdout_mpsfix.json`,
  `results/router_accuracy_dpo_5000_valid_holdout_mpsfix.json`.
- 9-technique ablation: `results/injection_hypotheses_eval_full_20260408.jsonl`.
- External 100-item: `results/md_external_v2_comparison_summary.json`,
  `results/tcar_collaborative_dpo5000_mpsfix_100.jsonl`,
  `results/md_head_to_head_v2_mistral_only_100.jsonl`.
- 30-item blind judge: `results/md_pairwise_*_summary.json`.
- 10-item TCAR pilot: `results/tcar_collaborative_sft5000_mpsfix_pilot10.jsonl`,
  `results/tcar_collaborative_dpo5000_mpsfix_pilot10.jsonl`.
- CoT / embedding routing baselines: `docs/MASTER_KNOWLEDGE_BASE.md`
  (H5/H6 narrative).
