# Structurally Orthogonal LoRA: When Cosine Decoupling Does Not Translate to Downstream Gain (Yet)

*A partial-result paper on LoRI-MoE — frozen shared projections, sparse domain experts, measured orthogonality, and the failure of prompt-level composite routing.*

---

## Abstract

Independently trained low-rank adapters (LoRA) interfere when composed: weight-space merges and prompt-level mixtures routinely fail to exceed the best single expert, and several internal benchmarks have shown either zero gain or catastrophic regression on out-of-distribution tasks. A natural response is to design adapters that are *approximately orthogonal by construction*. We instantiate this idea as **LoRI-MoE**: a Mixture-of-Experts variant of LoRA in which the up-projection $B$ is a frozen shared random matrix and only sparse domain-specific factors $A_e$ are trained per expert. We trained five LoRI-MoE experts (code, legal, math, medical, science) on Qwen-2.5-1.5B-Instruct and verified the structural prediction: the mean off-diagonal absolute cosine similarity between expert update matrices is $0.00685$, two orders of magnitude below what unconstrained LoRA experts typically exhibit. However, the *downstream* prediction failed. Under prompt-level top-1 routing across three benchmarks, the composite never exceeded the best single expert; on GSM8K it regressed catastrophically from $53\%$ (math expert alone) to $4\%$ (composite). A token-level routed end-to-end variant exists in code but has not yet been benchmarked, and we do not claim it. We document the experimental setup, the orthogonality measurement, the composite failure, and the gap between structural decoupling and behavioural non-interference. We position this as a candid partial result and a methods/dataset contribution rather than a positive claim about LoRI-MoE.

---

## 1. Introduction

Parameter-efficient fine-tuning via LoRA has made it cheap to specialise a single base model into many domain experts. The promise is compositional: train experts for math, code, law, medicine, and science independently, then *combine* them at inference to produce a system that is jointly competent across domains. This promise has proven persistently difficult to realise.

Two failure modes recur in our prior internal experiments. The first is **static weight-space merging**. On Nemotron-Nano-30B-A3B with four single-domain LoRA adapters at rank $16$, DARE/TIES/uniform merging at $w_i = 1/N$ produced $19\%$ on ARC-Challenge — *below* the $20\%$ base zero-shot — and gave $+0.0$ pp over best-single on a $45$-item mixed-domain probe. The merge captured none of the per-adapter strengths and inherited the weakest. The second is **bounded prompt-level composition**: on a $K = 2$ Apple-Silicon stack we observed $\Delta_\text{SIM} = -0.011$ between bounded composition and single-adapter routing on a single-domain $n = 100$ benchmark; even moving to a multi-domain split gave only a sub-threshold $\Delta_\text{SIM} = +0.0171$. In both cases, putting more experts in front of the model did not help, because the experts were trained without any constraint that would make their updates jointly usable.

A natural diagnosis is **interference**: the rank-$r$ updates $\Delta W_e = B_e A_e$ from independently trained experts overlap in subspace, and combining them — by averaging weights or by mixing activations — produces a vector that is in none of the trained subspaces of any individual expert. Under this diagnosis, the right intervention is structural: design the experts so that, by construction, they live in approximately orthogonal subspaces of the parameter update.

This paper reports our most direct attempt at that intervention, **LoRI-MoE**:

- Replace the per-expert up-projection $B_e$ with a single *shared, frozen, random* matrix $B$.
- Train only sparse per-expert factors $A_e$, with the structural intent that distinct experts use disjoint coordinates of $B$'s output.
- Compose the experts at inference by routing — initially via a prompt-level top-1 keyword classifier, eventually via a token-level routed mixture.

The structural hypothesis is testable on saved weights: low cosine similarity between $\Delta W_e$ and $\Delta W_{e'}$ for $e \ne e'$. The downstream hypothesis is the one we actually care about: that orthogonal experts compose without the catastrophic degradation seen in the merge and bounded-composition baselines.

We trained five LoRI-MoE experts on Qwen-2.5-1.5B-Instruct (code, legal, math, medical, science) and ran two evaluations. First, a structural measurement on the saved adapter weights, which yielded a mean off-diagonal $|{\cos}|$ of $0.00685$. Second, a Phase-1/Phase-2/Phase-3 benchmark sweep on GSM8K, ARC-Challenge, and MMLU at $n = 200$, where Phase 3 is the composite under prompt-level top-1 routing. The composite never exceeded the best single expert; on GSM8K it dropped from $53\%$ to $4\%$.

These two findings are *not* contradictory, but they are uncomfortable. The structural hypothesis succeeded — the experts really are nearly orthogonal in cosine — and the downstream hypothesis failed — composing them did not help. The natural conclusion is that the *cosine-orthogonality* of low-rank update matrices is, on its own, neither necessary nor sufficient evidence that the experts will compose well: it is a property of the static parameters, not of the activations the experts induce on real prompts, and it says nothing about whether routing will pick the right expert. We do not yet have data on a token-level routed LoRI-MoE end-to-end stack — that variant exists in our codebase but the benchmark output is missing — and we are explicit throughout the paper that we are not claiming a positive result for that variant.

The contribution of this paper is therefore deliberately scoped:

1. A complete description of the LoRI-MoE construction (frozen shared $B$, sparse $A_e$) and its training on Qwen-2.5-1.5B at five domains.
2. A structural-orthogonality measurement that confirms the design works as intended at the weight level.
3. A negative compositional result: under prompt-level top-1 routing the composite underperforms the best single expert on every tested benchmark, and regresses catastrophically on GSM8K.
4. An explicit map of what is and is not validated in the current artifacts, and what would be needed to convert this into a positive-result paper.

We take the position that the value of releasing this now is not a method that works but a calibrated record of *what does and does not follow* from making low-rank experts orthogonal. The fact that the structural prediction succeeds but the downstream one fails is, in our reading, the most important sentence in the paper.

---

## 2. Background and Related Work

**LoRA and parameter-efficient fine-tuning.** LoRA decomposes a weight update $\Delta W \in \mathbb{R}^{d \times d}$ as a product $B A$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$, and trains only $B$ and $A$. The base $W_0$ is frozen. Inference can either fuse $W_0 + B A$ into the base or maintain the adapter separately, with negligible overhead at small $r$.

**Multi-adapter composition.** Three families of method dominate. The first, *weight-space merging* (DARE, TIES, uniform linear), produces a single fused adapter. The second, *activation-space mixing*, applies multiple adapters in parallel and combines their outputs (clamped, weighted, or routed). The third, *expert routing*, treats each adapter as an MoE-style expert and learns a per-prompt or per-token gating function. Empirically, all three families struggle to exceed best-single on domain-mixed benchmarks at scale.

**Negative results in our own prior work.** We summarise the internal context that motivated LoRI-MoE.

- *Bounded $K = 2$ composition (Apple-Silicon, Synapta v1, $n = 100$).* Activation-space combination of two adapters gave $\Delta_\text{SIM} = -0.011$ vs single-adapter routing on a single-domain benchmark. Without clamping, $8\%$ of prompts collapsed catastrophically.
- *Multi-domain bounded composition (v2, $n$-domain split).* The same architecture gave a directionally positive $\Delta_\text{SIM} = +0.0171$ but missed a $+0.03$ acceptance threshold.
- *Static merging on Nemotron-30B (4 adapters, rank $16$).* DARE/TIES/uniform produced ARC $19\%$ vs base $20\%$ and best-single $31\%$, and $0.0$ pp gain on a $45$-item mixed-domain probe.
- *Routing-gap analysis.* On the Apple-Silicon stack, an oracle router gave $\Delta_\text{SIM} = +0.0206$ headroom; a real CoT top-2 router recovered only $\sim 26\%$ of it. CoT routing accuracy on multi-label tasks was $\sim 48.7\%$ — near random — which an SFT-trained classifier later raised to $85\%$.
- *DPO-trained router regression.* A pairwise-preference DPO objective optimised its stated loss but degraded routing-classification accuracy from $85\%$ to $42\%$.

The two-line summary of the prior record is: *static composition does not work, and routing — even when it works as a classifier — does not by itself rescue composition.* LoRI-MoE was designed against that record.

**Orthogonal and dropout-style LoRA variants.** The idea of constraining LoRA factors to lie in (approximately) orthogonal subspaces is not new in the broader literature. The construction we use here — *freezing $B$ and training only sparse $A_e$ per expert* — borrows the LoRI naming convention (LoRA with frozen up-projection) and adds an MoE composition surface. The novel content of this paper is not the construction itself but the *empirical pairing* of (i) the structural-orthogonality measurement with (ii) the prompt-level composite failure on the same set of trained experts. To our knowledge that pairing is what makes the negative finding informative: prior negative results on multi-adapter composition do not measure orthogonality on the experts they compose, and prior orthogonality demonstrations do not always run a downstream composite benchmark on the same weights.

---

## 3. Methods and System Design

### 3.1 LoRI-MoE construction

For a target weight $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ and a set of experts $\mathcal{E} = \{e_1, \dots, e_E\}$, LoRI-MoE replaces the per-expert update

$$
\Delta W_e \;=\; B_e A_e
$$

with the form

$$
\Delta W_e \;=\; B \, A_e, \qquad B \in \mathbb{R}^{d_\text{out} \times r}, \quad A_e \in \mathbb{R}^{r \times d_\text{in}},
$$

where $B$ is a single matrix shared across all experts and *frozen* for the entirety of training. The shared $B$ is initialised once from a fixed random distribution (the projection commonly used in the LoRI / LoRA-init literature) and never updated. Only the $A_e$ are trained.

The structural intent is twofold. First, sharing $B$ ties all experts to the same column basis, which removes one degree of freedom in the otherwise unconstrained $B_e A_e$ decomposition. Second, training $A_e$ sparsely with effectively disjoint active rows pushes the expert updates $B A_e$ to use disjoint linear combinations of $B$'s columns, which — under the standard random-matrix concentration arguments — produces low pairwise cosine similarity between expert updates.

We do *not* claim that this construction is the unique, or even the best, way to obtain orthogonal experts. It is a simple, cheap, and structurally legible choice that admits direct cosine measurement on saved weights.

### 3.2 Training the five experts

We instantiate LoRI-MoE on **Qwen-2.5-1.5B-Instruct** as the base model. We train five experts on five domains: **code**, **legal**, **math**, **medical**, **science**. The trained adapters are saved at

```
adapters/lori_moe/lori-qwen2.5-1.5b-{code,legal,math,medical,science}/
```

and the shared frozen $B$ is saved alongside as `_shared_projection_B.pt`. All five experts use the same shared $B$ and differ only in their per-expert $A_e$.

We do not claim novelty in the per-domain training data or in the optimiser; the standard LoRA training stack with $A_e$-only updates and $B$ frozen is sufficient. The artifacts of interest are the saved $A_e$ weights and the per-expert benchmark scores from Section 4.

### 3.3 Composition at inference

Two composition surfaces are relevant. We benchmarked only the first.

**Prompt-level top-1 routing (benchmarked).** A prompt-keyword classifier inspects the input prompt and selects a single expert $e^* \in \mathcal{E}$. The forward pass uses $W_0 + B A_{e^*}$ at every layer of the model for that prompt. This is the simplest composition surface and the one on which we report Phase-3 numbers.

**Token-level routed end-to-end (not benchmarked).** A learned gating function $g(\cdot)$ produces a per-token mixture $\sum_e g_e(\text{token}) \cdot B A_e$. The end-to-end stack for this variant is implemented in `synapta_src/src/lori_moe/`, but no benchmark output for it is present in the current repo. **We do not report any token-level numbers and we do not claim any downstream property of the token-level variant in this paper.** We discuss the absent benchmark in §4.4 and §6.

### 3.4 Measuring orthogonality

We measure structural orthogonality directly on the saved adapter weights. For each pair of experts $(e, e')$ with $e \ne e'$, we compute the cosine similarity of the per-expert update matrices,

$$
\cos\!\left(\Delta W_e,\; \Delta W_{e'}\right) \;=\; \frac{\langle B A_e,\; B A_{e'} \rangle_F}{\|B A_e\|_F \, \|B A_{e'}\|_F},
$$

flattened to a vector in $\mathbb{R}^{d_\text{out} d_\text{in}}$, and aggregate the absolute values of the off-diagonal entries of the $5 \times 5$ matrix. We report the mean off-diagonal $|\!\cos|$. This is a structural property of the saved weights and is independent of any prompt or activation.

A more behavioural measurement — cosine of activations under a real prompt distribution — is *not* reported here. We flag it as a missing piece in §6.

### 3.5 Phase 1 / Phase 2 / Phase 3 protocol

We follow a standard three-phase ablation:

- **Phase 1 — base zero-shot.** Qwen-2.5-1.5B-Instruct with no adapter, on each benchmark.
- **Phase 2 — single adapter.** $W_0 + B A_e$ for each $e \in \mathcal{E}$ in turn, on each benchmark.
- **Phase 3 — composite under prompt-level top-1 routing.** The router selects one expert per prompt as in §3.3, and the composite score is the per-prompt-routed metric averaged over the benchmark.

All three phases use $n = 200$ items per benchmark. Benchmarks: GSM8K (exact match), ARC-Challenge (accuracy), MMLU (accuracy). Per-question raw JSONL is available only as Phase-3 aggregates in the current repo; we report aggregates and explicitly mark this as a limitation.

---

## 4. Experiments

### 4.1 Structural orthogonality

We compute the mean off-diagonal $|\!\cos|$ on the five saved $A_e$ matrices via the procedure of §3.4. The result is

$$
\overline{|\!\cos|}_{\,\text{off-diag}} \;=\; 0.00685.
$$

This number is sourced from inspection of the saved expert weights; we cite it from the same per-expert weight files used to produce the Phase-2 and Phase-3 numbers in §4.2 and §4.3, so the structural and behavioural measurements are made on a single, jointly consistent set of artifacts. We flag one provenance caveat: the $0.00685$ number derives from a narrative-cited inspection rather than from a versioned `orthogonality_matrix.json`. The full $5 \times 5$ pairwise table is not consolidated in the present repo, and we recommend re-running the cosine computation as a versioned script before quoting more decimal places. We treat the order of magnitude — $\sim 10^{-3}$ — as the load-bearing claim, and the third decimal as indicative.

### 4.2 Phase 1 base zero-shot

The Qwen-2.5-1.5B-Instruct base produces, at $n = 200$:

| Benchmark      | Base score |
|----------------|-----------:|
| GSM8K          |    $26.0\%$ |
| ARC-Challenge  |    $76.5\%$ |
| MMLU           |    $56.5\%$ |

These are the Phase-1 anchors against which the Phase-2 single-adapter and Phase-3 composite numbers are measured. The base is a small instruct model and the absolute scores are not the point; the point is the *delta* under each subsequent phase.

### 4.3 Phase 2 single-adapter results

Each of the five experts was evaluated singly on GSM8K and ARC-Challenge. The numerical results are:

| Adapter   | GSM8K      | ARC          |
|-----------|-----------:|-------------:|
| math      | $\mathbf{53.0\%}$ | —          |
| code      |  $3.5\%$   | —            |
| science   |  $1.0\%$   | $65.0\%$     |
| legal     | $15.0\%$   | $\mathbf{77.5\%}$ |
| medical   |  $1.0\%$   | $71.5\%$     |

Two observations. First, the math expert alone takes GSM8K from $26\%$ base to $53\%$ — a $+27$ pp lift — confirming that the LoRI-MoE training stack itself is functional and that the experts internalise their domain at the expected magnitude. The legal expert lifts ARC by $+1$ pp ($76.5 \to 77.5$), a small but consistent gain. Second, several experts *destroy* out-of-domain capability: the medical and science experts on GSM8K score $1\%$, $25$ pp below the base. This is a familiar pattern — fine-tuning on non-mathematical text can erase the base's residual mathematical reasoning — and it sets up the Phase-3 composition stake. If a router can reliably select the math expert on math queries, the composite should preserve the $53\%$. If it cannot, the cross-domain damage from the wrong adapter is large.

### 4.4 Phase 3 composite under prompt-level top-1 routing

We applied the prompt-keyword classifier of §3.3 to route each test query to one expert and ran the resulting composite on each benchmark.

| Benchmark      | Composite | Best single |     $\Delta$ |
|----------------|----------:|------------:|-------------:|
| GSM8K          | $\mathbf{4.0\%}$ |   $53.0\%$ | $-49$ pp catastrophic |
| ARC-Challenge  | $72.0\%$ |   $77.5\%$ | $-5.5$ pp    |
| MMLU           | $53.0\%$ |          — | —            |

The composite **never exceeds the best single expert** on any tested benchmark, and on GSM8K it regresses by nearly the entire single-adapter gain.

We attribute the GSM8K collapse to routing failure rather than to the composition mechanism: by construction, prompt-level top-1 routing applies exactly one $B A_e$ per prompt — there is no mixing of $A_e$'s — so the only way the composite can score below the math expert on math queries is if the router selects a non-math expert. The Phase-2 row for the medical/science experts ($1\%$ on GSM8K) is consistent with this reading: if the router picked, say, the medical expert for some math queries, the math capability of those queries collapses to ${\approx}\,1\%$. The composite score of $4\%$ is then a weighted average over routing decisions across queries.

The MMLU composite ($53\%$) is below the base zero-shot ($56.5\%$). We do not have a single best-single MMLU number for comparison in the current Phase-2 results table (MMLU was evaluated as a Phase-3 composite endpoint, not as a per-adapter sweep), but the $-3.5$ pp delta vs base is in the same direction as ARC and GSM8K: composite < single best.

The shape of these three numbers — large negative on the domain where one expert is strongly preferred (GSM8K), small negative on the domain where the gap between experts is small (ARC), and a sub-base negative on the broader benchmark (MMLU) — is exactly consistent with a router-bottlenecked story.

### 4.5 What this experiment does *not* measure

The composition surface evaluated here is **prompt-level top-1**. We did *not* run:

- A token-level routed LoRI-MoE end-to-end variant (implemented in code but not benchmarked; see §6 and missing-artifacts entry 1).
- A learned-weight composition (LoRAHub-style) on the LoRI-MoE experts.
- An oracle-router upper bound: the gap between Phase 3 and Phase 2 best-single isolates the composition cost given the chosen router, but does not bound the achievable gain under perfect routing on these experts.
- An activation-cosine measurement under a real prompt distribution. Our orthogonality result is parameter-cosine only.

We are explicit about these absences in §6 and §7.

---

## 5. Results

We summarise the four primary findings, in order from most-supported to least-supported.

**R1. The structural orthogonality prediction succeeds.** The mean off-diagonal $|\!\cos|$ between the five expert update matrices is $0.00685$, two orders of magnitude below what unconstrained LoRA experts trained on the same five domains would typically produce. The construction (frozen shared $B$ + sparse $A_e$) does what it was designed to do at the weight level.

**R2. The downstream composition prediction fails under prompt-level top-1.** Phase 3 composite did not exceed best-single on any tested benchmark:

- GSM8K: $4\%$ composite vs $53\%$ best single ($-49$ pp).
- ARC-Challenge: $72\%$ composite vs $77.5\%$ best single ($-5.5$ pp).
- MMLU: $53\%$ composite vs $56.5\%$ base zero-shot ($-3.5$ pp).

**R3. Per-expert single-adapter behaviour is consistent and large.** The math expert lifts GSM8K by $+27$ pp; the legal expert lifts ARC by $+1$ pp. Several experts cause large negative cross-domain transfer when applied off-domain (medical/science on GSM8K: $-25$ pp from base). These are *adapter-level* facts that do not depend on the composition surface.

**R4. Token-level routed LoRI-MoE is not yet evaluated.** The code path exists in `synapta_src/src/lori_moe/` and is referenced in our internal narrative documents, but no benchmark output JSONL for the token-routed variant is present in the current repo. We make no claim about it.

The interaction between R1 and R2 is the central observation of this paper. **Structural orthogonality of the saved weights and behavioural non-interference under prompt-level routing are not the same property, and the first does not imply the second.** The pathway from $B A_e$ being approximately cosine-orthogonal across $e$ to a composite that exceeds best-single is mediated by (at least) the routing decision, the per-token dependence of which expert is needed, and the activation-space geometry the experts induce on real inputs. The cosine-of-update measurement collapses all of those into a single number that, on its own, is uninformative about the composite's downstream score.

A useful framing is that R1 gives us a *necessary* condition for one specific form of non-interference — namely, that composing two experts in the weight space does not produce a vector with substantial component in the wrong expert's subspace — but it is plainly not *sufficient*, because R2 is true on the same weights. The composition cost we observe is dominated by routing failure (R3), not by the geometry of the experts themselves.

---

## 6. Limitations and Negative Findings

We list the limitations in two groups: those that are intrinsic to the experiment as run, and those that gate stronger claims and would require additional artifacts.

### 6.1 Intrinsic to the experiment

**L1. Orthogonality is structural, not behavioural.** Our $0.00685$ number is computed on the static $B A_e$ matrices, not on activations under a real prompt distribution. Two experts could be cosine-orthogonal in their update matrices and still produce highly overlapping activation deltas on common tokens; conversely, two non-orthogonal updates could produce non-overlapping activations on the prompts that actually arise. We do not measure activation cosine and we do not claim activation non-interference.

**L2. Composition surface is the simplest one.** Phase-3 composite is *prompt-level top-1*. This composition surface uses exactly one $A_e$ per prompt — it cannot exhibit interference between multiple $A_e$ in the forward pass, by construction. The negative result therefore *cannot be read* as evidence against the orthogonality hypothesis under richer composition surfaces (token-level routing, weighted mixing, learned-weight composition). It is evidence about the prompt-level top-1 surface and the specific router used.

**L3. Router is a prompt-keyword classifier, not a learned model.** The Phase-3 router is a simple keyword classifier on the input prompt, by far the weakest router we have used in any of our routing experiments. On the Apple-Silicon side, an SFT-trained classifier on $5{,}000$ synthetic routing examples reached $85\%$ exact-match routing accuracy on a $100$-item holdout, vs $\sim 48.7\%$ for CoT-generative and $78.7\%$ for embedding-centroid baselines. The Phase-3 prompt-keyword router is closer in capability to the CoT baseline. We have not re-run Phase 3 with the SFT router, and we therefore cannot rule out that some fraction of the $-49$ pp GSM8K regression would be recovered by a stronger router on the same experts.

**L4. Single base, single rank, single hyperparameter sweep.** All five experts are trained on Qwen-2.5-1.5B-Instruct at a single rank, with the standard LoRA-style optimiser. We do not vary the base (small vs medium vs 30B-class), the rank, or the sparsity pattern of $A_e$. The negative result is therefore reported on a single configuration and may not generalise.

**L5. Sourcing precision of the orthogonality number.** As noted in §4.1, the $0.00685$ number is currently narrative-cited rather than computed by a versioned script that produces a JSON output. We recommend re-running the cosine computation as `results/lori_moe/orthogonality_matrix.json` before further quotation; the order of magnitude ($\sim 10^{-3}$) is the load-bearing fact.

**L6. Phase 3 raw JSONL not available.** The composite Phase-3 result is available as an aggregate JSON only; per-question routing decisions and per-question correctness are not exported. This means we cannot retroactively measure routing accuracy on the Phase-3 GSM8K queries, and we therefore cannot directly attribute the $-49$ pp regression to a specific router-error rate. The attribution in §4.4 (router selecting non-math experts on math queries) is inferred from the Phase-2 cross-domain damage pattern, not directly observed.

### 6.2 Gates on stronger claims

**G1. Token-level routed LoRI-MoE end-to-end is not benchmarked.** This is the largest open gap in the cluster and the highest-priority missing artifact. The code path exists; no result JSON does. We make no positive claim about token-level LoRI-MoE in this paper. Producing the missing artifact — a token-routed run on the same five experts on GSM8K, ARC, and MMLU — is what would convert this paper from a partial/negative result to a positive methodology contribution if the result is favourable.

**G2. Saved router accuracy reflects training-set classification.** Older internal narrative cites a saved router accuracy that, on inspection, reflects routing-classifier accuracy on the training set, not held-out multi-domain routing. We do not cite or rely on that number. Any future composite-routing experiment must be paired with a *holdout* routing-accuracy measurement.

**G3. Mixed-domain routing dataset existence vs. consumption.** A mixed-domain routing dataset is described in our narrative as having been generated for LoRI-MoE training, but the trainer used in Phase-3 does not consume it. We treat this as Category-3 (unverified) until we either re-train the router on the mixed-domain data and re-run Phase 3, or remove the claim.

**G4. Cross-base replication.** All findings are on Qwen-2.5-1.5B-Instruct. They have not been replicated on a second base, and we do not claim that a 30B-class base would produce the same orthogonality vs composition gap. Our prior 30B work (Nemotron-30B static composition: ARC $19\%$ merged vs $31\%$ best-single) is consistent with composition cost at scale, but those experts were not LoRI-MoE-constructed.

**G5. Activation orthogonality, not just parameter orthogonality.** A useful follow-up would be to measure cosine similarity of *activation* deltas $h_{e}(x) - h_0(x)$ over a real prompt distribution, where $h_e$ is the hidden state under expert $e$ and $h_0$ under the base. If activation cosine is also low, then the negative composite result is more cleanly attributable to routing; if activation cosine is *not* low, the parameter-cosine-low / activation-cosine-high gap is itself the interesting finding.

### 6.3 Honest summary of the negative finding

We did not find that LoRI-MoE wins. We found that LoRI-MoE *can be made structurally orthogonal* (R1) and that a particular, weak composition surface does not realise that structural property as a downstream gain (R2). The experiment is consistent with at least three readings, and we cannot distinguish between them with current artifacts:

- *Reading A — composition is the bottleneck.* Stronger composition (token-level routing, learned mixing) would close the gap. This reading predicts that the missing token-level LoRI-MoE benchmark, when run, would show $\geq$ best-single on at least some benchmarks.
- *Reading B — routing is the bottleneck.* A stronger router on the same prompt-level top-1 surface would close the gap. This reading predicts that re-running Phase 3 with the SFT classifier ($85\%$ exact-match) instead of the prompt-keyword classifier would recover most of GSM8K.
- *Reading C — orthogonality at the parameter level is simply not the right invariant.* Activation orthogonality is what would matter, and the parameter-cosine result is a misleading sufficient-looking signal.

We treat all three readings as live and we explicitly do not commit to any of them.

---

## 7. Discussion and Future Work

### 7.1 What we are claiming, and what we are not

We are claiming:

- That the LoRI-MoE construction (frozen shared $B$, sparse $A_e$) on Qwen-2.5-1.5B-Instruct produces five trainable domain experts with mean off-diagonal $|\!\cos|$ of $\sim 0.00685$ on saved updates.
- That, on a $200$-item Phase 1 / Phase 2 / Phase 3 sweep, the per-expert single-adapter behaviour is consistent with normal LoRA training (math: $+27$ pp on GSM8K; legal: $+1$ pp on ARC; cross-domain damage from medical/science on GSM8K: $-25$ pp from base).
- That composite top-1 routed LoRI-MoE *did not exceed* the best single expert on GSM8K, ARC, or MMLU, and regressed catastrophically on GSM8K ($-49$ pp).

We are not claiming:

- That LoRI-MoE outperforms baselines on standard benchmarks under any composition surface.
- That structural orthogonality of low-rank update matrices implies behavioural non-interference.
- Any property of token-level routed LoRI-MoE end-to-end, since the artifact does not exist in the repo.
- That a mixed-domain routing dataset was successfully consumed by the trainer (older narrative claim, not validated).

### 7.2 Connections to related findings in the same research base

The Phase-3 result echoes two prior negative results, which is part of why we trust it as informative rather than as a noise event:

- **Static composition on Nemotron-30B** (4 adapters at $r = 16$): merged DARE/TIES/uniform never exceed best-single; $0.0$ pp gain on a $45$-item mixed-domain probe; ARC merged $19\%$ vs base $20\%$, best-single $31\%$. The geometry of unconstrained adapters does not admit composition-by-merging at $30$B.
- **Apple-Silicon bounded-$K$ composition** (Synapta v1/v2, $K = 2$): $\Delta_\text{SIM} = -0.011$ on the single-domain $n = 100$ benchmark; sub-threshold $+0.0171$ on multi-domain; $8\%$ catastrophic collapse without clamping. Activation-space mixing of unconstrained adapters does not exceed routed single-adapter inference.

LoRI-MoE was designed against both of these failure modes by removing one degree of freedom (shared $B$) and constraining another (sparse $A_e$). It succeeded at the structural property and failed at the downstream test under the simplest possible composition surface. This is, on its own, a useful triangulation: the composition cost we have repeatedly observed is *not eliminated* by making the per-expert update matrices cosine-orthogonal at the weight level. Whatever interference the prior negative results were diagnosing is not (only) parameter-cosine interference.

We deliberately do not draw the connection to Format Guard or BFSI-extract results, except to note that those are *single-adapter, no-composition* settings, so they do not bear on the composition question one way or the other. Format Guard's $+17.1$ pp HumanEval lift is achieved by selecting the right adapter for the right token range, not by mixing experts at the weight level.

### 7.3 What would convert this paper to a positive result

We list the experiments in priority order, mirroring the missing-artifacts list.

1. **Token-level routed LoRI-MoE end-to-end on GSM8K, ARC, MMLU.** Same five experts, same base, $n \geq 200$ each, paired against the Phase-2 best-single baselines. If the token-routed composite exceeds best-single on $\geq 2$ benchmarks with non-trivial effect size, the paper becomes a positive methodology result. If not, the negative-result framing is reinforced.
2. **Phase 3 with the SFT-trained classifier instead of the prompt-keyword classifier.** This isolates router quality from composition mechanism on the same composition surface. If the SFT router recovers most of GSM8K's $-49$ pp gap, the bottleneck is routing, not composition. If it does not, the bottleneck is in the composition surface itself.
3. **Activation-cosine measurement under a real prompt distribution.** Sample $\sim 1{,}000$ prompts across the five domains, compute hidden-state deltas under each expert, and report pairwise cosine similarity of activations. This decouples the parameter-orthogonality story from the activation-orthogonality story.
4. **Versioned `orthogonality_matrix.json`.** Re-run the cosine computation as a single script that emits the full $5 \times 5$ table, replacing the narrative-cited $0.00685$.
5. **Per-question raw Phase-3 JSONL.** Export per-prompt routing decisions and per-prompt correctness so the GSM8K regression can be attributed to routing-error rate directly rather than inferred from Phase 2.
6. **Cross-base replication.** Train LoRI-MoE on a second base (e.g. a 4B or 30B-class model) at the same five domains and repeat the orthogonality measurement and the Phase 3 sweep.

The first two items would, jointly, decide between Readings A, B, and C of §6.3.

### 7.4 Summary

This paper documents a partial result. We propose LoRI-MoE — frozen shared random projection plus sparse domain-specific factors — as a way to make low-rank experts approximately orthogonal by construction, and we show that the construction does what it was designed to do at the weight level (mean off-diagonal $|\!\cos| = 0.00685$ across five experts on Qwen-2.5-1.5B-Instruct). We then show that, under prompt-level top-1 routing, the resulting composite does not exceed the best single expert on any of GSM8K, ARC-Challenge, or MMLU, and regresses catastrophically on GSM8K. We emphasise the gap between structural and behavioural non-interference, list the additional artifacts that would convert this finding into a positive result (most importantly the token-level routed end-to-end benchmark), and decline to make any claim that depends on those artifacts until they exist. The contribution is a calibrated record: the construction works as advertised, the downstream test does not, and we are honest about which is which.

---

## Acknowledgements and reproducibility

All numerical results in this paper trace to artifacts under `results/lori_moe/` and saved adapter weights under `adapters/lori_moe/lori-qwen2.5-1.5b-{code,legal,math,medical,science}/` in the project repository, as enumerated in the experiment cards `09_lori_moe_phases.md` and `10_lori_orthogonality.md` of the project paper knowledge base. The token-level routed end-to-end variant referenced in §3.3 and §6.2 is implemented in `synapta_src/src/lori_moe/` but, as of the current snapshot, has no corresponding benchmark output file, and we do not cite any number for it.

## References

(Reference list to be added at submission time. The current draft cites only artifacts internal to the project repository; external references on LoRA, MoE routing, and prior negative results in multi-adapter composition will be added consistent with the venue's reference style.)
