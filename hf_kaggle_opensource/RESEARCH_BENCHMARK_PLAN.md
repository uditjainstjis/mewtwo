# DPO Adapter Research Plan

## Objective

Turn the current Qwen 0.8B, Nemotron 4B, and Nemotron 30B adapter inventory into a paper-grade study on:

1. How rank changes adapter capability and transfer.
2. How DPO changes a merged multi-domain adapter relative to its pre-DPO state.
3. Whether adapter geometry predicts downstream interference, mergeability, and cross-task generalization.
4. Whether the same knowledge encoded at multiple model scales produces aligned or divergent adapter representations.

## Main Paper Narrative

The strongest version of this project is not:

- "LoRA rank scaling works"
- "DPO improves adapters"
- "DARE plus DPO is a neat engineering pipeline"

The strongest version is:

- low-rank adapter geometry predicts mergeability, agent reliability, and failure modes
- DPO changes policy behavior more than it changes raw competence
- the same knowledge at 0.8B, 4B, and 30B does not map to identical adapter subspaces or identical agent policies

Working title direction:

- `Subspace Geometry of Low-Rank Adapters Predicts Alignment and Failure in Agentic LLMs`

## Claim Discipline

Use these rules in the paper and in internal notes:

- Do not headline the paper with generic rank-scaling claims. That is supporting evidence, not the main novelty.
- Do not claim `DARE + DPO` is novel by itself. The novelty is in the geometry and behavior analysis that this pipeline enables.
- Treat Perplexity or blog-discovered citations as scouting hints until they are verified against primary sources.
- Separate `evidence` from `inference` whenever we discuss novelty or mechanistic interpretation.

## Experimental Guardrails

- `qwen_0.8b` with `rank3072` should be treated as an overparameterized or near-full-rank control, not as a standard low-rank point.
- Main LoRA-style claims should rely primarily on `r in {1, 2, 8, 128}` and then use `1024, 3072` as boundary-condition ablations.
- Agent claims must not rely on single-run scores. Repeated trials and reliability-style metrics are mandatory.
- Geometry claims should be framed as predictive correlations unless we run explicit causal interventions.

## Core Comparator Matrix

Primary axis:

- Models: `qwen_0.8b`, `nemotron_4b`, `nemotron_30b`
- Stages: base, single-domain SFT, merged DARE, final DPO
- Ranks: `1, 2, 8, 128, 1024, 3072` where available

Immediate overnight matrix:

- `qwen_0.8b_math_DPO_rank{1,2,8,128,1024,3072}`
- `nemotron_4b_math_DPO_rank{128,1024,3072}` plus already completed `1,2,8`
- Matched pre-DPO comparators: `*_merged_DARE_rank*`

Secondary matrix for the paper:

- Single-domain SFT adapters by domain and rank
- Nemotron 30B adapters trained on the same knowledge
- Merged vs unmerged vs DPO-refined variants

## Benchmark Tiers

### Tier 1: Overnight, fully automatable

These are already wired or straightforward in this repo and should run after DPO completion:

- `GSM8K`: math transfer and answer-format reliability
- `MATH-500`: harder symbolic reasoning
- `ARC-Challenge`: general science/knowledge reasoning
- `MMLU`: broad knowledge retention and cross-domain transfer
- `HumanEval`: executable code synthesis
- `MBPP`: short-form practical coding

Why this tier matters:

- It gives a balanced math/code/general-knowledge profile.
- It exposes whether DPO sharpens one capability while damaging another.
- It is enough to estimate rank-performance curves quickly.

### Tier 2: Paper-grade static benchmarks

These should be added once the environment is confirmed and the overnight Tier 1 run is stable:

- `MMLU-Pro`: harder multi-domain knowledge than vanilla MMLU
- `GPQA`: expert-level science reasoning
- `IFEval`: instruction-following precision
- `LiveCodeBench`: contamination-resistant code evaluation
- `TruthfulQA`: factual robustness and hallucination pressure

Why this tier matters:

- It upgrades the paper from "good internal study" to "competitive external benchmark study."
- It addresses the usual reviewer criticism that MMLU/GSM8K/MBPP are too narrow or too saturated.

### Tier 3: Core agentic benchmark track

This should be treated as a separate axis of the paper, not a minor appendix. The central question is:

- Do DPO and rank change only answer quality, or do they also change how the model behaves as an agent under long horizons, tool constraints, and recovery pressure?

Recommended core stack:

- `PlanBench`: controlled planning and reasoning-about-change tasks
- `tau-bench`: tool-agent-user interaction with policy/rule constraints and pass^k reliability
- `ToolSandbox` or `BFCL`: structured tool-calling validity, stateful tool use, and invalid-call analysis
- `WebArena-Verified` subset: realistic browser workflows after the lighter agentic suite is stable

Optional later-stage additions:

- `GAIA` subset: broader assistant behavior and knowledge-to-action transfer
- `DeepPlanning`: harder long-horizon constrained planning if we want a newer planning stress test
- `OSWorld` slice: GUI and desktop recovery behavior if infra is justified
- `SWE-bench` slice: only if we deliberately want a coding-agent subsection
- `WorkArena`: only if we want an enterprise-workflow angle

Why this tier matters:

- Existing knowledge benchmarks tell us what the adapter knows.
- Agentic benchmarks tell us whether the adapter can sequence, recover, obey policies, and remain stable over trajectories.
- This is exactly where small representational changes can create large behavioral differences, which is much more novel than another static QA table.

### Tier 4: Reliability and robustness overlays

These are not separate leaderboards. They are overlays on the core agentic benchmarks above.

- repeated-run reliability: `pass@k`, trajectory variance, answer variance
- scaffold sensitivity: direct vs ReAct vs plan-execute vs reflection
- tool temptation: unnecessary optional tools
- policy pressure: explicit constraints and forbidden actions
- injected-failure recovery: corrupted tool outputs, missing observations, delayed steps
- wrong-adapter routing: forced mismatch followed by possible recovery

### Tier 5: Novel representation and adapter-science analyses

- Subspace similarity between adapters using cosine, CKA, and principal-angle overlap
- Rank-efficiency curves: score gain per trainable parameter
- DPO delta analysis: `DPO - merged_DARE` by benchmark, model, and rank
- Cross-scale alignment: does `nemotron_30b` learn the same adapter directions as 0.8B and 4B?
- Mergeability vs geometry: do more orthogonal adapters compose better?
- Interference prediction: can similarity metrics predict benchmark degradation after merge?
- Specialization leakage: does a math DPO adapter improve code or science unexpectedly?
- Saturation thresholds: does rank 1024 or 3072 stop buying usable gains?
- DPO rotation magnitude: how far the adapter subspace moves from `merged_DARE` after preference tuning
- Geometry-to-behavior modeling: can geometry features predict policy failures, recovery failures, or scaffold sensitivity

## Main Hypotheses

### H1: Rank has diminishing returns, but the saturation point is model-size dependent

Expected result:

- Qwen 0.8B should saturate earlier than Nemotron 4B and 30B.
- Very high ranks should show weak marginal gains or even instability for small models.

### H2: DPO improves answer selection and benchmark reliability more than raw knowledge breadth

Expected result:

- Stronger gains on exact-match and instruction-sensitive tasks than on broad knowledge tasks.
- Largest gains on medium ranks where the adapter has enough capacity but is not bloated.
- On agentic tasks, DPO should move policy-compliance and consistency metrics more than pure planning or search-depth metrics.

### H3: Adapter geometry predicts mergeability

Expected result:

- Low overlap or low principal-angle similarity should correlate with lower interference after merge.
- DPO may rotate the adapter into a more benchmark-friendly but less merge-friendly subspace.

### H4: Cross-scale knowledge is not simply a rescaled copy

Expected result:

- The same domain data on 0.8B, 4B, and 30B will not produce identical adapter directions.
- Larger models may encode cleaner and more transferable low-rank task subspaces.

### H5: DPO changes agent policy reliability more than it changes raw planning competence

Expected result:

- DPO should improve constraint-following, answer commitment, and trajectory consistency more than it improves pure search depth.
- This should show up most clearly on `tau-bench`, `IFEval`, and multi-trial agent runs.

### H6: Rank controls agent style, not just agent skill

Expected result:

- Low ranks should look more conservative, brittle, and easier to derail.
- Medium ranks should produce the best tradeoff between decisive action and policy compliance.
- Very high ranks may overfit local preferences and become less stable under long trajectories or tool noise.
- The strongest evidence should come from repeated-run tool-use and planning traces, not from single scalar scores.

### H7: Agentic failures are more predictable from geometry than static benchmark failures

Expected result:

- Adapters with similar static benchmark scores may still diverge sharply in trajectory-level behavior.
- Subspace spread, singular spectra, and DPO rotation magnitude may predict retry rates, tool misuse, and recovery failure better than QA accuracy alone.

## Figures and Tables To Target

### Table A: Core benchmark matrix

- Rows: model + rank + stage
- Columns: GSM8K, MATH-500, ARC, MMLU, HumanEval, MBPP

### Table B: DPO delta table

- Rows: model + rank
- Columns: benchmark-wise `DPO - merged_DARE`

### Table C: Efficiency table

- Score normalized by trainable parameters and wall-clock training time

### Table D: Agent reliability table

- Rows: model + rank + stage + scaffold
- Columns: success, `pass@k`, first-action accuracy, invalid-tool rate, policy violations, recovery rate, mean trajectory length

### Figure 1: Rank scaling curves

- Benchmark score vs rank for each model

### Figure 2: Geometry vs interference

- X-axis: subspace overlap / CKA / principal-angle similarity
- Y-axis: performance drop after merge or mixed activation

### Figure 3: Cross-scale alignment heatmap

- Pairwise similarity between adapters trained on the same knowledge across 0.8B, 4B, 30B

### Figure 4: Agentic stability curves

- X-axis: retry budget or trial count
- Y-axis: success, pass^k, or trajectory consistency

### Figure 5: Tool-use confusion matrix

- Rows: model + rank + stage
- Columns: correct tool call, extra tool call, missing tool call, invalid arguments, policy violation, recovery success

### Figure 6: DPO rotation vs reliability delta

- X-axis: pre/post-DPO principal-angle or CKA delta
- Y-axis: change in `pass@k`, policy compliance, or recovery rate

## Agentic Experiment Design

### Core protocol

For every comparable adapter, test the same underlying model under multiple scaffolds:

- Direct single-shot answer
- `Chain-of-thought` or hidden reasoning variant if supported
- `ReAct` style interleaved reasoning and acting
- `Plan-then-execute`
- `Self-critique` or reflection-on-failure retry

This isolates whether the adapter changes intrinsic capability or only changes sensitivity to agent scaffolding.

### Metrics beyond accuracy

- `pass@k` or `success@k` over repeated trials
- first-action accuracy
- tool precision / recall
- invalid-call rate
- over-tooling rate
- under-tooling rate
- policy-violation rate
- recovery rate after injected failure
- trajectory length to success
- answer change rate across reruns
- action-sequence variance across reruns
- latency-normalized success
- latency / tokens / tool calls per successful task

### Truly novel experiments worth doing

These are the experiments that are less settled in the literature and more likely to produce a paper-worthy contribution.

1. `Scaffold sensitivity map`

- Run the same adapter across direct, ReAct, plan-execute, and reflection settings.
- Measure whether DPO reduces scaffold dependence or merely shifts the best scaffold.

2. `Tool temptation test`

- Give the agent optional but unnecessary tools.
- Measure over-calling vs under-calling by rank and by DPO stage.

3. `Policy-pressure test`

- Mix tool tasks with explicit constraints such as "do not refund without verification" or "do not browse external sites."
- Measure whether DPO improves policy obedience at the expense of task completion.

4. `Injected-failure recovery`

- Corrupt one tool result, remove one observation, or delay one step.
- Measure which adapters recover versus spiral.

5. `Trajectory consistency`

- Re-run identical tasks 8 to 16 times with different seeds.
- Use pass^k, action edit distance, and answer variance to quantify whether some adapters are more behaviorally coherent.

6. `Cross-scale same-knowledge agent study`

- Compare 0.8B, 4B, and 30B adapters trained on the same knowledge under the same agent loop.
- Test whether scale mainly improves planning horizon, tool calibration, or recovery.

7. `Wrong-adapter routing stress test`

- Intentionally route a task through a mismatched adapter before allowing correction.
- Measure whether the model can self-diagnose mismatch and recover.

## Recommended Benchmark Stack

### Minimum publishable stack

- `PlanBench`
- `tau-bench`
- `ToolSandbox` or `BFCL`

This is the cheapest serious stack that still supports claims about planning, tool behavior, policy reliability, and repeated-run stability.

### Strong conference stack

- `PlanBench`
- `tau-bench`
- `ToolSandbox`
- `WebArena-Verified` subset
- reliability and noise overlays on at least `tau-bench` and `ToolSandbox`

This should be the default target for the first serious paper.

### Frontier stretch stack

- everything in the strong conference stack
- `GAIA` subset
- `DeepPlanning`
- small `OSWorld` slice

This should only be attempted after the core paper matrix is stable.

## Benchmark Order To Actually Run

1. `Tier 1` static sanity benchmarks already wired in the repo
2. `PlanBench` as the first true agentic benchmark
3. `tau-bench` as the first policy-and-tool benchmark
4. `ToolSandbox` or `BFCL` for invalid-call and structured tool-use analysis
5. reliability/noise overlays on the above
6. `WebArena-Verified` subset for realism
7. optional `GAIA`, `DeepPlanning`, or `OSWorld` after the main story is already supported

## Market / Literature Positioning

What the current benchmark landscape already covers:

- `PlanBench`: planning competence
- `GAIA`: broad assistant ability
- `tau-bench`: tool + user + policy interaction
- `WebArena`: realistic web trajectories
- `SWE-bench`: software engineering task completion
- `OSWorld`: multimodal computer control

What remains relatively open, and where this project can contribute:

- How low-rank adapters alter trajectory-level behavior, not just answer quality
- Whether DPO changes policy reliability, recovery ability, and consistency under repeated trials
- Whether adapter geometry can predict agent failure modes
- Whether the same knowledge at different scales yields aligned or qualitatively different agent policies
- Whether very high-rank adapters become more capable or merely more behaviorally unstable

## Unresolved Questions Worth Testing

1. Does DPO mostly sharpen the readout of existing knowledge, or does it materially reshape the adapter subspace?
2. Is high-rank DPO actually useful, or is it just extra capacity that the optimizer cannot exploit cleanly?
3. Are merged adapters limited by benchmark mismatch, routing mismatch, or representational overlap?
4. Can a geometry-based metric predict in advance which adapters are safe to merge?
5. Does the same task become more low-rank or more distributed as base model size increases?

## Immediate Tasklist

### Phase 1: Training and static sanity

- Finish all missing Qwen DPO ranks first.
- Finish missing Nemotron 4B DPO ranks second.
- Start Tier 1 benchmark sweep immediately after training.
- Save JSON results and a markdown summary automatically.

### Phase 2: Comparator cleanup

- Evaluate pre-DPO `merged_DARE` comparators for the same model/rank pairs.
- Add base-model baselines once per model size.
- Add single-domain SFT adapters where needed for attribution.

### Phase 3: Planning competence

- Add `PlanBench` harness first.
- Run direct, ReAct, and plan-execute scaffolds on a bounded subset.
- Compare `merged_DARE` vs `DPO` first on a smaller rank set before expanding to all ranks.

### Phase 4: Tool and policy reliability

- Add `tau-bench` next.
- Add `ToolSandbox` or `BFCL` after `tau-bench` logging is stable.
- Log success, `pass@k`, first-action accuracy, invalid calls, over-tooling, under-tooling, policy violations, and retries.

### Phase 5: Reliability overlays

- Add repeated-run evaluation to `PlanBench`, `tau-bench`, and `ToolSandbox`.
- Add tool-noise and observation-noise injections.
- Add policy-pressure prompts and wrong-adapter routing tests.

### Phase 6: Representation science

- Export adapter tensors for all comparable checkpoints.
- Compute subspace overlap, CKA, singular value spectra, and rank utilization.
- Compute DPO rotation magnitude from `merged_DARE` to final `DPO`.
- Correlate geometry metrics with benchmark deltas, interference, scaffold sensitivity, and failure modes.

### Phase 7: Realistic web evaluation

- Run a curated `WebArena-Verified` subset with the best scaffolds from earlier phases.
- Add `GAIA` subset only after the browser/tool harness is stable.
- Keep this phase small enough that it validates the story rather than dominating the whole paper.

### Phase 8: Scale study

- Bring Nemotron 30B adapters into the same benchmark and geometry pipeline.
- Compare whether larger models need lower or higher effective adapter rank.
- Measure cross-scale CKA and principal-angle structure for the same domain and stage.

### Phase 9: Final paper package

- Freeze benchmark splits and seeds.
- Re-run the winning matrix at full sample counts.
- Produce tables, plots, and failure-case examples.
- Write discussion around what DPO changes in adapters beyond raw benchmark lift.
- Include one section explicitly on agent policy, stability, and recovery rather than only benchmark accuracy.

## Reviewer Attack Surface

The likely objections are:

- too many moving parts, with unclear causal attribution
- benchmark sprawl without a clean central story
- overclaiming from single-run agent results
- presenting `rank3072` as a normal LoRA regime
- treating geometry correlations as mechanistic proof

Mitigations:

- pre-commit to a small core matrix before exploratory expansions
- keep the headline on `geometry -> reliability / failure`, not on leaderboard count
- use repeated trials and confidence intervals for every agentic claim
- label `1024, 3072` as near-full-rank or overparameterized controls where appropriate
- report geometry as predictive structure unless explicit interventions are added
