# Synapta Site Brief

## Purpose

This site is for a startup pitch event application. It is not a generic company website. Its job is to make reviewers understand, quickly:

1. what Synapta is,
2. why it matters,
3. what has already been built,
4. why the technical work is credible,
5. why this deserves a live pitch slot.

The site should be strong, visual, and high-agency, but factually disciplined.

## Core Positioning

Synapta builds privacy-first AI systems using small, specialized models instead of depending only on large hosted LLM APIs. The company combines:

- research,
- deployment infrastructure,
- productized services,
- custom enterprise AI integrations.

Synapta's thesis:

- smaller, specialized systems can be materially cheaper,
- privacy-sensitive companies need local or controlled deployment,
- targeted optimization can make small models competitive on niche workflows,
- research and services should reinforce each other.

## Important Framing Rules

### Do say

- Synapta is a research-led AI company building privacy-first, cost-efficient, specialized AI systems.
- The current adapter-composition work is one proof point inside the larger Synapta vision.
- Early results show benchmark-specific cases where Synapta's specialized system is competitive with or better than larger models.
- Synapta is building toward self-sustaining research + infrastructure + services.

### Do not say

- "Synapta beats all large models."
- "Synapta universally outperforms LLMs."
- "This one paper is the whole company."
- "The benchmark proves general superiority."

### Critical honesty constraint

The current flagship proof in this repo is based on `Qwen2.5-1.5B-Instruct-4bit` with dynamic adapters, not a verified Qwen 3B result.

If the site says "Qwen 3B beat Mistral 7B," you must provide separate hard evidence from that exact system. Otherwise the site should say:

"A small Qwen-based Synapta system outperformed Mistral-7B on our benchmark."

or more specifically:

"A Qwen2.5-1.5B-based Synapta system outperformed Mistral-7B on the repository's multi-domain benchmark."

## Approved Startup Summary

Synapta is building a privacy-first AI stack for companies that want lower cost, deeper customization, and local deployment. The company develops research improvements on small language models, then turns those improvements into deployable APIs, on-prem systems, and custom AI solutions for startups and businesses.

## Flagship Proof To Surface On The Site

### Flagship system

- Base model: `Qwen2.5-1.5B-Instruct-4bit`
- Runtime: MLX on Apple Silicon
- Method: dynamic multi-adapter routing and composition
- Adapters: 20 domain-specific LoRA experts

### Best benchmark-specific claims currently supported by this repo

1. On the genuine multi-domain split, `AdaptiveClamp-v2` scored:
   - semantic similarity: `0.6505`
   - vs `SingleAdapter`: `0.6334`
   - delta: `+0.0171`

2. In a separate verified Synapta-vs-Mistral comparison artifact:
   - Mistral-7B semantic similarity: `0.617`
   - Synapta semantic similarity: `0.6525`
   - memory footprint: `~4.4 GB` vs `~1.1 GB`

3. Routing-gap ablation:
   - oracle headroom: `+0.0206`
   - real top-2 router realized gain: `+0.0054`
   - about `26%` of oracle headroom recovered

4. Clamp ablation:
   - norm-ratio clamp vs weight cap difference: `-0.0003`
   - meaning clamp formulation is not the main bottleneck at this scale

### Honest interpretation to use

Synapta's research does not claim universal superiority. It shows that on targeted, multi-domain tasks, a small specialized system can outperform a larger general model while using much less memory, and that the company is capable of building original inference systems and evaluating them rigorously.

## How To Present The Mistral Comparison

Use wording like:

"On Synapta's multi-domain benchmark, our specialized routed system achieved `0.6525` semantic similarity versus `0.617` for Mistral-7B, while operating at roughly one quarter of the memory footprint."

Do not use wording like:

"We beat Mistral-7B overall."

## Site Sections

1. Hero
2. Why Synapta
3. Flagship proof
4. Interactive benchmark demo
5. How Synapta works: research -> infra -> services
6. Additional research
7. Founder
8. Links: GitHub, paper, pitch PDF

## Hero Copy Direction

Short, sharp, technical.

Good direction:

- "Private AI. Lower Cost. Built for Real Workflows."
- "Synapta builds specialized AI systems that run where your data lives."
- "Smaller models. Better economics. Full deployment control."

Avoid generic startup fluff.

## Interactive Benchmark Demo

The benchmark demo should be explicitly labeled as a benchmark visualization based on measured results, not a fake "live intelligence" claim.

### Suggested comparison set

- Mistral-7B
- Synapta SingleAdapter
- Synapta Dynamic Multi-Adapter

### Good metrics to animate

- semantic similarity
- memory footprint
- routing mode
- benchmark outcome

### Use with caution

- latency
- perplexity

Only animate these if you have exact source numbers you want shown and can cite them.

## Additional Research Section

This section should show that Synapta is not a one-paper startup.

Recommended structure:

- Flagship proof: dynamic adapter composition
- Research pipeline: additional systems and experiments
- NeuralGravity: speculative decoding experiment, presented honestly as a useful negative-result research effort

### Honest NeuralGravity framing

Use language like:

"NeuralGravity explored speculative decoding for faster inference. The result did not produce a strong practical win in our target setting, but it generated useful engineering knowledge about UMA constraints, multi-tenant tradeoffs, and where speculative acceleration breaks down."

Do not try to spin a failed experiment as a fake breakthrough.

## Services Section Copy Direction

Synapta monetizes through deployment and customization:

- private AI APIs
- on-prem or local deployment
- domain-specific copilots
- custom AI workflow integrations
- consulting + implementation for startups and businesses

## Founder Section

Synapta is currently founder-led by Udit Jain. The site should communicate execution and technical ownership, not ego.

Good phrasing:

"Built and driven by Udit Jain, with working research systems, benchmarks, and deployable prototypes already completed."

## Assets / Links Needed From User

Fill these before finalizing the site:

- GitHub repo URL for Synapta flagship work
- public paper URL or PDF link for flagship work
- pitch deck PDF link
- NeuralGravity GitHub repo URL
- NeuralGravity paper URL or PDF
- any founder photo or Synapta logo, if available
- exact statement on whether to present the flagship system as `Qwen2.5-1.5B` or a separate verified `Qwen 3B`

## Recommended Files To Feed AI Studio

Use these as the primary inputs:

1. `SYNAPTA_SITE_BRIEF.md`
2. `AI_STUDIO_PROMPT.md`
3. `Main_Paper_Composition_Updated.md`
4. `RESEARCH_EXTRACTION.md`
5. `results/mistral_vs_synapta_verified.md`
6. `results/v2_decision_summary.md`
7. `results/v2_routing_gap_summary.md`
8. `results/v2_clamp_ablation_summary.md`
9. `results/real_benchmark_table.md`
10. your pitch PDF

## Files Not Recommended As Primary Inputs

Avoid feeding these directly unless you manually curate them first:

- `Main_Paper_Composition.docx`
- `paper.md`
- `synapta_iclr_final_draft.md`
- `FINAL_EXPERIMENT_REPORT.md`
- `FULL_PROJECT_SUMMARY.md`

Reason: they contain older framing, promotional language, or stale interpretations that can make the generated site overclaim.
