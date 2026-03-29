Build a world-class single-page startup site for **Synapta**.

This site is being used as the primary reference in a startup pitch event application. Reviewers may know nothing else about the company. The site must therefore act as the entire narrative: startup vision, technical proof, benchmark evidence, research credibility, and commercialization path.

## Non-negotiable constraints

1. Use **only** the claims supported by the provided files.
2. Do **not** invent metrics, model sizes, customers, funding, or traction.
3. Keep the tone high-conviction and premium, but not hypey or dishonest.
4. Treat the flagship adapter-composition work as **one strong proof point inside Synapta**, not as the whole company.
5. If a claim is ambiguous, choose the more conservative wording.

## Company positioning

Synapta is a research-led AI company building privacy-first, cost-efficient, specialized AI systems. The company combines:

- model research,
- deployment infrastructure,
- APIs and productized services,
- custom enterprise AI integrations.

The thesis is that smaller, specialized systems can be dramatically cheaper, more private, and more customizable than relying only on large hosted LLM APIs.

## Design direction

The site must feel elite, technical, and memorable. Avoid generic SaaS layouts.

### Visual tone

- editorial / frontier-lab / premium deep-tech
- bold typography
- dark or near-dark cinematic canvas is fine if executed well
- use strong contrast, structured grids, elegant motion
- no cliché purple AI gradients unless they are highly controlled
- make it feel like a startup that has actually built something difficult

### Motion

Use meaningful animation:

- staggered reveals
- benchmark bars / cards that animate into place
- number counters for metrics
- a button-triggered benchmark demo sequence
- subtle parallax or depth where it helps

### UX

- mobile responsive
- very scannable
- clear CTAs to GitHub, paper, and pitch PDF
- benchmark evidence visible without forcing the user to open external links

## Site structure

### 1. Hero

Headline direction:

- "Private AI. Lower Cost. Built for Real Workflows."

Subheadline direction:

- "Synapta builds specialized AI systems that companies can run privately, customize deeply, and scale without large-model API costs."

Include 2 CTAs:

- `View Proof`
- `Open GitHub / Paper`

Also include a compact proof strip with 3 or 4 metrics visible immediately.

### 2. What Synapta Is

Explain clearly that Synapta is not just a paper and not just an API wrapper.

Present three connected layers:

- Research Lab
- Infrastructure / Deployment
- Services / Enterprise Integrations

Show this as a pipeline:

`Research -> Infra -> Productized Services`

### 3. Flagship Proof

This section is the centerpiece.

Explain that Synapta's recent flagship research built a dynamic multi-adapter Qwen-based system and evaluated it rigorously.

Use the benchmark-supported facts from the provided files, especially:

- `AdaptiveClamp-v2`: `0.6505`
- `SingleAdapter`: `0.6334`
- Mistral-7B benchmark artifact: `0.617`
- Synapta benchmark artifact: `0.6525`
- memory comparison: `~1.1 GB` vs `~4.4 GB`

Use careful phrasing such as:

"On Synapta's multi-domain benchmark, the specialized routed system outperformed Mistral-7B while using roughly one quarter of the memory footprint."

Do not phrase it as universal superiority.

### 4. Interactive Benchmark Demo

Build an animated benchmark component with a button like:

- `Run Benchmark View`

When clicked, animate a comparison between:

- Mistral-7B
- Synapta SingleAdapter
- Synapta Dynamic Multi-Adapter

The animation should feel sexy and technical, but clearly benchmark-based.

Suggested animated elements:

- semantic similarity bars
- memory footprint bars
- cards showing routing mode
- a short sequence showing Synapta dynamic routing activating domain experts

If there is no real live backend, make the animation a polished deterministic visualization of measured numbers from the artifacts.

Label it clearly as benchmark evidence / measured results visualization.

### 5. Why This Matters

Explain the startup wedge:

- lower inference cost
- privacy-first deployment
- local / on-prem capability
- niche customization
- better economics for startups and companies with domain-specific workflows

### 6. Additional Research

Create a section showing that Synapta has a broader research pipeline.

Include cards for:

- Dynamic adapter composition paper
- NeuralGravity
- additional papers / repos / experiments from provided links

NeuralGravity must be framed honestly:

- useful research experiment
- speculative decoding investigation
- not a fake breakthrough
- valuable because it clarified why certain acceleration ideas fail under UMA / deployment constraints

### 7. Founder

Brief founder section for Udit Jain.

Tone:

- technically strong
- execution-focused
- founder-led
- no inflated ego language

Suggested direction:

"Built and driven by Udit Jain, with working systems, research artifacts, and benchmarked prototypes already completed."

### 8. Links / CTA footer

Include:

- GitHub
- flagship paper
- additional papers
- pitch PDF

This section should be clean and frictionless.

## Content style

- concise
- precise
- ambitious without exaggeration
- technical where helpful
- never corporate fluff

## Implementation guidance

- Build as a polished modern landing page
- Use semantic sections and clean component structure
- Make the benchmark module visually dominant
- Make the evidence visible on-page so reviewers do not need to open links to understand the substance
- Prioritize clarity and credibility over overloaded copy

## Use these source files

- `SYNAPTA_SITE_BRIEF.md`
- `Main_Paper_Composition_Updated.md`
- `RESEARCH_EXTRACTION.md`
- `results/mistral_vs_synapta_verified.md`
- `results/v2_decision_summary.md`
- `results/v2_routing_gap_summary.md`
- `results/v2_clamp_ablation_summary.md`
- `results/real_benchmark_table.md`
- pitch PDF and supplied GitHub / paper links

## Final requirement

The finished site should make a reviewer think:

"This founder has real technical depth, a credible wedge, and enough proof that I want to see the live pitch."
