# Adapter routing and multi-expert inference: progress narrative (draft)

**Audience:** investors, design partners, technical diligence.  
**Status:** draft for internal review; numbers below mix **completed runs** and **planned** validation. Update the “External MD” section after the second benchmark is populated and re-evaluated.

---

## One-line pitch

We run a **small instruction-tuned base model** with **on-disk expert LoRA adapters** and measure **when and how** those adapters should attach (merge vs layer gates vs sequential segments). On our multi-domain slice, **sequential adapter segments** improve **semantic alignment** versus a simple merge and run **faster** than a local **Mistral 7B** baseline, while we treat **self-generated benchmarks** as a **development set** only.

---

## Why two benchmarks matter

### Track A — Internal MD (`data/multidomain_eval_v2.json`)

- **Role:** fast iteration, regression testing, hypothesis comparison across nine injection settings.
- **Caveat:** Items were produced in a **closed loop** (e.g. Qwen-family involvement in question or reference generation). **Cosine similarity to the packaged reference** can be **optimistically biased**: the model may share phrasing, entities, or style with the reference it is scored against.
- **Use in storytelling:** “We use Track A to **engineer** routing and adapter attachment; we **do not** cite it as independent proof of end-user quality.”

### Track B — External MD (recommended: `data/multidomain_eval_external.json`)

- **Role:** **credible** showcase metrics when someone asks “isn’t this just the model agreeing with itself?”
- **Recipe (recommended):**
  1. **Questions:** drafted by a **different** stack than the scored generator (e.g. **Claude** or **Perplexity-backed** research mode via `backend/proxy_bridge.py` when `CLUSTER_USE_PERPLEXITY_PROXY=1`, or another local API).
  2. **References:** **human expert** short answers, or externally authored text with citation to sources where applicable.
  3. **Optional quality gate:** **LLM-as-judge** (rubric: correctness, coverage, safety) with **human audit** on a subset.
- **Provenance:** optional `provenance` object per item (see `data/multidomain_eval_external.example.json`). The eval pipeline ignores unknown keys; it helps diligence and versioning.

**Strategic line:** “We report **Track A** for **R&D velocity** and **Track B** for **external validity**.”

---

## What we measure (aligned across baselines)

| Metric | Meaning |
|--------|--------|
| **Semantic similarity** | Cosine similarity of model output vs reference using `sentence-transformers/all-MiniLM-L6-v2` (same family as legacy Mistral vs Synapta scripts). |
| **Latency** | Wall-clock generation time per item (hardware-dependent; report machine class). |
| **Token F1 / exact match** | Strict overlap vs reference; long references often drive EM to zero; F1 is still useful for relative comparisons. |
| **Perplexity** | Model-side score of reference continuation under stated adapter routing (where defined; some layer-gate paths need careful interpretation). |

---

## Results snapshot (Track A, completed)

**Setup:** 40 multi-domain items requiring **≥2** adapters; Qwen `mlx-community/Qwen2.5-1.5B-Instruct-4bit` + registry LoRAs; nine methods (`--real --extra --more`).  
**Artifact:** `results/injection_hypotheses_eval_full_20260408.jsonl` (and log `results/injection_hypotheses_full_run.log`).

**Aggregate means (Track A):**

| Method | Mean semantic sim | Mean latency (s) | Mean token F1 |
|--------|-------------------|------------------|---------------|
| weighted_merge | 0.637 | 4.88 | 0.193 |
| late_layer_injection | 0.649 | 4.46 | 0.189 |
| **sequential_token_segments** | **0.654** | **5.10** | 0.186 |
| late_last_quarter | 0.645 | 3.30 | 0.185 |
| **sequential_reverse** | **0.657** | **5.09** | **0.197** |
| early_third_only | 0.661 | 2.23 | 0.195 |
| oracle_single_d1 | 0.646 | 4.83 | 0.186 |
| oracle_single_d2 | 0.656 | 4.84 | 0.197 |
| merge_high_clamp | 0.637 | 4.84 | 0.193 |

**Note:** `early_third_only` mean perplexity was dominated by rare capped outliers in the log; treat **PPL** for that method as **not showcase-ready** until cleaned or recomputed with a stable estimator.

**Mistral 7B baseline (cached, same MD file, different decode budget):**  
From `results/mistral_md_results.json`: mean semantic sim **0.617**, mean latency **9.20 s** (Ollama `num_predict`: 100). Qwen runs used **max_tokens** 180, so **latency and semantic sim are not matched for length**; for diligence, re-run Mistral with aligned `num_predict` and quote both.

**Talking point (careful):** On Track A, **sequential adapter scheduling** sits **above** Mistral on **mean MiniLM similarity** and **below** Mistral on **wall time**, with the **decode-length caveat** above and the **self-benchmark caveat** for semantics.

---

## Planned table for the deck (Track B — fill after run)

| Benchmark | N items | Best showcase method | Semantic sim | Latency | F1 | Notes |
|-----------|---------|----------------------|--------------|---------|-----|-------|
| Track A (internal) | 40 | e.g. sequential_reverse | (see above) | (see above) | (see above) | R&D / regression |
| Track B (external) | TBD | TBD | TBD | TBD | TBD | **Investor-grade** |

---

## How to reproduce and extend

**Single command (bootstrap Track B, Mistral A+B, Qwen injection A+B, aggregates):**

```bash
cd /path/to/adapter
PYTHONUNBUFFERED=1 python3 src/eval/run_full_showcase_pipeline.py --real
# Smoke: add --limit 2
# Partial proxy Track B: CLUSTER_USE_PERPLEXITY_PROXY=1 ... --proxy-items 5
```

Artifacts: `results/mistral_track_a.json`, `results/mistral_track_b.json`, `results/injection_track_a.jsonl`, `results/injection_track_b.jsonl`, `results/showcase_pipeline_summary.{json,txt}`. Add `--with-sats` for `results/sats_eval_showcase.jsonl`, `--with-cluster` for `results/cluster_strict_showcase.jsonl`.

Generic per-method averages: `python3 src/eval/aggregate_eval_jsonl.py results/sats_eval.jsonl`

Preflight (data + optional Ollama / mlx): `python3 src/eval/check_eval_environment.py`  
Injection aggregates with capped PPL mean (e.g. early_third outliers):  
`python3 src/eval/aggregate_injection_jsonl.py results/injection_track_a.jsonl --ppl-cap 500`

SATS and cluster evals default to the **same ≥2-domain subset** as injection (`md_dataset.prepare_md_items`); use `--all-items` to include every row.

Mistral preflight calls Ollama `GET /api/tags` (override host with `OLLAMA_HOST`). Failed runs record `n_ollama_errors` and per-item `ollama_error` in the JSON.

**Full hypothesis suite on default MD (manual):**

```bash
cd /path/to/adapter
PYTHONUNBUFFERED=1 python3 src/eval/run_eval_injection_hypotheses.py \
  --real --extra --more \
  --output results/injection_hypotheses_eval_full.jsonl
```

**Same suite on external MD file (after you replace placeholders):**

```bash
PYTHONUNBUFFERED=1 python3 src/eval/run_eval_injection_hypotheses.py \
  --real --extra --more \
  --data data/multidomain_eval_external.json \
  --output results/injection_hypotheses_eval_external.jsonl
```

Copy `data/multidomain_eval_external.example.json` to `data/multidomain_eval_external.json` and expand to a full list. Items must include `id`, `question`, `reference_answer`, and two domains present in `backend/expert_registry.json` (via `domains` or `required_adapters`).

**Mistral MD (refresh for semantics on Track B):**

```bash
# From repo root; requires Ollama + mistral:7b
python3 backend/mistral_eval_md.py --data data/multidomain_eval_v2.json --num-predict 100

# Align decode budget with Qwen injection (max_tokens=180) before publishing comparisons:
python3 backend/mistral_eval_md.py --data data/multidomain_eval_v2.json --num-predict 180 \
  --output results/mistral_md_results_aligned.json

# Track B file when ready:
python3 backend/mistral_eval_md.py --data data/multidomain_eval_external.json \
  --output results/mistral_md_external.json
```

Output JSON includes `dataset_path`, `num_predict`, `mean_similarity`, `mean_latency_s`, and `items` (per-item list). Older `results/mistral_md_results.json` may be a bare array from an earlier script version.

**Summarize any injection JSONL:**

```bash
python3 src/eval/aggregate_injection_jsonl.py results/injection_hypotheses_eval_full_20260408.jsonl
python3 src/eval/aggregate_injection_jsonl.py results/injection_hypotheses_eval_external.jsonl --by-dataset
```

**Draft one Track B item (Perplexity proxy):**

```bash
CLUSTER_USE_PERPLEXITY_PROXY=1 python3 src/eval/draft_external_md_item.py \
  --id ext_02 --domain-a LEGAL_ANALYSIS --domain-b PYTHON_LOGIC
```

Paste the JSON into a `data/multidomain_eval_external.json` array; then **edit** references for human expert verification before trusting metrics.

**Perplexity / Claude for content generation:** enable `CLUSTER_USE_PERPLEXITY_PROXY=1` and install the proxy per `backend/proxy_bridge.py`; use it for **dataset authoring** and/or **agent-cluster recovery**, not as a substitute for frozen benchmark JSON in published numbers unless you document it explicitly.

---

## Risk language (use verbatim if helpful)

- Semantic scores on **internally generated** Q&A are **directional**, not **independent** validation.
- We are building a **second benchmark** with **external authorship** and optional **human + judge** review so **showcase metrics** and **R&D metrics** stay **separated**.
- All latency figures are **machine- and token-budget-specific**; we will publish **both** budget and hardware when quoting comparisons.

---

## Changelog (for your team)

- **2026-04-08:** Draft narrative; Track A full nine-method eval logged; injection script supports `--data`, `--output`, and per-row `dataset_path` in JSONL; external MD example + reproduction commands added.
- **2026-04-08 (continued):** `mistral_eval_md.py` CLI (`--data`, `--num-predict`, `--output`, two-domain filter); `aggregate_injection_jsonl.py`; `draft_external_md_item.py` for proxy-assisted Track B drafts.
- **2026-04-08 (continued 2):** Ollama preflight + structured errors in Mistral JSON; `run_eval_sats.py --output`; showcase pipeline `--with-sats`, `--mistral-model`, Mistral summaries in `showcase_pipeline_summary.txt`.
- **2026-04-08 (continued 3):** `aggregate_eval_jsonl.py`; `run_eval_cluster_strict.py` and `run_eval_sats.py` accept `--data`; cluster eval `--output`/`--limit`; fixed `cluster_strict` standard-branch undefined locals; pipeline `--with-cluster`.
- **2026-04-08 (continued 4):** `md_dataset.py` shared loader/filter; SATS/cluster aligned with injection’s ≥2-domain default (`--all-items` to override); `aggregate_injection_jsonl.py --ppl-cap` + `PPL_med`; `check_eval_environment.py`.
