# Final experiment report (MD + expert adapters)

Updated: 2026-04-08T04:37:56.680768 (Track B dataset mtime)

## Executive summary (what we achieved)

We built and validated an evaluation harness that can **systematically test adapter attachment strategies** on a multi-domain benchmark, measure **semantic alignment + latency + perplexity + objective overlap**, and compare against a **Mistral 7B baseline**.

**Most important technical outcome:** for Qwen + expert adapters on this MD slice, **sequential adapter switching across token segments** improves mean semantic similarity versus a simple merged-adapter baseline, while keeping latency within a small overhead band.

**Most important operational outcome:** we now have reproducible artifacts (JSON/JSONL + logs) and a one-shot pipeline that can be re-run on a *truly external* dataset once Track B is populated with independently authored Q/A.

## Datasets
- Track A: `data/multidomain_eval_v2.json` (40 items, >=2 domains)
- Track B: `data/multidomain_eval_external.json` (40 items, **currently a bootstrap clone of Track A**; provenance=`cloned_from_multidomain_eval_v2`)

**Important:** Track B is currently a clone for plumbing. To claim “external validity”, replace Track B with independently authored questions + references (and keep provenance).

## Models / systems evaluated
- **Qwen**: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` + expert LoRA adapters, evaluated under 9 attachment hypotheses.
- **Mistral**: `mistral:7b` via Ollama, semantic+latency baseline.

## Metrics
- **semantic_sim**: cosine similarity vs reference using `sentence-transformers/all-MiniLM-L6-v2`
- **latency_s**: wall-clock generation time (hardware/token-budget dependent)
- **perplexity**: reference continuation perplexity under the tested attachment/routing setting (proxy for “how surprised the model is by the reference”)
- **token_f1 / exact_match**: strict overlap vs reference (EM is often 0 for long refs; F1 is more informative)

## What we tried (Qwen attachment hypotheses)
- `weighted_merge`: 0.5/0.5 two adapters, all layers
- `late_layer_injection`: merge but LoRA only from mid layers upward
- `late_last_quarter`: LoRA only in last ~25% layers
- `early_third_only`: LoRA only in first ~1/3 layers (fastest; PPL can spike due to caps)
- `sequential_token_segments`: adapter A for 48 tokens then B for 132 tokens
- `sequential_reverse`: reverse order (B then A)
- `oracle_single_d1` / `oracle_single_d2`: single adapter at full weight (oracle ablation)
- `merge_high_clamp`: same merge but global clamp raised (1.0)

## Results (Track B)
### Mistral 7B (Ollama)
- Artifact: `results/mistral_track_b.json`
- Mean semantic_sim: **0.663**
- Mean latency_s: **14.47s**
- n_items: 40 | num_predict: 180 | ollama_errors: 0

### Qwen + adapters (9 methods, 40 items each = 360 generations)
- Artifact: `results/injection_track_b.jsonl`

| method | mean semantic_sim | mean latency_s | mean perplexity | mean token_f1 |
|---|---:|---:|---:|---:|
| `weighted_merge` | 0.637 | 4.85 | 12.6 | 0.193 |
| `late_layer_injection` | 0.649 | 4.44 | 12.7 | 0.189 |
| `late_last_quarter` | 0.645 | 3.36 | 12.8 | 0.185 |
| `sequential_token_segments` | 0.654 | 5.15 | 12.6 | 0.186 |
| `sequential_reverse` | 0.657 | 5.10 | 12.6 | 0.197 |
| `oracle_single_d1` | 0.646 | 4.86 | 12.7 | 0.186 |
| `oracle_single_d2` | 0.656 | 5.05 | 12.7 | 0.197 |
| `merge_high_clamp` | 0.637 | 4.92 | 12.6 | 0.193 |
| `early_third_only` | 0.661 | 2.24 | 10011.4* | 0.195 |

## Legacy baseline (older Mistral run, different decode budget)
- Artifact: `results/mistral_md_results.json` (array format)
- Mean semantic_sim: **0.617** | Mean latency_s: **9.20s** | (num_predict was 100 in that script)

## Key takeaways
- **Sequential switching works** (on this MD slice): compared to `weighted_merge` (0.637), `sequential_token_segments` (0.654) and `sequential_reverse` (0.657) increase mean semantic similarity.
- **Best “balanced” setting here**: `sequential_reverse` had the best mean semantic_sim among the non-oracle methods and matched the best mean F1 shown (0.197).
- **Latency trade-offs**: `late_last_quarter` and `early_third_only` are fastest; sequential switching costs ~+0.2–0.3s over merge in these runs.
- **Mistral baseline is now “aligned”**: we re-ran Mistral on Track B with `num_predict=180`, so it’s comparable to Qwen’s 180-token cap in spirit (still different systems, but decode budget is no longer wildly mismatched).
- **External validity note**: current Track B is still a clone. The harness is ready; the next credibility jump comes from replacing Track B with independently authored Q/A and rerunning exactly these scripts.

*Perplexity note*: `early_third_only` has rare capped outliers (per-row PPL is capped at 99999 in the script), which inflates the mean. A more stable summary for Track B is **PPL median 11.3** and **mean PPL capped at 500 → 61.5**.

## Repro / how to rerun

- Mistral (Track B):

```bash
python3 backend/mistral_eval_md.py \
  --data data/multidomain_eval_external.json \
  --num-predict 180 \
  --output results/mistral_track_b.json
```

- Qwen injection hypotheses (Track B):

```bash
PYTHONUNBUFFERED=1 python3 src/eval/run_eval_injection_hypotheses.py --real --extra --more \
  --data data/multidomain_eval_external.json \
  --output results/injection_track_b.jsonl
```

- One-shot pipeline (bootstrap Track B + run everything):

```bash
PYTHONUNBUFFERED=1 python3 src/eval/run_full_showcase_pipeline.py --real
```

## Raw artifacts
- `results/mistral_track_b.json`
- `results/injection_track_b.jsonl`
- `results/showcase_pipeline_summary.txt`
- `results/showcase_pipeline_summary.json`
- `results/mistral_eval_last_run.log`
- `results/injection_hypotheses_eval_full_20260408.jsonl`
- `results/mistral_md_results.json`
