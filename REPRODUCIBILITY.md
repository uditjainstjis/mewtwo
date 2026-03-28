# Reproducibility Guide

> **Paper:** Prompt-Level Multi-Adapter Composition with Norm-Proportional Clamping: A Pre-Registered Negative Result
>
> **Tag:** `v1-negative-result` | **Model:** Qwen2.5-1.5B-Instruct-4bit | **Hardware:** Apple Silicon (M-series)

---

## 1. Environment Setup

### Hardware Requirements
- Apple Silicon Mac (M1/M2/M3/M4, any tier) with ≥ 16 GB unified memory
- ~5 GB disk for model weights + adapters

### Software
```bash
# Python 3.11+
python3 --version

# Clone the repo
git clone https://github.com/<your-username>/adapter.git
cd adapter

# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
| Package | Purpose |
|---------|---------|
| `mlx >= 0.21.0` | Apple Silicon ML framework |
| `mlx-lm >= 0.19.0` | LLM loading & generation |
| `safetensors >= 0.4.0` | Adapter weight I/O |
| `sentence-transformers >= 2.2.0` | Semantic similarity evaluation |
| `numpy >= 1.24.0` | Numerics |
| `pyyaml >= 6.0` | Config parsing |

---

## 2. Model & Adapter Weights

### Base Model
The base model is auto-downloaded on first run:
```
mlx-community/Qwen2.5-1.5B-Instruct-4bit
```
Hosted on HuggingFace. No manual download needed — `mlx_lm.load()` handles it.

### Expert Adapters (20 domains)
The 20 LoRA adapters live in `backend/expert_adapters/`. Each is a ~20 MB `.safetensors` file.

| Domain | Path |
|--------|------|
| LEGAL_ANALYSIS | `backend/expert_adapters/LEGAL_ANALYSIS/adapters.safetensors` |
| MEDICAL_DIAGNOSIS | `backend/expert_adapters/MEDICAL_DIAGNOSIS/adapters.safetensors` |
| PYTHON_LOGIC | `backend/expert_adapters/PYTHON_LOGIC/adapters.safetensors` |
| MATHEMATICS | `backend/expert_adapters/MATHEMATICS/adapters.safetensors` |
| ... (16 more) | See `backend/expert_registry.json` for full list |

**To re-train adapters from scratch:**
```bash
cd backend
python3 train_adapters.py
```
Training data is in `backend/data_expert/`. Each adapter trains on synthetic domain-specific QA pairs.

---

## 3. Reproducing the Full Benchmark (Table 1)

This is the core experiment: 100 hard questions × 4 methods = 400 real MLX inferences.

```bash
cd /path/to/adapter
export PYTHONPATH=$(pwd)

# Clear any previous results
rm -f results_db.jsonl

# Run the full benchmark (~20 min on M3 Max)
python3 src/eval/real_benchmark.py
```

### What It Does
1. Loads `Qwen2.5-1.5B-Instruct-4bit` into UMA
2. Injects all 20 LoRA adapters via `RoutedLoRALinear`
3. For each of 100 questions (5 per domain × 20 domains):
   - Routes via CoT `Orchestrator`
   - Runs 4 methods: **Baseline**, **SingleAdapter**, **UnclampedMix**, **AdaptiveClamp**
   - Measures: semantic similarity (sentence-transformers), perplexity, wall-clock latency
4. Writes per-question logs to `results_db.jsonl`
5. Generates summary tables in `results/`

### Expected Output Files
| File | Contents |
|------|----------|
| `results/real_benchmark_table.md` | Paper-ready Table 1 (per-domain + summary) |
| `results/real_benchmark_results.json` | Raw per-question data (400 entries) |
| `results_db.jsonl` | Structured logs with `"real_mode": true` |

### Expected Results (±noise from non-deterministic generation)

| Method | K | Clamp | Avg Sim ↑ | Avg PPL ↓ | Avg Latency |
|--------|---|-------|-----------|-----------|-------------|
| Baseline | 1 | 0.001 | ~0.62 | ~64 | ~2.8s |
| SingleAdapter | 1 | 0.5 | ~0.62 | ~61 | ~2.7s |
| UnclampedMix | 2 | 999 | ~0.56 | ~51 | ~2.5s |
| AdaptiveClamp | 2 | 0.5 | ~0.61 | ~58 | ~2.7s |

**Key result:** Δ_SIM(AdaptiveClamp − SingleAdapter) ≈ −0.01 → **FAIL** (threshold was > +0.05)

---

## 4. Reproducing Individual Methods

### Single config run (simulation mode — no GPU, for pipeline testing)
```bash
python3 src/eval/run_eval.py --config_id debug_single_real
```

### Single config run (real mode — requires MLX + adapters)
```bash
python3 src/eval/run_eval.py --config_id debug_single_real --real
python3 src/eval/run_eval.py --config_id debug_adaptive_real --real
```

### Available config IDs
See `configs/uma_experiments.yaml` for the full list.

---

## 5. Reproducing the Original Ablation (Table 2)

This is the earlier 3-config ablation (Baseline / Synapta-Aggressive / Synapta-Balanced) from the DARSA study:

```bash
cd backend
python3 ablation_benchmark.py
```

Outputs `table2_ablation.md` and `ablation_raw_results.json`.

---

## 6. Evaluation Protocol

### Semantic Similarity
- Model: `sentence-transformers/all-MiniLM-L6-v2` (auto-downloaded)
- Metric: Cosine similarity between model output and ground-truth answer
- Range: [0, 1], higher is better

### Perplexity
- Computed via `DynamicEngine.compute_perplexity()`
- Measures log-likelihood of ground-truth text conditioned on the prompt under each routing config
- Lower is better

### Latency
- Wall-clock `time.time()` around `engine.generate()` call
- Includes tokenization + generation + detokenization
- Measured in seconds

### Questions
- 100 hard domain-specific questions in `backend/ablation_benchmark.py::HARD_QUESTIONS`
- 5 questions per domain × 20 domains
- Each question has a synthetic ground-truth answer that ONLY the correct adapter has been trained on
- The base model has NO knowledge of these answers → adapter recall is the signal

---

## 7. Key Architecture Files

| File | Role |
|------|------|
| `backend/dynamic_mlx_inference.py` | `DynamicEngine` + `RoutedLoRALinear` — the real inference engine |
| `backend/orchestrator.py` | CoT-based domain router using the base model itself |
| `src/eval/real_benchmark.py` | Full 4-method benchmark script |
| `src/eval/run_eval.py` | Flexible eval harness with `--real` flag |
| `backend/expert_registry.json` | Adapter paths and metadata for all 20 experts |
| `configs/uma_experiments.yaml` | Experiment configurations |

---

## 8. Verifying Integrity

### Check that results are real (not simulated)
```bash
# Every line should have "real_mode": true
grep -c '"real_mode": true' results_db.jsonl
# Expected: 400

# No simulated entries
grep -c '"real_mode": false' results_db.jsonl
# Expected: 0
```

### Check prediction quality (not template stubs)
```bash
# Predictions should contain actual model text, NOT "def black_scholes(): pass"
python3 -c "
import json
lines = [json.loads(l) for l in open('results_db.jsonl')]
real = [l for l in lines if l['real_mode']]
print(f'Total real entries: {len(real)}')
print(f'Sample prediction: {real[0][\"prediction_preview\"][:150]}')
print(f'Avg similarity: {sum(l[\"metric_value\"] for l in real)/len(real):.3f}')
"
```

---

## 9. Non-Determinism Note

MLX generation is non-deterministic (temperature, sampling). Results will vary slightly across runs.
Our reported numbers are from a single run (no cherry-picking). Reviewers should expect:
- Semantic similarity: ±0.02 variation per method
- Perplexity: ±5 variation per method
- Relative ordering of methods should be stable

---

## 10. Contact & Citation

For questions about reproducibility, open an issue on the GitHub repo.

```bibtex
@misc{adapter2026negative,
  title={Prompt-Level Multi-Adapter Composition with Norm-Proportional Clamping:
         A Pre-Registered Negative Result},
  author={Jain, Udit},
  year={2026},
  note={Pre-registered pilot study. Tag: v1-negative-result}
}
```
