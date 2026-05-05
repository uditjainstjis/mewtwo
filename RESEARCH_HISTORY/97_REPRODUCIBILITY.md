# Reproducibility Reference

## Hardware
- Primary headline experiments: single RTX 5090 (32 GB VRAM).
- Smaller-base zoo (Nemotron-Mini-4B, Qwen-3.5-0.8B): combinations of HuggingFace Spaces and Kaggle community GPU resources.
- Disk: NVMe SSD for adapter cold-swap profiling.

## Software environment
- PyTorch 2.5
- transformers 4.45
- peft 0.13
- bitsandbytes 0.44
- scipy 1.14
- pdfplumber, PyMuPDF (PDF extraction)
- HuggingFace `Rajveer-code/IndiaFinBench` (external benchmark)

## Known compatibility note
Mamba CUDA fast-path is unavailable under 4-bit weight loading on Nemotron-Nano-30B-A3B. We use the naive Mamba implementation with `HybridMambaAttentionDynamicCache`. This is a $\sim 2.5\times$ slowdown (70 s/step floor) but produces correct outputs. Patched at `synapta_src/data_pipeline/08_eval_bfsi_extract.py:get_hybrid_cache()`.

## Pipeline scripts (numbered, runnable end-to-end)
| Script | Purpose |
|---|---|
| `01_scrape_rbi_mds.py` | Async scrape RBI Master Directions |
| `01b_download_hf_datasets.py` | Download HF supplemental datasets |
| `01c_scrape_sebi_circulars.py` | Async scrape SEBI Master Circulars |
| `01d_scrape_sebi_broader.py` | SEBI broader circulars (200 cap) |
| `02_extract_text.py` | PDF → text via pdfplumber+PyMuPDF |
| `03_chunk_circulars.py` | Smart-chunk on numbered sections |
| `04_build_qa_pairs.py` | 3-tier QA construction (v1) |
| `04b_build_qa_pairs_v2.py` | 3-tier QA construction (v2 cleaner) |
| `05_integrate_hf_data.py` | Merge HF auxiliary data |
| `06_validate_qa.py` | 10-check validator |
| `07_train_bfsi_extract.py` | LoRA training (the headline result) |
| `08_eval_bfsi_extract.py` | Held-out 3-mode eval (resumable) |
| `09_demo_bfsi.py` | CLI demo |
| `10_build_recall_dataset.py` | Build no-context QA |
| `11_frontier_comparison.py` | Frontier API/subagent comparison |
| `12_train_bfsi_recall.py` | Train recall adapter |
| `13_eval_bfsi_recall.py` | Eval recall adapter |
| `14_publish_benchmark.py` | HF push (Synapta Benchmark v1) |
| `15_publish_kaggle.py` | Kaggle push |
| `16_eval_indiafinbench.py` | Synapta on IndiaFinBench OOD |
| `17_eval_benchmark_v1.py` | Synapta on Benchmark v1 paired |
| `18_eval_benchmark_v1_fg.py` | Format Guard on Benchmark v1 |

Total wall-clock to reproduce from blank repo to evaluated adapter: $\approx 8$ hours on a single RTX 5090.

## Decoding settings (constant across modes)
- `do_sample = False` (greedy)
- `max_new_tokens = 200`
- `max_input_tokens = 1536` (truncated when context exceeds)
- `pad_token_id = tok.pad_token_id`
- `use_cache = True`
- `past_key_values = HybridMambaAttentionDynamicCache(...)`

## System prompt (constant)
> *"You are a senior banking and financial regulation expert in India. Read the provided regulatory context carefully and answer the question precisely with the specific number, term, rule, or section citation. Quote directly from the regulation when possible."*

## LoRA hyperparameters (bfsi_extract, primary configuration)
- `r = 16`, `alpha = 32`, `dropout = 0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Optimiser: paged AdamW 8-bit, `lr = 2e-4`, cosine schedule
- 1 epoch over 2,931 train pairs
- `MAX_LEN = 1024`
- Wall-clock: 3h 28min on RTX 5090
- Trainable params: 434.6M / 32B = 1.36\%

## Format Guard hyperparameters
- $K = 10$ (swap window in tokens)
- Suffix length for routing: 200 tokens
- Router: regex over decoded suffix (see `04_format_guard.md` for the function)
- All adapters pre-loaded into VRAM via `model.load_adapter(...)`
- Active swap: `model.set_adapter(target)` ($O(1)$ pointer flip)

## How to verify a number
For any quotable number, refer to `99_HEADLINE_NUMBERS.md` which links each number to its primary source artefact (`results/<dir>/summary.json` or `results/<dir>/eval_results.jsonl`). If a number cited in any document is not in `99_HEADLINE_NUMBERS.md`, it should be added there or removed from the citing document.
