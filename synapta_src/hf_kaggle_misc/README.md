# Mewtwo Open-Source Multi-Rank MoE Pipeline

Welcome to the Mewtwo pipeline for training advanced domain adapters (LoRI-MoE Architecture) on 4B and Sub-1B LLMs utilizing SFT, DPO, and proprietary ranking.

## Project Structure

- `advanced_training_pipeline.py`: DPO runner for the merged multi-domain adapters. It discovers completed `*_merged_DARE_rank*` artifacts and trains the final preference adapters for both Qwen 0.8B and Nemotron 4B across ranks `[1, 2, 8, 128, 1024, 3072]`.
- `Multi_Rank_MoE_Training.ipynb`: Kaggle/Google Colab ready Jupyter Notebook utilizing `trl` that you can directly upload and execute for open-source sharing.
- `README.md`: This configuration file.

## Why use this pipeline?

This pipeline is intentionally biased toward survival on limited VRAM. The DPO stage uses 4-bit loading, gradient checkpointing, shorter sequence defaults, and CPU offload folders so failed high-rank attempts do not take the whole matrix down.

## DPO Target Matrix

The current open-source DPO stage is not a full `domain x technique` matrix. It trains one final preference adapter per `model x rank`, using the merged DARE adapter as the starting point and writing outputs with the legacy names:

- `qwen_0.8b_math_DPO_rank{1,2,8,128,1024,3072}`
- `nemotron_4b_math_DPO_rank{1,2,8,128,1024,3072}`

The `math_DPO` suffix is kept for compatibility with the existing output tree and dashboard, even though the stage consumes the merged multi-domain adapter.

### Supported Models
- `nvidia/Nemotron-Mini-4B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct` (Or `Qwen3.5-0.8B` equivalent)

## Getting Started
Ensure you have the required dependencies:
```bash
pip install transformers peft trl datasets bitsandbytes accelerate
```
Then execute:
```bash
python advanced_training_pipeline.py --output_dir ./kaggle_outputs
```
