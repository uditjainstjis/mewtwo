# CUDA and MLX Compatibility Changes

This repo was refactored to keep MLX support intact while adding a CUDA runtime path for NVIDIA GPUs such as the RTX 5090.

## Runtime behavior

- Backend selection is now automatic.
- If `torch.cuda.is_available()` is true, the repo prefers CUDA.
- If CUDA is unavailable and MLX is installed, it falls back to MLX.
- You can override selection with `MEWTWO_BACKEND=mlx` or `MEWTWO_BACKEND=cuda`.

## Model resolution

- Existing code still passes `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
- On CUDA, that MLX model id is translated to a Transformers model id.
- Default CUDA fallback is `Qwen/Qwen2.5-1.5B-Instruct`.
- Override with `CUDA_MODEL_ID=...` if you want a different NVIDIA-friendly checkpoint.

## VRAM usage

- The CUDA loader uses `device_map="auto"` and a VRAM budget derived from `CUDA_VRAM_GB`.
- Default budget assumes a 32 GB GPU with 2 GB reserved for headroom.
- Override with:
  - `CUDA_VRAM_GB=32`
  - `CUDA_RESERVE_GB=2`

## Files changed

- `backend/runtime_backend.py`
  Added backend detection, model-id translation, CUDA VRAM budgeting, and unified LoRA training command selection.

- `backend/dynamic_mlx_inference.py`
  Reworked from MLX-only inference into a dual-backend engine.
  Added:
  - MLX LoRA wrapper path
  - Torch CUDA LoRA wrapper path
  - CUDA generation via Transformers
  - CUDA perplexity computation
  Preserved:
  - existing clamp logic
  - existing adapter registry loading
  - existing entrypoint name `DynamicEngine`

- `backend/train_cuda_lora.py`
  New CUDA LoRA trainer using Transformers + PEFT for training adapters on NVIDIA hardware.

- `backend/train_adapters.py`
  Replaced hardcoded `mlx_lm lora` subprocess invocation with backend-aware command generation.

- `backend/setup_expert_20.py`
  Replaced hardcoded MLX training command with backend-aware training selection.

- `backend/setup.py`
  Removed MLX-only model inspection assumptions so synthetic adapter generation works for either MLX or CUDA model loading.

- `backend/main.py`
  API model status endpoint now reports the resolved backend and model path.

- `backend/ablation_benchmark.py`
  Removed an unused direct MLX import so the script does not fail immediately on CUDA-only environments.

- `backend/requirements.txt`
  Added CUDA path dependencies:
  - `torch`
  - `transformers`
  - `peft`
  - `accelerate`

- `requirements.txt`
  Added the same CUDA path dependencies at repo level.

## Notes

- MLX was not removed.
- The same public inference class name is preserved so existing evaluation scripts keep working.
- Adapter safetensors remain loadable in both paths because the LoRA A/B matrices are read from safetensors and injected at runtime.
- CUDA training assumes a standard Hugging Face causal LM checkpoint compatible with `transformers` and `peft`.
