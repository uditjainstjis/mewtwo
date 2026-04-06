import os
import sys
from typing import Optional


DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_CUDA_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CUDA_VRAM_ENV = "CUDA_VRAM_GB"
CUDA_RESERVE_ENV = "CUDA_RESERVE_GB"


def _normalize_backend(value: Optional[str]) -> str:
    if not value:
        return "auto"
    value = value.strip().lower()
    if value in {"auto", "mlx", "cuda"}:
        return value
    raise ValueError(f"Unsupported backend '{value}'. Expected one of: auto, mlx, cuda.")


def detect_runtime_backend(preferred: Optional[str] = None) -> str:
    backend = _normalize_backend(
        preferred
        or os.getenv("MEWTWO_BACKEND")
        or os.getenv("INFERENCE_BACKEND")
        or os.getenv("MODEL_BACKEND")
    )
    if backend != "auto":
        return backend

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    try:
        import mlx.core  # noqa: F401

        return "mlx"
    except Exception:
        pass

    raise RuntimeError(
        "No supported inference backend found. Install either MLX (`mlx`, `mlx-lm`) "
        "or CUDA dependencies (`torch`, `transformers`, `peft`, `accelerate`)."
    )


def resolve_model_path(model_path: Optional[str], backend: Optional[str] = None) -> str:
    runtime = detect_runtime_backend(backend)
    explicit = model_path or os.getenv("BASE_MODEL")
    if explicit:
        if runtime == "cuda":
            cuda_override = os.getenv("CUDA_MODEL_ID")
            if cuda_override:
                return cuda_override
            if explicit.startswith("mlx-community/"):
                stripped = explicit.split("/", 1)[1]
                if stripped.endswith("-4bit"):
                    stripped = stripped[: -len("-4bit")]
                return os.getenv("CUDA_MODEL_FALLBACK", f"Qwen/{stripped}")
        return explicit

    if runtime == "cuda":
        return os.getenv("CUDA_MODEL_ID", DEFAULT_CUDA_MODEL)
    return os.getenv("MLX_MODEL_ID", DEFAULT_MLX_MODEL)


def get_runtime_summary(model_path: Optional[str] = None, backend: Optional[str] = None) -> dict:
    runtime = detect_runtime_backend(backend)
    summary = {"backend": runtime, "model_path": resolve_model_path(model_path, runtime)}
    if runtime == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                summary["device"] = torch.cuda.get_device_name(0)
                summary["total_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
        except Exception:
            pass
    return summary


def get_cuda_max_memory() -> dict:
    total_gb = float(os.getenv(CUDA_VRAM_ENV, "32"))
    reserve_gb = float(os.getenv(CUDA_RESERVE_ENV, "2"))
    usable_gb = max(1, int(total_gb - reserve_gb))
    return {0: f"{usable_gb}GiB", "cpu": os.getenv("CPU_OFFLOAD_RAM", "64GiB")}


def build_lora_train_command(
    base_model: str,
    data_dir: str,
    iters: int,
    batch_size: int,
    learning_rate: float,
    adapter_path: str,
) -> list[str]:
    runtime = detect_runtime_backend()
    resolved_model = resolve_model_path(base_model, runtime)
    if runtime == "mlx":
        return [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            resolved_model,
            "--train",
            "--data",
            data_dir,
            "--iters",
            str(iters),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--adapter-path",
            adapter_path,
        ]

    trainer_script = os.path.join(os.path.dirname(__file__), "train_cuda_lora.py")
    return [
        sys.executable,
        trainer_script,
        "--model",
        resolved_model,
        "--data",
        data_dir,
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--adapter-path",
        adapter_path,
    ]
