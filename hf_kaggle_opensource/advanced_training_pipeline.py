import argparse
import gc
import inspect
import json
import os
import shutil
import traceback
import time
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import DPOConfig, DPOTrainer

# Reduce allocator fragmentation during repeated model load/unload cycles.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_OFFLOAD_DIR = ROOT / "offload_cache"
DEFAULT_DATA_CACHE_DIR = ROOT / "data" / "dpo_cache"
RANKS = [1, 2, 8, 128, 1024, 3072]
DPO_STAGE_LABEL = "math"  # Legacy naming used by existing artifacts/dashboard.
MODEL_ORDER = {"qwen_0.8b": 0, "nemotron_4b": 1}

MODEL_SPECS = {
    "qwen_0.8b": {
        "local_path": ROOT.parent / "models" / "Qwen3.5-0.8B",
        "hub_id": "Qwen/Qwen3.5-0.8B",
        "cache_dir_name": "models--Qwen--Qwen3.5-0.8B",
        "batch_size": 2,
        "max_length": 640,
        "max_prompt_length": 384,
    },
    "nemotron_4b": {
        "local_path": ROOT.parent / "models" / "nemotron",
        "hub_id": "nvidia/Nemotron-Mini-4B-Instruct",
        "cache_dir_name": "models--nvidia--Nemotron-Mini-4B-Instruct",
        "batch_size": 1,
        "max_length": 512,
        "max_prompt_length": 256,
    },
}

DPO_DATASET_CACHE = {}


class ProgressCallback(TrainerCallback):
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss", "N/A")
        print(f">>> [{self.experiment_name}] step={state.global_step} loss={loss}", flush=True)


def experiment_name(model_key: str, rank: int) -> str:
    return f"{model_key}_{DPO_STAGE_LABEL}_DPO_rank{rank}"


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r") as handle:
            head = handle.read(128)
        return head.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def local_model_snapshot_is_usable(model_key: str) -> bool:
    spec = MODEL_SPECS[model_key]
    local_path = spec["local_path"]
    if not local_path.exists():
        return False

    tokenizer_json = local_path / "tokenizer.json"
    if tokenizer_json.exists() and is_lfs_pointer(tokenizer_json):
        return False

    weight_index = local_path / "model.safetensors.index.json"
    shard_files = list(local_path.glob("model-*.safetensors"))
    single_weight = local_path / "model.safetensors"
    if weight_index.exists() and not shard_files:
        return False
    if not weight_index.exists() and not single_weight.exists():
        return False

    return True


def cached_hf_snapshot(model_key: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / MODEL_SPECS[model_key]["cache_dir_name"]
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        return None

    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        try:
            ref = refs_main.read_text().strip()
            candidate = snapshots_dir / ref
            if candidate.exists():
                return candidate
        except Exception:
            pass

    candidates = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resolve_model_name(model_key: str) -> str:
    spec = MODEL_SPECS[model_key]
    local_path = spec["local_path"]
    if local_model_snapshot_is_usable(model_key):
        return str(local_path)
    cached_snapshot = cached_hf_snapshot(model_key)
    if cached_snapshot is not None:
        return str(cached_snapshot)
    return spec["hub_id"]


def resolve_merged_adapter_path(output_dir: Path, model_key: str, rank: int) -> Path | None:
    merged_root = output_dir / f"{model_key}_merged_DARE_rank{rank}"
    direct = merged_root / "adapter_config.json"
    nested = merged_root / "merged_sft" / "adapter_config.json"
    if direct.exists():
        return merged_root
    if nested.exists():
        return merged_root / "merged_sft"
    return None


def build_dpo_targets(output_dir: Path, model_keys: list[str], ranks: list[int]) -> list[tuple[str, int, Path]]:
    targets: list[tuple[str, int, Path]] = []
    for model_key in model_keys:
        for rank in ranks:
            merged_path = resolve_merged_adapter_path(output_dir, model_key, rank)
            if merged_path is not None:
                targets.append((model_key, rank, merged_path))
    return sorted(targets, key=lambda item: (MODEL_ORDER.get(item[0], 99), item[1]))


def dataset_cache_path(cache_dir: Path, limit: int) -> Path:
    return cache_dir / f"ultrafeedback_binarized_train_prefs_{limit}"


def get_dpo_dataset(limit: int, cache_dir: Path):
    if limit in DPO_DATASET_CACHE:
        return DPO_DATASET_CACHE[limit]

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = dataset_cache_path(cache_dir, limit)
    if dataset_dir.exists():
        print(f"Loading cached DPO dataset from {dataset_dir}...", flush=True)
        dataset = load_from_disk(str(dataset_dir))
        DPO_DATASET_CACHE[limit] = dataset
        return dataset

    print(f"Loading DPO dataset (limit={limit})...", flush=True)
    raw_ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split=f"train_prefs[:{limit}]",
        cache_dir=str(cache_dir / "hf_downloads"),
    )

    def format_dpo(example):
        chosen = example["chosen"]
        rejected = example["rejected"]
        return {
            "prompt": example["prompt"],
            "chosen": chosen[1]["content"] if isinstance(chosen, list) else chosen,
            "rejected": rejected[1]["content"] if isinstance(rejected, list) else rejected,
        }

    keep_columns = raw_ds.column_names
    dataset = raw_ds.map(format_dpo, remove_columns=keep_columns)
    dataset.save_to_disk(str(dataset_dir))
    DPO_DATASET_CACHE[limit] = dataset
    return dataset


def available_cpu_ram_gib(reserve_gib: int = 8) -> int:
    try:
        meminfo = {}
        with Path("/proc/meminfo").open("r") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                meminfo[key] = value.strip()
        available_kib = int(meminfo.get("MemAvailable", "0 kB").split()[0])
        available_gib = max(8, int(available_kib / (1024 * 1024)) - reserve_gib)
        return available_gib
    except Exception:
        return 64


def build_max_memory(cpu_ram_gib: int | None = None):
    if not torch.cuda.is_available():
        return None

    total_vram_gib = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
    gpu_budget_gib = max(4, int(total_vram_gib * 0.92))
    cpu_budget_gib = cpu_ram_gib or available_cpu_ram_gib()
    return {0: f"{gpu_budget_gib}GiB", "cpu": f"{cpu_budget_gib}GiB"}


def reset_offload_dir(offload_dir: Path):
    shutil.rmtree(offload_dir, ignore_errors=True)
    offload_dir.mkdir(parents=True, exist_ok=True)


def load_trainable_adapter(
    model_key: str,
    adapter_path: Path,
    offload_dir: Path,
):
    model_name = resolve_model_name(model_key)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    reset_offload_dir(offload_dir)
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "quantization_config": quant_config,
        "attn_implementation": "eager",
        "offload_folder": str(offload_dir),
        "offload_state_dict": True,
        "low_cpu_mem_usage": True,
    }

    max_memory = build_max_memory()
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model, tokenizer


def write_timing(output_path: Path, start_time: float):
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "timing.json").open("w") as handle:
        json.dump({"duration_seconds": time.time() - start_time}, handle)


def cleanup(model=None, trainer=None):
    if trainer is not None:
        del trainer
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)


def build_dpo_config_kwargs(
    output_dir: Path,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    max_length: int,
    max_prompt_length: int,
):
    signature = inspect.signature(DPOConfig.__init__)
    supported = set(signature.parameters.keys())
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": 1,
        "learning_rate": learning_rate,
        "bf16": torch.cuda.is_available(),
        "max_length": max_length,
        "optim": "paged_adamw_8bit",
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "logging_steps": 5,
    }
    if "max_prompt_length" in supported:
        kwargs["max_prompt_length"] = max_prompt_length
    if "truncation_mode" in supported:
        kwargs["truncation_mode"] = "keep_start"
    if "activation_offloading" in supported:
        kwargs["activation_offloading"] = False
    return kwargs


def is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text and "memory" in text


def rank_memory_profile(model_key: str, rank: int, max_length_override: int | None, max_prompt_length_override: int | None):
    spec = MODEL_SPECS[model_key]
    default_max_length = max_length_override or spec["max_length"]
    default_max_prompt_length = max_prompt_length_override or spec["max_prompt_length"]

    if rank >= 3072:
        return min(default_max_length, 320), min(default_max_prompt_length, 160)
    if rank >= 1024:
        return min(default_max_length, 384), min(default_max_prompt_length, 192)
    if rank >= 128:
        return min(default_max_length, 512), min(default_max_prompt_length, 256)
    return default_max_length, default_max_prompt_length


def run_dpo(
    model_key: str,
    rank: int,
    merged_adapter_path: Path,
    output_dir: Path,
    offload_dir: Path,
    dataset_limit: int,
    effective_batch_size: int,
    learning_rate: float,
    data_cache_dir: Path,
    batch_size_override: int | None = None,
    max_length_override: int | None = None,
    max_prompt_length_override: int | None = None,
):
    spec = MODEL_SPECS[model_key]
    exp_name = experiment_name(model_key, rank)
    exp_output = output_dir / exp_name

    default_batch_size = batch_size_override or spec["batch_size"]
    default_max_length, default_max_prompt_length = rank_memory_profile(
        model_key=model_key,
        rank=rank,
        max_length_override=max_length_override,
        max_prompt_length_override=max_prompt_length_override,
    )
    attempt_plan = [
        (default_batch_size, default_max_length, default_max_prompt_length),
        (max(1, default_batch_size // 2), default_max_length, default_max_prompt_length),
        (1, min(default_max_length, 384), min(default_max_prompt_length, 192)),
        (1, min(default_max_length, 256), min(default_max_prompt_length, 128)),
    ]

    seen_attempts = set()
    attempts = [cfg for cfg in attempt_plan if not (cfg in seen_attempts or seen_attempts.add(cfg))]
    dataset = get_dpo_dataset(dataset_limit, cache_dir=data_cache_dir)

    for batch_size, max_length, max_prompt_length in attempts:
        print(
            f"\n🚀 DPO RUN: {exp_name} | rank={rank} | bs={batch_size} | "
            f"max_length={max_length} | max_prompt_length={max_prompt_length}",
            flush=True,
        )
        model = None
        trainer = None
        start_time = time.time()
        try:
            model, tokenizer = load_trainable_adapter(
                model_key=model_key,
                adapter_path=merged_adapter_path,
                offload_dir=offload_dir / exp_name,
            )
            grad_accum = max(1, effective_batch_size // batch_size)
            train_args = DPOConfig(
                **build_dpo_config_kwargs(
                    output_dir=exp_output,
                    batch_size=batch_size,
                    grad_accum=grad_accum,
                    learning_rate=learning_rate,
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                )
            )
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=train_args,
                train_dataset=dataset,
                processing_class=tokenizer,
                callbacks=[ProgressCallback(exp_name)],
            )
            trainer.train()
            model.save_pretrained(str(exp_output), save_embedding_layers=False)
            tokenizer.save_pretrained(str(exp_output))
            write_timing(exp_output, start_time)
            print(f"✅ FINISHED: {exp_name}", flush=True)
            cleanup(model=model, trainer=trainer)
            return True
        except Exception as exc:
            if is_oom_error(exc):
                print(f"⚠️ OOM: {exp_name} failed at bs={batch_size}, max_length={max_length}", flush=True)
                cleanup(model=model, trainer=trainer)
                continue
            print(f"❌ ERROR [{exp_name}]: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            cleanup(model=model, trainer=trainer)
            return False

    print(f"❌ FAILED AFTER ALL FALLBACKS: {exp_name}", flush=True)
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO adapters with CPU offload-aware k-bit loading.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offload_dir", type=Path, default=DEFAULT_OFFLOAD_DIR)
    parser.add_argument("--data_cache_dir", type=Path, default=DEFAULT_DATA_CACHE_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()), choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--ranks", nargs="+", type=int, default=RANKS)
    parser.add_argument("--dataset_limit", type=int, default=10000)
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--list_targets", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    args.data_cache_dir.mkdir(parents=True, exist_ok=True)

    targets = build_dpo_targets(args.output_dir, args.models, args.ranks)
    if not targets:
        print("No DPO targets found. Missing merged DARE adapters for the requested models/ranks.", flush=True)
        return 2

    print("DPO targets discovered from merged adapters:", flush=True)
    for model_key, rank, merged_path in targets:
        print(f" - {experiment_name(model_key, rank)} <= {merged_path}", flush=True)

    if args.list_targets:
        return 0

    all_ok = True
    for model_key, rank, merged_path in targets:
        exp_output = args.output_dir / experiment_name(model_key, rank)
        if (exp_output / "adapter_config.json").exists():
            print(f"⏩ SKIP: {exp_output.name} already completed", flush=True)
            continue

        ok = run_dpo(
            model_key=model_key,
            rank=rank,
            merged_adapter_path=merged_path,
            output_dir=args.output_dir,
            offload_dir=args.offload_dir,
            dataset_limit=args.dataset_limit,
            effective_batch_size=args.effective_batch_size,
            learning_rate=args.learning_rate,
            data_cache_dir=args.data_cache_dir,
            batch_size_override=args.batch_size,
            max_length_override=args.max_length,
            max_prompt_length_override=args.max_prompt_length,
        )
        all_ok = all_ok and ok

    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
