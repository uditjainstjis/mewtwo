import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

text_intro = """\
# Qwen 0.8B / Nemotron 4B DPO Recovery Notebook
This notebook is the open-source version of the `hf_kaggle_opensource` DPO stage.

It assumes the merged multi-domain adapters already exist, then trains the final preference adapters:

- `qwen_0.8b_math_DPO_rank{1,2,8,128,1024,3072}`
- `nemotron_4b_math_DPO_rank{1,2,8,128,1024,3072}`

The `math_DPO` suffix is legacy naming for compatibility with the local pipeline and output tree.
"""

code_setup = """\
!pip install -q transformers peft trl datasets bitsandbytes accelerate
"""

code_imports = """\
import gc
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import DPOConfig, DPOTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")
"""

code_config = """\
OUTPUT_DIR = Path("/kaggle/working/outputs")
OFFLOAD_DIR = Path("/kaggle/working/offload_cache")
RANKS = [1, 2, 8, 128, 1024, 3072]
DPO_STAGE_LABEL = "math"
DATASET_LIMIT = 10000
EFFECTIVE_BATCH_SIZE = 16

MODEL_SPECS = {
    "qwen_0.8b": {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "batch_size": 2,
        "max_length": 640,
        "max_prompt_length": 384,
    },
    "nemotron_4b": {
        "model_id": "nvidia/Nemotron-Mini-4B-Instruct",
        "batch_size": 1,
        "max_length": 512,
        "max_prompt_length": 256,
    },
}
"""

code_training = """\
class ProgressCallback(TrainerCallback):
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f">>> [{self.experiment_name}] step={state.global_step} loss={logs.get('loss', 'N/A')}", flush=True)


def experiment_name(model_key, rank):
    return f"{model_key}_{DPO_STAGE_LABEL}_DPO_rank{rank}"


def resolve_merged_adapter_path(model_key, rank):
    merged_root = OUTPUT_DIR / f"{model_key}_merged_DARE_rank{rank}"
    if (merged_root / "adapter_config.json").exists():
        return merged_root
    if (merged_root / "merged_sft" / "adapter_config.json").exists():
        return merged_root / "merged_sft"
    return None


def build_targets():
    targets = []
    for model_key in MODEL_SPECS:
        for rank in RANKS:
            merged_path = resolve_merged_adapter_path(model_key, rank)
            if merged_path is not None:
                targets.append((model_key, rank, merged_path))
    return targets


def get_dpo_dataset(limit=DATASET_LIMIT):
    raw_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=f"train_prefs[:{limit}]")

    def format_dpo(example):
        chosen = example["chosen"]
        rejected = example["rejected"]
        return {
            "prompt": example["prompt"],
            "chosen": chosen[1]["content"] if isinstance(chosen, list) else chosen,
            "rejected": rejected[1]["content"] if isinstance(rejected, list) else rejected,
        }

    return raw_ds.map(format_dpo, remove_columns=raw_ds.column_names)


def build_max_memory(cpu_ram_gib=64):
    if not torch.cuda.is_available():
        return None
    total_vram_gib = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
    gpu_budget_gib = max(4, int(total_vram_gib * 0.82))
    return {0: f"{gpu_budget_gib}GiB", "cpu": f"{cpu_ram_gib}GiB"}


def load_trainable_adapter(model_key, adapter_path, offload_dir):
    spec = MODEL_SPECS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

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

    base_model = AutoModelForCausalLM.from_pretrained(spec["model_id"], **load_kwargs)
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


def cleanup(model=None, trainer=None):
    if trainer is not None:
        del trainer
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)


def run_dpo(model_key, rank, merged_path, dataset):
    spec = MODEL_SPECS[model_key]
    exp_name = experiment_name(model_key, rank)
    exp_output = OUTPUT_DIR / exp_name
    attempts = [
        (spec["batch_size"], spec["max_length"], spec["max_prompt_length"]),
        (max(1, spec["batch_size"] // 2), spec["max_length"], spec["max_prompt_length"]),
        (1, min(spec["max_length"], 512), min(spec["max_prompt_length"], 256)),
        (1, min(spec["max_length"], 384), min(spec["max_prompt_length"], 192)),
    ]

    seen = set()
    attempts = [cfg for cfg in attempts if not (cfg in seen or seen.add(cfg))]

    for batch_size, max_length, max_prompt_length in attempts:
        print(
            f"\\n🚀 DPO RUN: {exp_name} | bs={batch_size} | "
            f"max_length={max_length} | max_prompt_length={max_prompt_length}"
        )
        model = None
        trainer = None
        try:
            model, tokenizer = load_trainable_adapter(model_key, merged_path, OFFLOAD_DIR / exp_name)
            grad_accum = max(1, EFFECTIVE_BATCH_SIZE // batch_size)
            args = DPOConfig(
                output_dir=str(exp_output),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                num_train_epochs=1,
                learning_rate=5e-5,
                bf16=torch.cuda.is_available(),
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                optim="paged_adamw_8bit",
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False,
                gradient_checkpointing=True,
                logging_steps=5,
            )
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=args,
                train_dataset=dataset,
                processing_class=tokenizer,
                callbacks=[ProgressCallback(exp_name)],
            )
            trainer.train()
            trainer.save_model(str(exp_output))
            tokenizer.save_pretrained(str(exp_output))
            print(f"✅ FINISHED: {exp_name}")
            cleanup(model=model, trainer=trainer)
            return True
        except torch.OutOfMemoryError:
            print(f"⚠️ OOM: {exp_name} failed at bs={batch_size}, max_length={max_length}")
            cleanup(model=model, trainer=trainer)
            continue
        except Exception as exc:
            print(f"❌ ERROR [{exp_name}]: {exc}")
            cleanup(model=model, trainer=trainer)
            return False

    print(f"❌ FAILED AFTER ALL FALLBACKS: {exp_name}")
    return False
"""

code_exec = """\
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

targets = build_targets()
print("Discovered DPO targets:")
for model_key, rank, merged_path in targets:
    print(f" - {experiment_name(model_key, rank)} <= {merged_path}")

dataset = get_dpo_dataset()

for model_key, rank, merged_path in targets:
    exp_output = OUTPUT_DIR / experiment_name(model_key, rank)
    if (exp_output / "adapter_config.json").exists():
        print(f"⏩ SKIP: {exp_output.name} already complete")
        continue
    run_dpo(model_key, rank, merged_path, dataset)
"""

nb["cells"] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_config),
    nbf.v4.new_code_cell(code_training),
    nbf.v4.new_code_cell(code_exec),
]

output_path = Path("/home/learner/Desktop/mewtwo/hf_kaggle_opensource/Multi_Rank_MoE_Training.ipynb")
with output_path.open("w") as handle:
    nbf.write(nb, handle)

print(f"Notebook successfully written to {output_path}")
