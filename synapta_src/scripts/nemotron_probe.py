"""
Probe Nemotron architecture and runtime readiness.

This script is deliberately two-stage:
  1. A fast source/config scan that works even when CUDA is unavailable.
  2. An optional runtime model load to confirm module names, memory use, and generation.

Usage:
    .venv/bin/python scripts/nemotron_probe.py
    .venv/bin/python scripts/nemotron_probe.py --force-load
"""
import argparse
import json
import re
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
MODEL_PATH = PROJECT_ROOT / "models" / "nemotron"
SOURCE_FILE = MODEL_PATH / "modeling_nemotron_h.py"
CONFIG_FILE = MODEL_PATH / "config.json"


def source_scan_linear_modules(source_text: str) -> dict:
    """Recover candidate linear targets from the shipped model source."""
    linear_modules = {}
    current_class = None

    for line in source_text.splitlines():
        class_match = re.match(r"class\s+([A-Za-z0-9_]+)\(", line)
        if class_match:
            current_class = class_match.group(1)
            continue

        linear_match = re.search(r"self\.(\w+)\s*=\s*nn\.Linear\(", line)
        if linear_match:
            leaf_name = linear_match.group(1)
            linear_modules.setdefault(leaf_name, []).append(
                {
                    "class_name": current_class,
                    "source_line": line.strip(),
                }
            )

    return linear_modules


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Probe Nemotron architecture and runtime")
    parser.add_argument("--force-load", action="store_true", help="Attempt full model load even when CUDA is unavailable")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    print("Reading config and source...")
    config = json.loads(CONFIG_FILE.read_text())
    source_text = SOURCE_FILE.read_text()
    source_modules = source_scan_linear_modules(source_text)

    source_results = {
        "model_path": str(MODEL_PATH),
        "config_summary": {
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "n_routed_experts": config.get("n_routed_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "mamba_num_heads": config.get("mamba_num_heads"),
            "mamba_head_dim": config.get("mamba_head_dim"),
        },
        "target_module_hypotheses": source_modules,
        "safe_target_sets": {
            "attention_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mamba_only": ["in_proj", "out_proj"],
            "shared_expert_only_requires_custom_filtering": ["up_proj", "down_proj"],
            "hybrid_safe_requires_custom_filtering": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "notes": [
            "Source scan is architecture-backed but not a substitute for a successful runtime probe.",
            "Using PEFT target_modules=['up_proj', 'down_proj'] will hit ALL MLP-like modules, not just shared experts.",
            "Nemotron's internal router weight is a Parameter in NemotronHTopkRouter, not an nn.Linear leaf.",
        ],
    }
    save_json(MODEL_PATH / "module_map_source_scan.json", source_results)

    print("\n=== SOURCE SCAN SUMMARY ===")
    print(f"Model arch: {config.get('architectures')}")
    print(f"Hidden size: {config.get('hidden_size')}")
    print(f"Num layers: {config.get('num_hidden_layers')}")
    for leaf_name, entries in sorted(source_modules.items()):
        class_names = sorted({entry['class_name'] for entry in entries if entry.get('class_name')})
        print(f"  {leaf_name}: classes={class_names}")

    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if not cuda_available and not args.force_load:
        print("Skipping runtime model load because CUDA is unavailable.")
        print(f"Saved source-backed probe results to {MODEL_PATH / 'module_map_source_scan.json'}")
        return

    runtime_results = {
        "runtime_probe_attempted": True,
        "cuda_available": cuda_available,
    }

    try:
        print("\nLoading model in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        print(f"\n{'=' * 80}")
        print(f"Model class: {model.__class__.__name__}")
        print(f"Config arch: {model.config.architectures}")
        print(f"Hidden size: {model.config.hidden_size}")
        print(f"Num layers: {model.config.num_hidden_layers}")
        print(f"Vocab size: {model.config.vocab_size}")
        print(f"{'=' * 80}\n")

        linear_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                key = name.split(".")[-1]
                linear_modules.setdefault(key, []).append(
                    {
                        "full_name": name,
                        "in": module.in_features,
                        "out": module.out_features,
                    }
                )

        print("=== RUNTIME LINEAR MODULE TYPES ===")
        for key, instances in sorted(linear_modules.items()):
            inst = instances[0]
            print(f"\n{key}: {len(instances)} instances")
            print(f"  Example: {inst['full_name']}")
            print(f"  Shape: ({inst['in']}, {inst['out']})")

        save_json(MODEL_PATH / "module_map.json", linear_modules)
        runtime_results["linear_modules"] = linear_modules

        print("\n=== GENERATION TEST ===")
        inputs = tokenizer(
            "Solve: 2x + 5 = 13. What is x?",
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded)
        runtime_results["generation_sample"] = decoded

        if torch.cuda.is_available():
            runtime_results["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9
            print(f"\nGPU Memory: {runtime_results['gpu_memory_gb']:.1f} GB allocated")

        runtime_results["status"] = "ok"
    except Exception as exc:
        runtime_results["status"] = "failed"
        runtime_results["error"] = repr(exc)
        runtime_results["traceback"] = traceback.format_exc()
        print("\n=== RUNTIME PROBE FAILED ===")
        print(traceback.format_exc())
    finally:
        save_json(MODEL_PATH / "probe_results.json", runtime_results)
        print(f"\nSaved runtime probe metadata to {MODEL_PATH / 'probe_results.json'}")


if __name__ == "__main__":
    main()
