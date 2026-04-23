#!/usr/bin/env python3
"""
Build Kaggle Submission: Merge 3 domain adapters → 1 rank-32 adapter.

Strategy:
1. Load all 3 adapters (math, code, science)
2. For each LoRA module, compute the full weight delta: W = B @ A
3. Average the deltas across all 3 domains
4. SVD-truncate the merged delta to rank 32
5. Save as a valid PEFT adapter with adapter_config.json
6. Zip for submission
"""
import json
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file

PROJECT = Path("/home/learner/Desktop/mewtwo")
ADAPTER_DIR = PROJECT / "checkpoints" / "nemotron_lori" / "adapters"
OUTPUT_DIR = PROJECT / "submission_adapter"
SUBMISSION_ZIP = PROJECT / "submission.zip"

DOMAINS = ["math", "code", "science"]
TARGET_RANK = 32
TARGET_ALPHA = 64.0  # ratio of 2.0 (alpha/rank)

def load_adapter_weights(domain: str) -> dict:
    """Load adapter weights, trying best then dare_sparsified then final."""
    for sub in ["best", "dare_sparsified", "final"]:
        path = ADAPTER_DIR / domain / sub / "adapter_model.safetensors"
        if path.exists():
            print(f"  Loading {domain} from {sub}")
            return load_file(str(path))
    raise FileNotFoundError(f"No adapter found for {domain}")


def get_module_pairs(weights: dict) -> dict:
    """Group lora_A and lora_B weights by module name."""
    modules = defaultdict(dict)
    for key, tensor in weights.items():
        # key like: base_model.model.backbone.layers.12.mixer.q_proj.lora_A.weight
        if "lora_A" in key:
            module_name = key.replace(".lora_A.weight", "")
            modules[module_name]["A"] = tensor
        elif "lora_B" in key:
            module_name = key.replace(".lora_B.weight", "")
            modules[module_name]["B"] = tensor
    return dict(modules)


def merge_and_compress():
    print("=" * 60)
    print("Building Kaggle Submission")
    print(f"  Domains: {DOMAINS}")
    print(f"  Target rank: {TARGET_RANK}")
    print(f"  Target alpha: {TARGET_ALPHA}")
    print("=" * 60)

    # 1. Load all adapters
    print("\n[1/5] Loading adapters...")
    all_modules = {}
    for domain in DOMAINS:
        weights = load_adapter_weights(domain)
        all_modules[domain] = get_module_pairs(weights)
        print(f"  {domain}: {len(all_modules[domain])} LoRA modules")

    # Get the union of all module names
    all_module_names = set()
    for domain in DOMAINS:
        all_module_names.update(all_modules[domain].keys())
    print(f"\n  Total unique modules: {len(all_module_names)}")

    # 2. Merge: average the weight deltas
    print("\n[2/5] Merging adapters (averaging weight deltas)...")
    merged_weights = {}

    for module_name in sorted(all_module_names):
        deltas = []
        for domain in DOMAINS:
            if module_name in all_modules[domain]:
                m = all_modules[domain][module_name]
                A = m["A"].float()  # (r, in_dim)
                B = m["B"].float()  # (out_dim, r)
                delta = B @ A       # (out_dim, in_dim)
                deltas.append(delta)

        if not deltas:
            continue

        # Average the deltas
        merged_delta = torch.stack(deltas).mean(dim=0)

        # 3. SVD truncation to target rank
        U, S, Vh = torch.linalg.svd(merged_delta, full_matrices=False)

        # Keep top-k singular values
        k = min(TARGET_RANK, len(S))
        U_k = U[:, :k]       # (out_dim, k)
        S_k = S[:k]           # (k,)
        Vh_k = Vh[:k, :]      # (k, in_dim)

        # Split into new A and B: B = U_k * sqrt(S_k), A = sqrt(S_k) * Vh_k
        sqrt_S = torch.sqrt(S_k)
        new_B = U_k * sqrt_S.unsqueeze(0)   # (out_dim, k)
        new_A = sqrt_S.unsqueeze(1) * Vh_k  # (k, in_dim)

        # Store in PEFT format
        merged_weights[f"{module_name}.lora_A.weight"] = new_A.to(torch.float32)
        merged_weights[f"{module_name}.lora_B.weight"] = new_B.to(torch.float32)

        # Compute reconstruction quality
        reconstructed = new_B @ new_A
        error = (merged_delta - reconstructed).norm() / merged_delta.norm()
        energy = (S_k ** 2).sum() / (S ** 2).sum()

    print(f"  Merged {len(merged_weights) // 2} modules")
    print(f"  Last module energy retained: {energy:.4f}")

    # 4. Save as PEFT adapter
    print("\n[3/5] Saving merged adapter...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Make all tensors contiguous (required by safetensors)
    merged_weights = {k: v.contiguous() for k, v in merged_weights.items()}

    # Save weights
    save_file(merged_weights, str(OUTPUT_DIR / "adapter_model.safetensors"))

    # Build adapter_config.json
    config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "nvidia/Nemotron-3-Nano-30B-A3B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": TARGET_ALPHA,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": TARGET_RANK,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["v_proj", "o_proj", "q_proj", "k_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    with open(OUTPUT_DIR / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files from one of the existing adapters
    src = ADAPTER_DIR / "math" / "best"
    for fname in ["tokenizer_config.json", "tokenizer.json", "chat_template.jinja"]:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(str(src_file), str(OUTPUT_DIR / fname))

    print(f"  Saved to: {OUTPUT_DIR}")

    # 5. Zip
    print("\n[4/5] Creating submission.zip...")
    if SUBMISSION_ZIP.exists():
        SUBMISSION_ZIP.unlink()

    with zipfile.ZipFile(str(SUBMISSION_ZIP), "w", zipfile.ZIP_DEFLATED) as z:
        for fpath in OUTPUT_DIR.iterdir():
            if fpath.is_file():
                z.write(str(fpath), fpath.name)
                print(f"  + {fpath.name} ({fpath.stat().st_size / 1024:.1f} KB)")

    zip_size = SUBMISSION_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n  submission.zip: {zip_size:.1f} MB")

    # 6. Verify
    print("\n[5/5] Verifying submission...")
    with zipfile.ZipFile(str(SUBMISSION_ZIP), "r") as z:
        names = z.namelist()
        assert "adapter_config.json" in names, "Missing adapter_config.json!"
        assert "adapter_model.safetensors" in names, "Missing adapter_model.safetensors!"

        cfg = json.loads(z.read("adapter_config.json"))
        assert cfg["r"] == TARGET_RANK, f"Rank is {cfg['r']}, expected {TARGET_RANK}!"
        assert cfg["peft_type"] == "LORA", "Not a LoRA adapter!"

    print("\n" + "=" * 60)
    print("✅ Submission ready!")
    print(f"  File: {SUBMISSION_ZIP}")
    print(f"  Size: {zip_size:.1f} MB")
    print(f"  Rank: {TARGET_RANK}")
    print(f"  Alpha: {TARGET_ALPHA}")
    print(f"  Modules: {len(merged_weights) // 2}")
    print(f"  Base model: nvidia/Nemotron-3-Nano-30B-A3B")
    print("=" * 60)


if __name__ == "__main__":
    merge_and_compress()
