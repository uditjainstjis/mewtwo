from __future__ import annotations

import json
import shutil
from pathlib import Path


MEWTWO_ROOT = Path("/Users/uditjain/Desktop/mewtwo")
STAGING_ROOT = Path("/Users/uditjain/Desktop/adapter/hf_publish")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
USERNAME = "uditjain"

DOMAINS = {
    "math": {
        "title": "LoRI Math Adapter for Qwen2.5-1.5B-Instruct",
        "repo_name": "lori-qwen2.5-1.5b-math",
        "tag": "math",
        "datasets": ["meta-math/MetaMathQA"],
        "summary": "Domain adapter for mathematical problem solving and step-by-step quantitative reasoning.",
        "intended_use": "Math reasoning, symbolic manipulation, and instruct-style quantitative assistance on top of the Qwen2.5-1.5B-Instruct base model.",
        "safety": "This adapter can still make arithmetic, symbolic, or reasoning mistakes. It should not be treated as a proof system.",
    },
    "code": {
        "title": "LoRI Code Adapter for Qwen2.5-1.5B-Instruct",
        "repo_name": "lori-qwen2.5-1.5b-code",
        "tag": "code",
        "datasets": ["sahil2801/CodeAlpaca-20k", "iamtarun/python_code_instructions_18k_alpaca (fallback)"],
        "summary": "Domain adapter for programming assistance, code generation, and implementation-oriented instructions.",
        "intended_use": "Programming help, implementation sketches, and code-focused chat on top of the Qwen2.5-1.5B-Instruct base model.",
        "safety": "Generated code may be insecure, incomplete, or incorrect and requires review before production use.",
    },
    "science": {
        "title": "LoRI Science Adapter for Qwen2.5-1.5B-Instruct",
        "repo_name": "lori-qwen2.5-1.5b-science",
        "tag": "science",
        "datasets": ["allenai/sciq"],
        "summary": "Domain adapter for science question answering and structured scientific explanations.",
        "intended_use": "Science tutoring, factual explanation, and principle-based question answering on top of Qwen2.5-1.5B-Instruct.",
        "safety": "Scientific explanations may omit caveats or oversimplify domain nuances and should not be treated as authoritative without checking sources.",
    },
    "legal": {
        "title": "LoRI Legal Adapter for Qwen2.5-1.5B-Instruct",
        "repo_name": "lori-qwen2.5-1.5b-legal",
        "tag": "legal",
        "datasets": ["nguha/legalbench: contract_nli_explicit_identification", "nguha/legalbench: learned_hands_benefits (fallback)"],
        "summary": "Domain adapter for legal-style text analysis and contract / policy oriented reasoning prompts.",
        "intended_use": "Legal-text analysis research, clause classification style prompts, and domain-steering experiments.",
        "safety": "This is not legal advice. The training corpus is small in the final prepared dataset, so robustness is limited and outputs must be reviewed by qualified professionals.",
    },
    "medical": {
        "title": "LoRI Medical Adapter for Qwen2.5-1.5B-Instruct",
        "repo_name": "lori-qwen2.5-1.5b-medical",
        "tag": "medical",
        "datasets": [
            "bigbio/med_qa (med_qa_en_4options_source)",
            "GBaker/MedQA-USMLE-4-options (fallback)",
            "openlifescienceai/medmcqa (fallback)",
        ],
        "summary": "Domain adapter for medical multiple-choice style reasoning and medical QA prompts.",
        "intended_use": "Medical QA research and domain adaptation experiments on top of Qwen2.5-1.5B-Instruct.",
        "safety": "This is not medical advice. Outputs can be unsafe, incomplete, or outdated and require expert review.",
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def render_card(domain: str, dataset_stats: dict, training_log: dict, orthogonality_summary: str) -> str:
    meta = DOMAINS[domain]
    repo_id = f"{USERNAME}/{meta['repo_name']}"
    count = dataset_stats[domain]["count"]
    avg_length = dataset_stats[domain]["avg_length"]
    best_loss = training_log["best_loss"]
    total_time = training_log["total_time_min"]
    optimizer_backend = training_log["config"]["optimizer_backend"]

    datasets_yaml = "\n".join(f"- {d}" for d in meta["datasets"])

    return f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
model_name: {repo_id}
tags:
- peft
- lora
- adapter
- qwen2
- text-generation
- domain-adaptation
- research
- {meta["tag"]}
datasets:
{datasets_yaml}
---

# {meta["title"]}

## Model Description

This repository contains a PEFT LoRA adapter trained by `uditjain` as part of the `mewtwo` research line on low-rank expert factorization and multi-domain adapter composition.

The adapter was trained on top of `{BASE_MODEL}` using a custom LoRI-style procedure:

- inject standard LoRA modules into the base model
- replace the LoRA down-projection with a shared frozen random projection
- keep the corresponding up-projection trainable as the domain-specific factor
- apply post-training DARE-style sparsification

This release is the domain adapter artifact only. It is **not** a standalone model.

## What Is In This Repo

- `adapter_model.safetensors`: adapter weights
- `adapter_config.json`: PEFT adapter configuration
- `artifacts/training_state.json`: saved training configuration/state snapshot
- `artifacts/training_log.json`: summarized training log

## Base Model

- Base model: `{BASE_MODEL}`
- Base model license: `apache-2.0`
- PEFT type: `LORA`
- Rank: `32`
- Alpha: `64`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Intended Use

{meta["intended_use"]}

## Training Data

This adapter was trained from the prepared domain corpus in `mewtwo/data/lori_moe`.

- Final prepared example count: `{count}`
- Average tokenized sequence length statistic: `{avg_length:.1f}`
- Source datasets / preparation path:
{chr(10).join(f"  - {d}" for d in meta["datasets"])}

The training pipeline that prepared these datasets is documented in the `mewtwo` repository under `src/lori_moe/data/prepare_datasets.py`.

## Training Procedure

- Training method: LoRI-style PEFT LoRA adaptation with shared frozen random projection and trainable domain-specific factor
- Post-processing: DARE-style sparsification
- Precision: BF16 mixed-precision training
- Optimizer: `{optimizer_backend}`
- Best recorded training loss: `{best_loss:.6f}`
- Recorded training time: `{total_time:.2f} minutes`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "{BASE_MODEL}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

## Research Context

This adapter belongs to a 5-domain family of LoRI-trained experts (`math`, `code`, `science`, `legal`, `medical`) explored in the `mewtwo` project as a successor direction to the earlier Synapta prompt-level composition work.

One family-level empirical result from the saved adapter weights is very low cross-domain overlap for the final `Qwen/Qwen2.5-1.5B-Instruct` expert set:

- {orthogonality_summary}

This supports the geometric motivation for the method, but it does **not** by itself prove superior end-to-end reasoning performance.

## Limitations

- This release is a domain-steering adapter, not a full validated routed MoE system.
- The surrounding router and end-to-end composition stack remain a separate research layer.
- Domain specialization does not guarantee factual correctness.
- {meta["safety"]}
- Legal and medical use cases require especially careful human oversight.

## Open Source Notes

This release is intended to make the trained adapter artifact reproducible and reusable by others working on:

- low-rank expert factorization
- adapter composition
- domain adaptation on small language models
- PEFT-based research baselines

If you build on this work, please cite the future paper or link back to the original `mewtwo` research repository when available.
"""


def main() -> None:
    dataset_stats = load_json(MEWTWO_ROOT / "data/lori_moe/dataset_stats.json")
    orthogonality_summary = "average absolute cross-domain cosine similarity was measured at approximately 0.00685 across the 5 published Qwen2.5-1.5B expert adapters."

    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    source_root = MEWTWO_ROOT / "adapters/lori_moe/qwen2.5_1.5b"

    for domain in DOMAINS:
        repo_dir = STAGING_ROOT / DOMAINS[domain]["repo_name"]
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "artifacts").mkdir(exist_ok=True)

        src = source_root / domain / "dare_sparsified"
        training_log_path = source_root / domain / "training_log.json"
        training_state_path = src / "training_state.json"
        training_log = load_json(training_log_path)

        shutil.copy2(src / "adapter_model.safetensors", repo_dir / "adapter_model.safetensors")
        shutil.copy2(src / "adapter_config.json", repo_dir / "adapter_config.json")
        shutil.copy2(training_state_path, repo_dir / "artifacts" / "training_state.json")
        shutil.copy2(training_log_path, repo_dir / "artifacts" / "training_log.json")

        readme = render_card(domain, dataset_stats, training_log, orthogonality_summary)
        (repo_dir / "README.md").write_text(readme)

    family_summary = {
        "base_model": BASE_MODEL,
        "published_domains": list(DOMAINS.keys()),
        "staging_root": str(STAGING_ROOT),
    }
    (STAGING_ROOT / "family_summary.json").write_text(json.dumps(family_summary, indent=2))


if __name__ == "__main__":
    main()
