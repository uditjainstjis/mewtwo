---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: apache-2.0
model_name: uditjain/lori-qwen2.5-1.5b-math
tags:
- peft
- lora
- adapter
- qwen2
- text-generation
- domain-adaptation
- research
- math
datasets:
- meta-math/MetaMathQA
---

# LoRI Math Adapter for Qwen2.5-1.5B-Instruct

## Model Description

This repository contains a PEFT LoRA adapter trained by `uditjain` as part of the `mewtwo` research line on low-rank expert factorization and multi-domain adapter composition.

The adapter was trained on top of `Qwen/Qwen2.5-1.5B-Instruct` using a custom LoRI-style procedure:

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

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model license: `apache-2.0`
- PEFT type: `LORA`
- Rank: `32`
- Alpha: `64`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Intended Use

Math reasoning, symbolic manipulation, and instruct-style quantitative assistance on top of the Qwen2.5-1.5B-Instruct base model.

## Training Data

This adapter was trained from the prepared domain corpus in `mewtwo/data/lori_moe`.

- Final prepared example count: `49999`
- Average tokenized sequence length statistic: `927.1`
- Source datasets / preparation path:
  - meta-math/MetaMathQA

The training pipeline that prepared these datasets is documented in the `mewtwo` repository under `src/lori_moe/data/prepare_datasets.py`.

## Training Procedure

- Training method: LoRI-style PEFT LoRA adaptation with shared frozen random projection and trainable domain-specific factor
- Post-processing: DARE-style sparsification
- Precision: BF16 mixed-precision training
- Optimizer: `bnb_paged_adamw_8bit`
- Best recorded training loss: `0.128739`
- Recorded training time: `31.98 minutes`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "uditjain/lori-qwen2.5-1.5b-math"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

## Research Context

This adapter belongs to a 5-domain family of LoRI-trained experts (`math`, `code`, `science`, `legal`, `medical`) explored in the `mewtwo` project as a successor direction to the earlier Synapta prompt-level composition work.

One family-level empirical result from the saved adapter weights is very low cross-domain overlap for the final `Qwen/Qwen2.5-1.5B-Instruct` expert set:

- average absolute cross-domain cosine similarity was measured at approximately 0.00685 across the 5 published Qwen2.5-1.5B expert adapters.

This supports the geometric motivation for the method, but it does **not** by itself prove superior end-to-end reasoning performance.

## Limitations

- This release is a domain-steering adapter, not a full validated routed MoE system.
- The surrounding router and end-to-end composition stack remain a separate research layer.
- Domain specialization does not guarantee factual correctness.
- This adapter can still make arithmetic, symbolic, or reasoning mistakes. It should not be treated as a proof system.
- Legal and medical use cases require especially careful human oversight.

## Open Source Notes

This release is intended to make the trained adapter artifact reproducible and reusable by others working on:

- low-rank expert factorization
- adapter composition
- domain adaptation on small language models
- PEFT-based research baselines

If you build on this work, please cite the future paper or link back to the original `mewtwo` research repository when available.
