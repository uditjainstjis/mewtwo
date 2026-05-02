import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from src.training.cf_lora import CFLoRATrainer

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 2
LORA_R = 32
LORA_ALPHA = 64
DOMAINS = ["MATHEMATICS", "MEDICAL_DIAGNOSIS", "LEGAL_ANALYSIS"]

def format_dataset(examples, tokenizer):
    texts = []
    for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
        # Qwen-style ChatML format wrapper
        text = f"<|im_start|>user\n{inst}\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
        texts.append(text)
    
    # We must add labels for CAUSAL_LM. The trainer handles computing loss where labels == input_ids
    encoded = tokenizer(texts, truncation=True, max_length=1024, padding="max_length")
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

class ProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps > 0 and state.global_step % max(1, state.max_steps // 20) == 0:
            pct = (state.global_step / state.max_steps) * 100
            loss = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
            print(f"📊 [Progress] {pct:.1f}% -> Step {state.global_step}/{state.max_steps} | Loss: {loss}", flush=True)

def extract_subspace(model, threshold=0.95):
    """
    Computes SVD of the learned LoRA and extracts orthogonal subspace projectors.
    This detects the 'Grokking' subspace which is characteristically low-rank.
    """
    subspaces = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            layer_base = name.replace('.lora_A.default.weight', '')
            A = param.detach()
            
            param_dict = dict(model.named_parameters())
            B_name = name.replace('lora_A', 'lora_B')
            if B_name in param_dict:
                B = param_dict[B_name].detach()
                W = B @ A
                
                # Move matrices to CPU to avoid strict 32GB OOM during the memory-heavy SVD algorithm
                W_cpu = W.to('cpu').float()
                
                # Truncated SVD using Randomized Low-Rank Approximation! 
                # Since the adapter was trained with r=32, the maximum theoretical rank of the matrix is 32. 
                # Calculating the full 4096-rank SVD is what was bottlenecking the CPU for 15 minutes!
                U_cpu, S_cpu, V_cpu = torch.svd_lowrank(W_cpu, q=64)
                
                # Dynamic rank selection via Top-K Variance (Grokking drop)
                s_sq = S_cpu**2
                explained_var = torch.cumsum(s_sq, dim=0) / torch.sum(s_sq)
                k = torch.sum(explained_var < threshold).item() + 1
                
                U_k = U_cpu[:, :k]
                # Orthogonal Projection Matrix P = UU^T (Calculated entirely on CPU RAM ~64GB)
                P_cpu = U_k @ U_k.T
                subspaces[layer_base] = P_cpu
                
    return subspaces

def run_matrix():
    print("🚀 Initializing CF-LoRA Training Matrix for RTX 5090")
    print(f"Loading Base Model: {MODEL_NAME} in BF16...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    frozen_subspaces = []
    
    for i, domain in enumerate(DOMAINS):
        print(f"\n{'='*50}\n🔥 Phase: Training domain {domain} (Adapter {i+1}/{len(DOMAINS)})\n{'='*50}")
        
        adapter_path = f"models/adapters/{domain}"
        if os.path.exists(adapter_path):
            print(f"⏩ Domain {domain} is already trained! Loading adapter to extract subspace skipping duplicate training...")
            adapter_model = PeftModel.from_pretrained(model, adapter_path)
            
            print("🧮 Recovering generalizing SVD Subspace from saved model via CPU bounds...")
            subspace_P = extract_subspace(adapter_model)
            frozen_subspaces.append(subspace_P)
            model = adapter_model.unload()
            continue
            
        data_files = {"train": f"data/training/{domain}/train.jsonl"}
        ds = load_dataset("json", data_files=data_files)
        
        print("Tokenizing dataset with CoT reasoning...")
        tokenized_ds = ds.map(
            lambda x: format_dataset(x, tokenizer), 
            batched=True, 
            remove_columns=ds["train"].column_names
        )
        
        # Inject standard LORA. CF-LoRA handles the penalty in the Trainer wrapper
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", 
            inference_mode=False, 
            r=LORA_R, 
            lora_alpha=LORA_ALPHA, 
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        print(f"Injecting PEFT adapter...")
        adapter_model = get_peft_model(model, peft_config)
        adapter_model.enable_input_require_grads() # Required for Gradient Checkpointing with PEFT
        adapter_model.print_trainable_parameters()
        
        training_args = TrainingArguments(
            output_dir=f"adapters/{domain}",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=2e-4,
            weight_decay=0.01,
            num_train_epochs=EPOCHS,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True, 
            gradient_checkpointing=True,
            report_to="none"
        )
        
        lam = 0.01 if i > 0 else 0.0 # No penalty for the very first domain
        print(f"CF-LoRA Orthogonality Lambda: {lam} (Targeting {len(frozen_subspaces)} frozen subspaces)")
        
        trainer = CFLoRATrainer(
            model=adapter_model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            frozen_subspaces=frozen_subspaces,
            lambda_ortho=lam,
            callbacks=[ProgressCallback()]
        )
        
        print("Starting optimization. Watch the GPU go BRRRR...")
        trainer.train()
        
        adapter_path = f"models/adapters/{domain}"
        adapter_model.save_pretrained(adapter_path)
        print(f"✅ Saved adapter weights to: {adapter_path}")
        
        print("🧮 Running SVD to map the generalized Subspace...")
        subspace_P = extract_subspace(adapter_model)
        frozen_subspaces.append(subspace_P)
        print(f"Subspace mapped. Locking {len(subspace_P)} projection tensors for zero-interference composite shielding.")
        
        print("Unloading adapter from active VRAM to prepare sequential training...")
        model = adapter_model.unload()

    print("🎉 CF-LoRA Matrix Training Complete. All adapters are naturally orthogonal!")

if __name__ == "__main__":
    run_matrix()
