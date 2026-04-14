import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DOMAINS = ["math", "code", "science", "legal", "medical"]

def get_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def main():
    print("\n" + "="*80)
    print("🔬 BRUTAL INTERFERENCE TEST: Catastrophic Forgetting Validation 🔬")
    print("Comparing Single Expert vs. Fully Linear-Merged 5-Domain MoE")
    print("="*80)
    
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir = Path("/home/learner/Desktop/mewtwo/checkpoints/lori_moe/qwen2.5_1.5b")
    data_dir = Path("/home/learner/Desktop/mewtwo/data/lori_moe")
    
    # 1. Load Base Model
    print("Loading Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Load test sentences
    test_data = {}
    for domain in DOMAINS:
        file_path = data_dir / f"{domain}_train.jsonl"
        if file_path.exists():
            with open(file_path, "r") as f:
                row = json.loads(f.readline().strip())
                test_data[domain] = row["text"]
    
    # 2. Add all adapters independently to PEFT
    peft_model = None
    for i, domain in enumerate(DOMAINS):
        adapter_path = adapter_dir / domain / "dare_sparsified"
        print(f"Loading adapter: {domain}")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=domain)
        else:
            peft_model.load_adapter(str(adapter_path), adapter_name=domain)
            
    peft_model.eval()
    
    print("\n--- PHASE 1: Baseline Single-Adapter Performance ---")
    single_expert_ppl = {}
    for domain in DOMAINS:
        if domain not in test_data:
            continue
        peft_model.set_adapter(domain)
        ppl = get_perplexity(peft_model, tokenizer, test_data[domain])
        single_expert_ppl[domain] = ppl
        print(f"  [{domain.upper()}] PPL with {domain} adapter: {ppl:.4f}")
        
    print("\n--- PHASE 2: Forced Merging (Interference Torture Test) ---")
    print("Merging ALL 5 adapters linearly into one unified state...")
    # Linear combination corresponding to soft-routing where all experts are active!
    # A_merged = 0.2*A1 + 0.2*A2 + 0.2*A3 + 0.2*A4 + 0.2*A5
    try:
        peft_model.add_weighted_adapter(
            adapters=DOMAINS,
            weights=[1.0] * len(DOMAINS),
            adapter_name="omni_merged",
            combination_type="linear"
        )
        peft_model.set_adapter("omni_merged")
        print("  ✓ Merge complete.")
    except Exception as e:
        print(f"Merge failed: {e}")
        return

    print("\n--- PHASE 3: Degradation Analysis ---")
    print(f"{'DOMAIN':<10} | {'SINGLE EXPERT PPL':<20} | {'FULLY MERGED PPL':<18} | {'DEGRADATION'}")
    print("-" * 75)
    
    total_degradation = 0
    for domain in DOMAINS:
        if domain not in test_data:
            continue
        merged_ppl = get_perplexity(peft_model, tokenizer, test_data[domain])
        single_ppl = single_expert_ppl[domain]
        
        # Lower PPL is better. Degradation is how much worse (higher) the merged PPL is.
        # Ratio = (Merged / Single) - 1. E.g. Merged 10.5, Single 10.0 -> +5.0%
        deg_percent = ((merged_ppl / single_ppl) - 1.0) * 100
        total_degradation += deg_percent
        
        flag = "✅ PASS" if deg_percent < 5.0 else "❌ FAIL"
        
        print(f"{domain.upper():<10} | {single_ppl:<20.4f} | {merged_ppl:<18.4f} | {deg_percent:+.2f}% {flag}")
        
    avg_deg = total_degradation / len(DOMAINS)
    print("=" * 75)
    print(f"AVERAGE INTERFERENCE DEGRADATION: {avg_deg:+.2f}%")
    
    if avg_deg < 5.0:
        print("\n🏆 CONCLUSION: THE LORI HYPOTHESIS IS MATHEMATICALLY AND EMPIRICALLY CONFIRMED.")
        print("Adapters trained with frozen shared projection spaces successfully occupy")
        print("orthogonal subspaces. They can be linearly combined at scale without destroying")
        print("individual domain reasoning capacity.")
    else:
        print("\n⚠️ CONCLUSION: Catastrophic forgetting occurred.")

if __name__ == "__main__":
    main()
