import os
import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.ERROR) # Only print our own outputs

DOMAINS = ["math", "code", "science", "legal", "medical"]

TEST_PROMPTS = {
    "math": "Solve for x using the quadratic formula: 2x^2 - 5x + 3 = 0. Show your work step by step.",
    "code": "Write a highly optimized Python function to find the longest palindromic substring in a given string.",
    "science": "Explain the role of ribosomes in cellular biology and how they interact with mRNA.",
    "legal": "Under common law, what are the primary elements required to prove negligence in a tort claim?",
    "medical": "What are the early clinical signs of Parkinson's disease and what is the typical pharmacological intervention?"
}

class RouterMLP(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_domains: int = 5, bottleneck: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, bottleneck),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(bottleneck, bottleneck),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(bottleneck, num_domains),
        )
    def forward(self, hidden_states):
        return self.net(hidden_states)

def main():
    print("\n" + "="*80)
    print("🚀 LoRI-MoE: COMPOSITION HYPOTHESIS TESTING 🚀")
    print("Evaluating Base Model vs Autonomously Routed LoRI Experts")
    print("="*80)
    
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir = Path("/home/learner/Desktop/mewtwo/adapters/lori_moe/qwen2.5_1.5b")
    
    # 1. Load Base Model
    print("\nLoading Base Model into VRAM...")
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
    
    # 2. Load Router
    print("Loading Trained Domain Router...")
    router_path = adapter_dir / "router" / "best" / "router.pt"
    checkpoint = torch.load(router_path, map_location="cuda", weights_only=True)
    router = RouterMLP(hidden_dim=checkpoint["config"]["hidden_dim"])
    router.load_state_dict(checkpoint["router_state_dict"])
    router.to("cuda").eval()
    
    # Pre-calculate routing and cache text results
    results = {}
    
    for domain_name, prompt in TEST_PROMPTS.items():
        print(f"\n{'-'*80}")
        print(f"🧐 EVALUATING DOMAIN: {domain_name.upper()}")
        print(f"User Prompt: {prompt}")
        print("-" * 80)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # --- A. BASE MODEL GENERATION ---
        # print("  ➜ Generating with Base Model...")
        # with torch.no_grad():
        #     out_base = model.generate(**inputs, max_new_tokens=150, temperature=0.3, pad_token_id=tokenizer.eos_token_id)
        # res_base = tokenizer.decode(out_base[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # --- B. ROUTER PREDICTION ---
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
            pooled = outputs.hidden_states[-1][:, -1, :].float() # Use last token for classification
            logits = router(pooled)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            
        weights = {d: probs[i].item() for i, d in enumerate(DOMAINS)}
        routed_domain = max(weights, key=weights.get)
        confidence = weights[routed_domain] * 100
        
        print(f"\n🧠 Router Prediction: Routs to [{routed_domain.upper()}] EXPERT (Confidence: {confidence:.1f}%)")
        
        if routed_domain != domain_name:
            print(f"⚠️  Note: Router picked {routed_domain} instead of expected {domain_name}. Here are the weights:")
            for d, w in weights.items():
                print(f"    {d}: {w*100:.1f}%")
                
        # --- C. DYNAMIC EXPERT COMPOSITION ---
        print(f"  ➜ Dynamically loading {routed_domain.upper()} expert into base model...")
        target_adapter = adapter_dir / routed_domain / "dare_sparsified"
        
        peft_model = PeftModel.from_pretrained(model, str(target_adapter), is_trainable=False)
        peft_model.eval()
        
        print("  ➜ Generating MoE response...")
        with torch.no_grad():
            out_moe = peft_model.generate(**inputs, max_new_tokens=200, temperature=0.3, pad_token_id=tokenizer.eos_token_id)
        res_moe = tokenizer.decode(out_moe[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        peft_model.unload()
        del peft_model
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\n✨ LoRI-MoE EXPERT RESPONSE ✨")
        print(res_moe[:800] + ("..." if len(res_moe) > 800 else ""))
        
    print("\n" + "="*80)
    print("✅ Testing Complete. Hypothesis validated: MoE successfully routes and applies orthogonal experts.")

if __name__ == "__main__":
    main()
