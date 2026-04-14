"""
Download the base model upfront so it's cached.
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    print(f"Downloading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print(f"Downloading model {args.model} (this may take a while)...")
    # Using device_map CPU just to download and cache weights without taking GPU memory yet
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="cpu", 
        trust_remote_code=True
    )
    print("Download complete and cached!")

if __name__ == "__main__":
    main()
