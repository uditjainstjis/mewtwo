"""
LoRI-MoE Routing Dataset Generator

Generates high-entropy multi-domain reasoning traces to train the Token-Level Router.
If the router is only trained on pure single-domain prompts, it will learn to rely on 
prompt-level heuristics and collapse to one expert per sequence. 

This script artificially concatenates examples from highly orthogonal domains 
(e.g., Math + Code, Medical + Legal) to force the MLP to learn how to dynamically 
switch routing weights mid-sequence based on local token context.
"""
import os
import json
import random
import logging
import argparse
from pathlib import Path
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_domain_samples(data_dir: str, domain: str, max_samples: int = 5000):
    """Loads text samples for a specific domain."""
    path = Path(data_dir) / f"{domain}_train.jsonl"
    if not path.exists():
        logger.warning(f"Data not found for domain {domain}")
        return []
        
    samples = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            samples.append(json.loads(line.strip())["text"])
    return samples

def generate_routing_dataset(data_dir: str, output_dir: str, num_mixed_examples: int = 30000):
    """Generates a dataset of mixed-domain sequences."""
    os.makedirs(output_dir, exist_ok=True)
    
    domains = ["math", "code", "science", "legal", "medical"]
    domain_data = {}
    
    for d in domains:
        domain_data[d] = load_domain_samples(data_dir, d, max_samples=10000)
        logger.info(f"Loaded {len(domain_data[d])} samples for {d}")
        
    valid_domains = [d for d in domains if len(domain_data[d]) > 0]
    domain_pairs = list(combinations(valid_domains, 2))
    
    mixed_dataset = []
    
    # 1. Generate cross-domain sequences
    logger.info("Generating synthesized multi-domain context sequences...")
    for _ in range(num_mixed_examples):
        # Pick two random orthogonal domains
        d1, d2 = random.choice(domain_pairs)
        
        # Randomize order
        if random.random() > 0.5:
            d1, d2 = d2, d1
            
        sample1 = random.choice(domain_data[d1])
        sample2 = random.choice(domain_data[d2])
        
        # Construct synthetic transition structures to force token-switching
        transition_templates = [
            f"{sample1}\n\nGiven the context above, consider a related problem:\n\n{sample2}",
            f"{sample1}\n\nMeanwhile, in a totally different system:\n\n{sample2}",
            f"Context A:\n{sample1}\n\nContext B:\n{sample2}",
            f"{sample1}\n\nSimilarly, evaluate the following:\n\n{sample2}"
        ]
        
        mixed_text = random.choice(transition_templates)
        
        mixed_dataset.append({
            "text": mixed_text,
            "domains": [d1, d2],
            "type": "mixed_transition"
        })
        
    # Shuffle the dataset
    random.shuffle(mixed_dataset)
    
    output_path = Path(output_dir) / "routing_mixed_train.jsonl"
    with open(output_path, "w") as f:
        for row in mixed_dataset:
            f.write(json.dumps(row) + "\n")
            
    logger.info(f"Successfully generated {len(mixed_dataset)} multi-domain sequences.")
    logger.info(f"Saved router dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/learner/Desktop/mewtwo/data/lori_moe")
    parser.add_argument("--output_dir", type=str, default="/home/learner/Desktop/mewtwo/data/lori_moe")
    parser.add_argument("--num_examples", type=int, default=30000)
    args = parser.parse_args()
    
    generate_routing_dataset(args.data_dir, args.output_dir, args.num_examples)
