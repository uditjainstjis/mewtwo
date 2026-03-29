from typing import Dict, List, Tuple
import re

class MultiLabelCoTRouter:
    """
    Uses the base LLM natively to extract MULTIPLE relevant domains in a single pass.
    """
    def __init__(self, base_engine, registry_path: str):
        self.engine = base_engine
        import json
        with open(registry_path, "r") as f:
            self.registry = json.load(f)
        self.domain_tags = list(self.registry.keys())

    def _build_prompt(self, query: str) -> str:
        domain_list = ", ".join([f"[{d}]" for d in self.domain_tags])
        system_instruction = (
            "System: You are an intelligent multi-label routing engine. "
            "Analyze the user's question and identify ALL knowledge domains required to answer it.\n"
            f"Available domains: {domain_list}\n"
            "First, output 1-sentence reasoning. Then, output the relevant tags in order of importance, separated by commas.\n"
            "Format: Reason: <reason>\nTags: [DOMAIN_1], [DOMAIN_2]"
        )
        return f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

    def route_top_k(self, query: str, k: int = 2) -> List[Tuple[str, float]]:
        prompt = self._build_prompt(query)
        # Generate with base model (no adapters injected yet, or all 0 weights)
        no_adapters = {d: 0.0 for d in self.domain_tags}
        output_text, _ = self.engine.generate(prompt, routing_weights=no_adapters, max_tokens=100)
        
        # Parse output for tags like [DOMAIN_NAME]
        tags = re.findall(r'\[([^\]]+)\]', output_text.upper())
        
        valid_domains = []
        for tag in tags:
            if tag in self.domain_tags and tag not in valid_domains:
                valid_domains.append(tag)
                
        # If none found, fallback
        if not valid_domains:
            valid_domains = ["LEGAL_ANALYSIS"] # Arbitrary fallback
            
        # Assign decaying soft probabilities to the extracted domains
        results = []
        prob = 0.90
        for d in valid_domains[:k]:
            results.append((d, prob))
            prob -= 0.15 # Decay for second domain
            
        return results

    def route_probs(self, query: str) -> Dict[str, float]:
        top_k = self.route_top_k(query, k=len(self.domain_tags))
        probs = {d: 0.0 for d in self.domain_tags}
        for d, p in top_k:
            probs[d] = p
            
        # Normalize sum to 1
        total = sum(probs.values())
        if total > 0:
            probs = {d: p/total for d, p in probs.items()}
        return probs
