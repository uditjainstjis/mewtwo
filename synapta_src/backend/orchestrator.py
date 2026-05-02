import json

class Orchestrator:
    def __init__(self, registry_path, base_engine):
        self.base_engine = base_engine
        with open(registry_path, "r") as f:
            self.registry = json.load(f)
        self.domains = list(self.registry.keys())
        self.domain_list_str = ", ".join([f"[{d}]" for d in self.domains])
        
    def route(self, query, top_k=1):
        prompt = f"<|im_start|>system\nYou are an intelligent routing engine. Your job is to classify the user's question into the correct domain. First, output a 1-sentence reasoning. Then, output EXACTLY one tag from this list: {self.domain_list_str}.\nExample: This is about curing a disease, so the domain is health. [MEDICAL_DIAGNOSIS]<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\nReasoning:"
        
        # Pre-Inference CoT Reasoning Step natively triggering via Base Model
        response, _ = self.base_engine.generate(prompt, routing_weights=None, max_tokens=30)
        
        route_text = "[" + response.strip().upper()
        matched_domain = None
        for d in self.domains:
            if f"[{d}]" in route_text or d in route_text:
                matched_domain = d
                break
                
        weights = {d: 0.0 for d in self.domains}
        if matched_domain:
            weights[matched_domain] = 1.0 # 0.5 Clamp happens gracefully in the forward propagation
            print(f"CoT Router identified Gating Tag: {matched_domain}")
            return weights, response.strip()
        else:
            fallback = self.domains[0]
            weights[fallback] = 1.0 # Fallback routing
            warning_msg = f"{response.strip()} (⚠️ SYSTEM FALLBACK: Model hallucinated tag. Defaulting to {fallback})"
            print(f"CoT Router FAILED exact match. Defaulting to Fallback: {fallback}")
            return weights, warning_msg

    def route_top2(self, query):
        """Extract top-2 domains via two sequential CoT calls.
        Returns (weights_dict, [domain1, domain2 or None], cot_text).
        """
        # First call: standard top-1
        weights1, cot1 = self.route(query, top_k=1)
        domain1 = None
        for d in self.domains:
            if weights1.get(d, 0.0) > 0.5:
                domain1 = d
                break
        if domain1 is None:
            domain1 = self.domains[0]  # fallback already set

        # Second call: exclude domain1, ask for second domain
        remaining = [d for d in self.domains if d != domain1]
        remaining_str = ", ".join([f"[{d}]" for d in remaining])
        prompt2 = f"<|im_start|>system\nYou are an intelligent routing engine. The question spans multiple domains. The primary domain is [{domain1}]. Now classify into the SECOND most relevant domain from: {remaining_str}.\nOutput EXACTLY one tag.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\nSecond domain:"
        response2, _ = self.base_engine.generate(prompt2, routing_weights=None, max_tokens=30)
        route_text2 = "[" + response2.strip().upper()

        domain2 = None
        for d in remaining:
            if f"[{d}]" in route_text2 or d in route_text2:
                domain2 = d
                break

        # Build weights
        weights = {d: 0.0 for d in self.domains}
        if domain2:
            weights[domain1] = 0.5
            weights[domain2] = 0.5
            print(f"  CoT Top-2: [{domain1}] + [{domain2}]")
            return weights, [domain1, domain2], cot1
        else:
            # Fallback: only got 1 domain
            weights[domain1] = 1.0
            print(f"  CoT Top-2 FAILED second domain. Using [{domain1}] only.")
            return weights, [domain1, None], cot1

