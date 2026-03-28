import json
import uuid
from typing import List, Dict

class PromptBuilder:
    def __init__(self):
        # Synthetic templates for logic instantiation
        self.math_template = "Evaluate the polynomial {equation} over the {field}."
        self.code_template = "Write an optimized {lang} kernel to perform {task} on a {data_struct}."
        self.fincode_template = "Write a modular Python script using `numpy` to calculate {financial_metric} based on {assumption}."

    def build_pure_math(self, n_synthetic: int) -> List[Dict]:
        prompts = []
        for i in range(n_synthetic):
            # Simulated LLM/Template generation logic
            prompts.append({
                "id": f"math_syn_{uuid.uuid4().hex[:8]}",
                "domain": "math",
                "subdomain": "algebra",
                "prompt": self.math_template.format(equation=f"{i}x^2 + 7 = 0", field="complex plane"),
                "expected_format": "boxed_equation",
                "difficulty": 3,
                "tags": ["polynomial", "synthetic"]
            })
        return prompts

    def build_pure_code(self, n_synthetic: int) -> List[Dict]:
        prompts = []
        for i in range(n_synthetic):
            prompts.append({
                "id": f"code_syn_{uuid.uuid4().hex[:8]}",
                "domain": "code",
                "subdomain": "gpu_kernels",
                "prompt": self.code_template.format(lang="Python", task="matrix exponentiation", data_struct="2D torch.Tensor"),
                "expected_format": "code_block",
                "difficulty": 4,
                "tags": ["python", "execution_eval"]
            })
        return prompts

    def build_mixed_fincode(self, n_synthetic: int) -> List[Dict]:
        prompts = []
        for i in range(n_synthetic):
            prompts.append({
                "id": f"mix_fc_{uuid.uuid4().hex[:8]}",
                "domains": ["finance", "code"],
                "prompt": self.fincode_template.format(financial_metric="Black-Scholes implied volatility", assumption="a European Call option with S=100, K=105"),
                "required_capabilities": ["numpy execution", "quantitative pricing"],
                "eval_metric": "pass@1/functionality"
            })
        return prompts

def save_prompts_to_file(path: str, prompts: List[Dict]) -> None:
    is_jsonl = path.endswith(".jsonl")
    with open(path, "w") as f:
        if is_jsonl:
            for p in prompts:
                f.write(json.dumps(p) + "\n")
        else:
            json.dump(prompts, f, indent=2)

if __name__ == "__main__":
    builder = PromptBuilder()
    save_prompts_to_file("data/pure_math.json", builder.build_pure_math(10))
    save_prompts_to_file("data/pure_code.json", builder.build_pure_code(10))
    save_prompts_to_file("data/mixed_fincode.jsonl", builder.build_mixed_fincode(10))
    print("Dummy datasets generated in data/ directory.")
