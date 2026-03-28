from typing import List, Tuple

def select_experts_cot(prompt: str, expert_registry_ids: List[str], k: int) -> List[Tuple[str, float]]:
    """
    Evaluates the input query and yields the Top-K targeted experts and their unnormalized
    Sigmoid confidence scores.
    """
    prompt_lower = prompt.lower()
    scores = {eid: 0.1 for eid in expert_registry_ids}  # Background noise base
    
    if "price" in prompt_lower or "option" in prompt_lower:
        scores["finance"] = 0.92
    if "python" in prompt_lower or "script" in prompt_lower:
        scores["code"] = 0.88
    if "statutory" in prompt_lower or "liability" in prompt_lower:
        scores["legal"] = 0.95
    if "calculate" in prompt_lower or "equation" in prompt_lower:
        scores["math"] = 0.85
        
    # Sort by score descending and return top K
    sorted_experts = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_experts[:k]
