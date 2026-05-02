import itertools
import random
from dataclasses import dataclass
from typing import List

@dataclass
class Hypothesis:
    id: str
    rank: int
    alpha: int
    lambda_ortho: float
    learning_rate: float
    description: str

class HypothesisGenerator:
    def __init__(self):
        # We will gridsearch CF-LoRA parameters 
        self.ranks = [16, 32, 64]
        self.lambdas = [0.001, 0.01, 0.05, 0.1]
        self.lrs = [1e-4, 2e-4, 3e-4]
        
        self.queue = self._generate_queue()
        
    def _generate_queue(self) -> List[Hypothesis]:
        combinations = list(itertools.product(self.ranks, self.lambdas, self.lrs))
        random.shuffle(combinations) # Shuffle to traverse hyper-space randomly
        
        queue = []
        for i, (r, lam, lr) in enumerate(combinations):
            queue.append(Hypothesis(
                id=f"EXP-{1000+i}",
                rank=r,
                alpha=r * 2,
                lambda_ortho=lam,
                learning_rate=lr,
                description=f"Phase-Transition Test (r={r}, λ={lam}, lr={lr})"
            ))
        return queue
        
    def get_next(self) -> Hypothesis:
        if self.queue:
            return self.queue.pop(0)
        # Endless nature: regenerate if empty
        self.queue = self._generate_queue()
        return self.get_next()
