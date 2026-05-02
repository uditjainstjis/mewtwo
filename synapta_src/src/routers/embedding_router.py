import os
import json
import numpy as np
import time
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer

class EmbeddingRouter:
    """
    Computes domain centroids from training questions.
    Routes queries by soft cosine similarity to the centroids.
    """
    def __init__(self, data_expert_dir: str, temperature: float = 0.05):
        self.data_expert_dir = data_expert_dir
        self.temperature = temperature
        print("Loading embedding router model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.domain_centroids = {}
        self._build_centroids()

    def _extract_question(self, text: str) -> str:
        # Expected format: <|im_start|>user\n[QUESTION]<|im_end|>\n<|im_start|>assistant...
        start_marker = "<|im_start|>user\n"
        end_marker = "<|im_end|>"
        start_idx = text.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = text.find(end_marker, start_idx)
            if end_idx != -1:
                return text[start_idx:end_idx].strip()
        return text.strip()  # Fallback

    def _build_centroids(self):
        print(f"Building domain centroids from {self.data_expert_dir}...")
        start_time = time.time()
        for domain in os.listdir(self.data_expert_dir):
            domain_dir = os.path.join(self.data_expert_dir, domain)
            if not os.path.isdir(domain_dir): continue
            
            train_path = os.path.join(domain_dir, "train.jsonl")
            if not os.path.exists(train_path): continue
            
            questions = []
            with open(train_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    q = self._extract_question(data.get("text", ""))
                    if q: questions.append(q)
                    
            if questions:
                embeddings = self.model.encode(questions, normalize_embeddings=True)
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)  # L2 normalize
                self.domain_centroids[domain] = centroid
                
        print(f"Computed {len(self.domain_centroids)} centroids in {time.time()-start_time:.2f}s")

    def route_probs(self, query: str) -> Dict[str, float]:
        """Returns softmax probabilities for all domains."""
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        
        domains = list(self.domain_centroids.keys())
        sims = []
        for d in domains:
            sims.append(np.dot(query_emb, self.domain_centroids[d]))
            
        sims = np.array(sims)
        # Softmax with temperature
        exp_sims = np.exp(sims / self.temperature)
        probs = exp_sims / np.sum(exp_sims)
        
        return {d: float(p) for d, p in zip(domains, probs)}

    def route_top_k(self, query: str, k: int = 2) -> List[Tuple[str, float]]:
        probs = self.route_probs(query)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]
