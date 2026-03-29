import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

class ClassifierRouter:
    """
    Trains a lightweight Logistic Regression classifier over the training data's embeddings.
    """
    def __init__(self, data_expert_dir: str):
        self.data_expert_dir = data_expert_dir
        print("Loading embedding classifier model (all-MiniLM-L6-v2) for training...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.domains = []
        self._train_classifier()

    def _extract_question(self, text: str) -> str:
        start_marker = "<|im_start|>user\n"
        end_marker = "<|im_end|>"
        start_idx = text.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = text.find(end_marker, start_idx)
            if end_idx != -1:
                return text[start_idx:end_idx].strip()
        return text.strip()

    def _train_classifier(self):
        print(f"Training domain classifier from {self.data_expert_dir}...")
        start_time = time.time()
        X_text = []
        y_labels = []
        
        domains = os.listdir(self.data_expert_dir)
        # Filter valid domains
        domains = [d for d in domains if os.path.exists(os.path.join(self.data_expert_dir, d, "train.jsonl"))]
        self.domains = sorted(domains)
        
        for domain in self.domains:
            train_path = os.path.join(self.data_expert_dir, domain, "train.jsonl")
            with open(train_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    q = self._extract_question(data.get("text", ""))
                    if q:
                        X_text.append(q)
                        y_labels.append(domain)
                        
        print(f"Extracted {len(X_text)} training queries across {len(self.domains)} domains")
        
        if len(X_text) > 0:
            X_emb = self.model.encode(X_text, normalize_embeddings=True)
            self.clf.fit(X_emb, y_labels)
            print(f"Classifier trained in {time.time()-start_time:.2f}s")
        else:
            raise ValueError("No training data found for the classifier!")

    def route_probs(self, query: str) -> Dict[str, float]:
        """Returns predict_proba probabilities for all domains."""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        probs = self.clf.predict_proba(query_emb)[0]
        # clf.classes_ might not match self.domains order exactly, use clf.classes_
        return {str(cls): float(p) for cls, p in zip(self.clf.classes_, probs)}

    def route_top_k(self, query: str, k: int = 2) -> List[Tuple[str, float]]:
        probs = self.route_probs(query)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]
