
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, candidates):
        inputs = self.tokenizer([f"{query} [SEP] {c}" for c in candidates], return_tensors="pt", padding=True, truncation=True)
        scores = self.model(**inputs).logits.squeeze()
        sorted_indices = torch.argsort(scores, descending=True)
        return [candidates[i] for i in sorted_indices]
