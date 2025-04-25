
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        logger.info("Loading Reranker model...")
        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model.eval()
        logger.info("Reranker model loaded successfully.")

    def rerank(self, query, candidates):
        logger.info(f"Reranking candidates... ({len(candidates)} candidates)")
        inputs = self.tokenizer([f"{query} [SEP] {c}" for c in candidates], return_tensors="pt", padding=True, truncation=True)
        scores = self.model(**inputs).logits.squeeze()
        sorted_indices = torch.argsort(scores, descending=True)
        logger.info("Reranking completed.")
        return [candidates[i] for i in sorted_indices]
