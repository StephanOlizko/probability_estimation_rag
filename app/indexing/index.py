import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self, index_type="faiss"):
        self.index_type = index_type
        self.index = None
        self.vectors = None
        self.chunks = None  # сюда будем сохранять тексты

    def build(self, vectors, chunks):
        logger.info("Building index...")
        self.vectors = vectors
        self.chunks = chunks

        dim = vectors.shape[1] if hasattr(vectors, 'shape') else len(vectors[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(vectors).astype("float32"))
        logger.info("Index built successfully.")

    def search(self, query_vector, top_k=5):
        logger.info("Searching index...")
        D, I = self.index.search(np.array([query_vector]).astype("float32"), top_k)
        logger.info(f"Search completed. Found {len(I[0])} results.")
        return [self.chunks[i] for i in I[0]]
