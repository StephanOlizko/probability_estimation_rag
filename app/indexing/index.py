import faiss
import numpy as np

class Indexer:
    def __init__(self, index_type="faiss"):
        self.index_type = index_type
        self.index = None
        self.vectors = None
        self.chunks = None  # сюда будем сохранять тексты

    def build(self, vectors, chunks):
        self.vectors = vectors
        self.chunks = chunks

        dim = vectors.shape[1] if hasattr(vectors, 'shape') else len(vectors[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(vectors).astype("float32"))

    def search(self, query_vector, top_k=5):
        D, I = self.index.search(np.array([query_vector]).astype("float32"), top_k)
        return [self.chunks[i] for i in I[0]]
