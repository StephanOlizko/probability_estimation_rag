from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

class EncoderFactory:
    @staticmethod
    def create(names):
        encoders = []
        for encoder_type in names:
            if encoder_type == "tfidf":
                encoders.append(TfidfEncoder)
            elif encoder_type == "sentence_transformer":
                encoders.append(SBERTEncoder)
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

class TfidfEncoder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

class SBERTEncoder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

