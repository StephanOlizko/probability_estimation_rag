import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Encoder(ABC):
    @abstractmethod
    def encode(self, texts):
        pass

class SentenceTransformerEncoder(Encoder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        logger.info(f"Инициализация кодировщика с моделью: {model_name}, устройство: {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts):
        logger.info(f"Кодирование {len(texts)} текстов...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.debug(f"Пример эмбеддинга для первого текста: {embeddings[0][:10]}")  # Лог первых 10 элементов
        return embeddings
    
class TfIdfEncoder(Encoder):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts):
        logger.info("Обучение TF-IDF...")
        self.vectorizer.fit(texts)

    def encode(self, texts):
        logger.info(f"Кодирование {len(texts)} текстов...")
        embeddings = self.vectorizer.transform(texts).toarray()
        logger.debug(f"Пример эмбеддинга для первого текста: {embeddings[0][:10]}")
        return embeddings

class Index(ABC):
    @abstractmethod
    def build_index(self, embeddings, article_ids):
        pass

    @abstractmethod
    def find_nearest(self, query_embedding, top_k):
        pass

class FaissIndex(Index):
    def __init__(self):
        self.index = None

    def build_index(self, embeddings, article_ids):
        logger.info("Нормализация эмбеддингов...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        logger.debug(f"Первый нормализованный эмбеддинг: {normalized_embeddings[0][:10]}")

        dimension = normalized_embeddings.shape[1]
        logger.info(f"Создание индекса FAISS с размерностью {dimension}...")
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self.index.add_with_ids(np.array(normalized_embeddings, dtype=np.float32), article_ids)
        logger.info(f"Индекс построен, добавлено {len(article_ids)} статей.")

    def find_nearest(self, query_embedding, top_k):
        logger.info("Нормализация эмбеддинга запроса...")
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        normalized_query = query_embedding / query_norm
        logger.debug(f"Нормализованный эмбеддинг запроса: {normalized_query[0][:10]}")

        logger.info(f"Поиск {top_k} ближайших статей...")
        distances, indices = self.index.search(np.array(normalized_query, dtype=np.float32), top_k)
        logger.debug(f"Найденные индексы: {indices}, расстояния: {distances}")
        return indices, distances

class Retriever:
    def __init__(self, encoder: Encoder, index: Index):
        logger.info("Инициализация ретривера...")
        self.encoder = encoder
        self.index = index
        self.news_texts = []
        self.article_ids = []

    def index_news(self, article_ids, news_texts):
        logger.info(f"Индексация {len(news_texts)} статей...")
        self.news_texts = news_texts
        self.article_ids = np.array(article_ids, dtype=np.int64)
        embeddings = self.encoder.encode(self.news_texts)
        self.index.build_index(embeddings, self.article_ids)

    def retrieve(self, query, top_k=10):
        logger.info(f"Поиск релевантных статей для запроса: {query}")
        query_embedding = self.encoder.encode([query])
        indices, distances = self.index.find_nearest(query_embedding, top_k)
        retrieved_articles = []
        for j in range(len(indices[0])):
            article_id = indices[0][j]
            if article_id == -1:
                continue
            text = self.news_texts[self.article_ids.tolist().index(article_id)]
            retrieved_articles.append((article_id, text, distances[0][j]))
        logger.info(f"Найдено {len(retrieved_articles)} статей.")
        return retrieved_articles

class MetricsFinder:
    @staticmethod
    def calculate_metrics(retriever, df: pd.DataFrame, k: int):
        logger.info(f"Расчёт метрик для k={k}...")
        correct_hits = 0
        reciprocal_ranks = []
        ranks = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Оценка ретривера"):
            query = row["description"]
            true_article_content = row["content"]
            retrieved_articles = retriever.retrieve(query, top_k=k)
            retrieved_contents = [article[1] for article in retrieved_articles]

            if true_article_content in retrieved_contents:
                correct_hits += 1
                rank = retrieved_contents.index(true_article_content) + 1
                ranks.append(rank)
                reciprocal_ranks.append(1 / rank)
                logger.debug(f"Правильный ответ найден на позиции {rank} для запроса: {query}")
            else:
                ranks.append(k + 1)
                reciprocal_ranks.append(0)
                logger.debug(f"Правильный ответ не найден для запроса: {query}")

        accuracy = correct_hits / len(df)
        mean_rank = np.mean(ranks)
        mrr = np.mean(reciprocal_ranks)

        logger.info(f"Accuracy@k: {accuracy}, Mean Rank: {mean_rank}, MRR: {mrr}")
        return {
            "Accuracy@k": accuracy,
            "Mean Rank": mean_rank,
            "MRR": mrr
        }

class Ensemble:
    def __init__(self, retrievers=None, weights=None, reranker=None):
        self.retrievers = retrievers or []
        self.weights = weights or []
        self.reranker = reranker

    def retrieve(self, query, top_k=10):
        logger.info(f"Поиск релевантных статей для запроса: {query}")
        all_retrieved_articles = []
        for retriever in self.retrievers:
            retrieved_articles = retriever.retrieve(query, top_k=top_k)
            all_retrieved_articles.extend(retrieved_articles)

        logger.info(f"Всего {len(all_retrieved_articles)} статей.")

        if self.reranker:
            logger.info("Переранжирование результатов...")
            all_retrieved_articles = self.reranker.rerank(all_retrieved_articles, query)

        return all_retrieved_articles
    
    
class Ranker(ABC):
    @abstractmethod
    def rerank(self, results, query):
        pass

class SimpleRanker(Ranker):
    def rerank(self, results, query):
        # Возвращаем результаты без изменений
        return results
    
class CrossEncoderRanker(Ranker):
    def __init__(self, model_name="cross-encoder/ms-marco-TinyBERT-L-6", device="cpu"):
        from sentence_transformers.cross_encoder import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, results, query):
        logger.info("Переранжирование результатов с помощью CrossEncoder...")
        article_texts = [result[1] for result in results]
        scores = self.model.predict([(query, text) for text in article_texts], show_progress_bar=True)
        reranked_results = [(results[i][0], results[i][1], scores[i]) for i in range(len(results))]
        reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)
        return reranked_results


"""# Пример использования
df = pd.read_csv("clean_data.csv", nrows=1000)
logger.info(f"Загружено {len(df)} строк данных.")

encoder = SentenceTransformerEncoder()
index = FaissIndex()

retriever = Retriever(encoder, index)
retriever.index_news(df.index.tolist(), df["content"].tolist())

metrics = MetricsFinder.calculate_metrics(retriever, df, k=6)
print(metrics)

#{'Accuracy@k': 0.749, 'Mean Rank': np.float64(1.907), 'MRR': np.float64(0.6836666666666668)} k=3
#{'Accuracy@k': 0.807, 'Mean Rank': np.float64(2.524), 'MRR': np.float64(0.6964166666666667)} k=6"""

"""df = pd.read_csv("clean_data.csv", nrows=1000)
logger.info(f"Загружено {len(df)} строк данных.")

encoder_sentence = SentenceTransformerEncoder()
index_sentence = FaissIndex()

encoder_tfidf = TfIdfEncoder()
encoder_tfidf.fit(df["content"].tolist())
index_tfidf = FaissIndex()


retriever_sentence = Retriever(encoder_sentence, index_sentence)
retriever_tfidf = Retriever(encoder_tfidf, index_tfidf)
retriever_sentence.index_news(df.index.tolist(), df["content"].tolist())
retriever_tfidf.index_news(df.index.tolist(), df["content"].tolist())

ensemble_retriever = Ensemble([retriever_sentence, retriever_tfidf])
metrics = MetricsFinder.calculate_metrics(ensemble_retriever, df, k=3)

print(metrics)

#{'Accuracy@k': 0.808, 'Mean Rank': np.float64(1.943), 'MRR': np.float64(0.6968166666666668)}"""

"""df = pd.read_csv("clean_data.csv", nrows=1000)
logger.info(f"Загружено {len(df)} строк данных.")

encoder_sentence = SentenceTransformerEncoder()
index_sentence = FaissIndex()

encoder_tfidf = TfIdfEncoder()
encoder_tfidf.fit(df["content"].tolist())
index_tfidf = FaissIndex()


retriever_sentence = Retriever(encoder_sentence, index_sentence)
retriever_tfidf = Retriever(encoder_tfidf, index_tfidf)
retriever_sentence.index_news(df.index.tolist(), df["content"].tolist())
retriever_tfidf.index_news(df.index.tolist(), df["content"].tolist())

reranker = CrossEncoderRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

ensemble_retriever = Ensemble([retriever_sentence, retriever_tfidf], reranker=reranker)
metrics = MetricsFinder.calculate_metrics(ensemble_retriever, df, k=3)

print(metrics)

#{'Accuracy@k': 0.808, 'Mean Rank': np.float64(1.857), 'MRR': np.float64(0.7314666666666667)}"""


"""# Пример использования
df = pd.read_csv("clean_data.csv", nrows=10000)
logger.info(f"Загружено {len(df)} строк данных.")

encoder = SentenceTransformerEncoder()
index = FaissIndex()

retriever = Retriever(encoder, index)
retriever.index_news(df.index.tolist(), df["content"].tolist())

metrics = MetricsFinder.calculate_metrics(retriever, df, k=10)
print(metrics)

#{'Accuracy@k': 0.7379, 'Mean Rank': np.float64(4.2395), 'MRR': np.float64(0.5996222222222221)} k=10"""

"""df = pd.read_csv("clean_data.csv", nrows=10000)
logger.info(f"Загружено {len(df)} строк данных.")

encoder_sentence = SentenceTransformerEncoder()
index_sentence = FaissIndex()

encoder_tfidf = TfIdfEncoder()
encoder_tfidf.fit(df["content"].tolist())
index_tfidf = FaissIndex()


retriever_sentence = Retriever(encoder_sentence, index_sentence)
retriever_tfidf = Retriever(encoder_tfidf, index_tfidf)
retriever_sentence.index_news(df.index.tolist(), df["content"].tolist())
retriever_tfidf.index_news(df.index.tolist(), df["content"].tolist())

ensemble_retriever = Ensemble([retriever_sentence, retriever_tfidf])
metrics = MetricsFinder.calculate_metrics(ensemble_retriever, df, k=5)

print(metrics)
#{'Accuracy@k': 0.7513, 'Mean Rank': np.float64(2.9231), 'MRR': np.float64(0.6019553571428572)}"""

"""df = pd.read_csv("clean_data.csv", nrows=10000)
logger.info(f"Загружено {len(df)} строк данных.")

encoder_sentence = SentenceTransformerEncoder()
index_sentence = FaissIndex()

encoder_tfidf = TfIdfEncoder()
encoder_tfidf.fit(df["content"].tolist())
index_tfidf = FaissIndex()


retriever_sentence = Retriever(encoder_sentence, index_sentence)
retriever_tfidf = Retriever(encoder_tfidf, index_tfidf)
retriever_sentence.index_news(df.index.tolist(), df["content"].tolist())
retriever_tfidf.index_news(df.index.tolist(), df["content"].tolist())

reranker = CrossEncoderRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

ensemble_retriever = Ensemble([retriever_sentence, retriever_tfidf], reranker=reranker)
metrics = MetricsFinder.calculate_metrics(ensemble_retriever, df, k=5)

print(metrics)

#{'Accuracy@k': 0.7513, 'Mean Rank': np.float64(2.7626), 'MRR': np.float64(0.6380567460317459)}"""