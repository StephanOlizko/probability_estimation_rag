from utils.helpers import OpenRouterClient, GroqClient, AsyncGroqClient
from utils.prompts import PromptFactory, ResponseProbJustification
from data_loader.loader import chunk_texts
from indexing.index import Indexer
from encoders.encoders import EncoderFactory, TfidfEncoder, SBERTEncoder
from reranker.reranker import Reranker
from utils.helpers import get_general_topic, get_relevant_news_links, get_news_text_from_links, get_news_text_from_links_async
from logger import setup_logger
from config import Config
import asyncio
import logging

setup_logger()
logger = logging.getLogger(__name__)

config = Config()

class BaseRAG:
    """
    Базовый класс для RAG (Retrieval-Augmented Generation) пайплайнов.
    """
    def __init__(self, model_name = "llama-3.1-8b-instant", clinet = GroqClient):
        """
        Инициализация RAG пайплайна.

        :param model_name: Имя модели для генерации ответов.
        :param clinet: Клиент для взаимодействия с моделью.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__} with model {model_name}")
        self.model_name = model_name
        self.client = clinet(model=model_name)
    
    def run(self, query):
        """
        Запуск RAG пайплайна.

        :param query: Запрос для обработки.
        :return: Ответ от модели.
        """
        raise NotImplementedError("Метод run должен быть переопределен в подклассах.")
    


class BinaryQuestionPlainLLM(BaseRAG):
    """
    RAG пайплайн для обработки бинарных вопросов с использованием LLM.
    """
    def __init__(self, model_name = "llama-3.1-8b-instant", clinet = GroqClient):
        super().__init__(model_name=model_name, clinet=clinet)
        
    
    def run(self, query):
        """
        Запуск RAG пайплайна для бинарных вопросов.

        :param query: Запрос для обработки.
        :return: Ответ от модели.
        """
        self.logger.info(f"Processing query: '{query}'")
        messages = PromptFactory.create_probability_prompt(query)
        self.logger.debug("Sending request to LLM")
        response = self.client.generate_response(messages=messages, response_format={"type": "json_object"})
        self.logger.info("Response received")
        
        return ResponseProbJustification.model_validate_json(response)
    


class AsyncBinaryQuestionPlainLLM(BaseRAG):
    """
    Асинхронный RAG пайплайн для обработки бинарных вопросов с использованием LLM.
    """
    def __init__(self, model_name = "llama-3.1-8b-instant", clinet = AsyncGroqClient):
        super().__init__(model_name=model_name, clinet=clinet)

    async def run(self, query):
        """
        Запуск асинхронного RAG пайплайна для бинарных вопросов.

        :param query: Запрос для обработки.
        :return: Ответ от модели.
        """
        self.logger.info(f"Processing query asynchronously: '{query}'")
        messages = PromptFactory.create_probability_prompt(query)
        self.logger.debug("Sending async request to LLM")
        response = await self.client.generate_response(messages=messages, response_format={"type": "json_object"})
        self.logger.info("Async response received")
        
        return ResponseProbJustification.model_validate_json(response)



class BinaryQuestionNaiveRAG(BaseRAG):
    """
    Наивный RAG пайплайн для обработки бинарных вопросов с использованием LLM и новостей.
    """
    def __init__(self, model_name = "llama-3.1-8b-instant", clinet = GroqClient):
        super().__init__(model_name=model_name, clinet=clinet)
        self.logger.info("Initializing SBERT encoder")
        self.encoder = SBERTEncoder()
    
    def get_data_from_theguardian(self, query):
        """
        Получение данных из The Guardian по запросу.

        :param query: Запрос для поиска новостей.
        :return: Список ссылок на новости.
        """
        self.logger.info(f"Fetching data from The Guardian for query: '{query}'")
        self.logger.info("Extracting general topic")
        topic = get_general_topic(query)
        self.logger.info(f"Extracted topic: '{topic}'")
        
        self.logger.info(f"Getting relevant news links (max results: {config.fetching_links_num})")
        links = get_relevant_news_links(topic, max_results=config.fetching_links_num)
        self.logger.info(f"Found {len(links)} news links")
        
        self.logger.info("Fetching text from links")
        texts = get_news_text_from_links(links)
        self.logger.info(f"Retrieved {len(texts)} text articles")

        return texts
    
    def get_data_from_local(self, query, articles):
        """
        Получение данных из локального источника по запросу.

        :param query: Запрос для поиска новостей.
        :param articles: Локальные статьи для поиска.
        :return: Список ссылок на новости.
        """
        self.logger.info(f"Retrieving data from local articles for query: '{query}'")
        texts = [article["text"] for article in articles]
        self.logger.info(f"Extracted {len(texts)} articles from local data")

        return texts
    
    def run(self, query, data = None):
        """
        Запуск RAG пайплайна для бинарных вопросов.

        :param query: Запрос для обработки.
        :param data: Данные для поиска.
        :return: Ответ от модели.
        """
        self.logger.info(f"Starting naive RAG pipeline for query: '{query}'")
        
        # Data acquisition
        self.logger.info("Acquiring data")
        if data == None:
            self.logger.info("No data provided, fetching from The Guardian")
            texts = self.get_data_from_theguardian(query)
        else:
            self.logger.info("Using provided local data")
            texts = self.get_data_from_local(query, data)

        # Text chunking
        self.logger.info(f"Chunking texts with size: {config.chunk_size}, overlap: {config.chunk_overlap}")
        documents = chunk_texts(texts, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = [doc.page_content for doc in documents]
        self.logger.info(f"Created {len(chunks)} chunks")

        # Encoding
        self.logger.info("Encoding chunks")
        vectors = self.encoder.encode(chunks)
        query_vector = self.encoder.encode([query])
        self.logger.info(f"Generated {len(vectors)} vectors")

        # Indexing and retrieval
        self.logger.info("Building index")
        indexer = Indexer(index_type=config.index_type)
        indexer.build(vectors, chunks)

        self.logger.info(f"Searching for relevant chunks (top-k={config.k})")
        results = indexer.search(query_vector=query_vector[0], top_k=config.k)
        self.logger.info(f"Found {len(results)} relevant chunks")

        # LLM response generation
        context = "\n\n".join(results)
        self.logger.info(f"Context size: {len(context)} characters")
        self.logger.info("Creating prompt with context")
        messages = PromptFactory.create_probability_prompt_with_context(query, context)

        self.logger.info(f"Sending request to LLM ({self.model_name})")
        response = self.client.generate_response(messages=messages, response_format={"type": "json_object"})
        self.logger.info("Response received")
        
        return ResponseProbJustification.model_validate_json(response)



class AsyncBinaryQuestionNaiveRAG(BaseRAG):
    """
    Асинхронный наивный RAG пайплайн для обработки бинарных вопросов с использованием LLM и новостей.
    """
    def __init__(self, model_name="llama-3.1-8b-instant", clinet=AsyncGroqClient):
        super().__init__(model_name=model_name, clinet=clinet)
        self.logger.info("Initializing SBERT encoder")
        self.encoder = SBERTEncoder()
    
    async def get_data_from_theguardian(self, query, date):
        """
        Асинхронное получение данных из The Guardian по запросу.

        :param query: Запрос для поиска новостей.
        :return: Список текстов новостей.
        """
        self.logger.info(f"Fetching data from The Guardian for query: '{query}'")
        self.logger.info("Extracting general topic")
        topic = get_general_topic(query)
        self.logger.info(f"Extracted topic: '{topic}'")
        
        self.logger.info(f"Getting relevant news links (max results: {config.fetching_links_num})")
        links = get_relevant_news_links(topic, max_results=config.fetching_links_num, params={"to-date": date})
        self.logger.info(f"Found {len(links)} news links")
        
        self.logger.info("Fetching text from links asynchronously")
        texts = await get_news_text_from_links_async(links)
        self.logger.info(f"Retrieved {len(texts)} text articles")

        return texts
    
    def get_data_from_local(self, query, articles):
        """
        Получение данных из локального источника по запросу.

        :param query: Запрос для поиска новостей.
        :param articles: Локальные статьи для поиска.
        :return: Список текстов статей.
        """
        self.logger.info(f"Retrieving data from local articles for query: '{query}'")
        texts = [article["text"] for article in articles]
        self.logger.info(f"Extracted {len(texts)} articles from local data")

        return texts
    
    async def run(self, query, data=None, date=None):
        """
        Запуск асинхронного RAG пайплайна для бинарных вопросов.

        :param query: Запрос для обработки.
        :param data: Данные для поиска.
        :return: Ответ от модели.
        """
        self.logger.info(f"Starting async naive RAG pipeline for query: '{query}'")
        
        # Data acquisition
        self.logger.info("Acquiring data")
        if data is None:
            self.logger.info("No data provided, fetching from The Guardian")
            texts = await self.get_data_from_theguardian(query, date)
        else:
            self.logger.info("Using provided local data")
            texts = self.get_data_from_local(query, data)

        # Text chunking
        self.logger.info(f"Chunking texts with size: {config.chunk_size}, overlap: {config.chunk_overlap}")
        documents = chunk_texts(texts, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = [doc.page_content for doc in documents]
        self.logger.info(f"Created {len(chunks)} chunks")

        # Encoding
        self.logger.info("Encoding chunks")
        vectors = self.encoder.encode(chunks)
        query_vector = self.encoder.encode([query])
        self.logger.info(f"Generated {len(vectors)} vectors")

        # Indexing and retrieval
        self.logger.info("Building index")
        indexer = Indexer(index_type=config.index_type)
        indexer.build(vectors, chunks)

        self.logger.info(f"Searching for relevant chunks (top-k={config.k})")
        results = indexer.search(query_vector=query_vector[0], top_k=config.k)
        self.logger.info(f"Found {len(results)} relevant chunks")

        # LLM response generation
        context = "\n\n".join(results)
        self.logger.info(f"Context size: {len(context)} characters")
        self.logger.info("Creating prompt with context")
        messages = PromptFactory.create_probability_prompt_with_context(query, context)

        self.logger.info(f"Sending async request to LLM ({self.model_name})")
        response = await self.client.generate_response(messages=messages, response_format={"type": "json_object"})
        self.logger.info("Response received")
        
        return ResponseProbJustification.model_validate_json(response)



class AsyncBinaryQuestionRAG_hybrid_crossencoder(BaseRAG):
    """
    Асинхронный RAG пайплайн для обработки бинарных вопросов с использованием LLM и новостей.
    """
    def __init__(self, model_name="llama-3.1-8b-instant", clinet=AsyncGroqClient):
        super().__init__(model_name=model_name, clinet=clinet)
        self.logger.info("Initializing AsyncBinaryQuestionRAG_hybrid_crossencoder")
        
        self.logger.info("Initializing dense encoder")
        self.encoder_dense = SBERTEncoder()
        self.logger.info(f"Dense encoder initialized: {type(self.encoder_dense).__name__}")
        
        self.logger.info("Initializing sparse encoder")
        self.encoder_sparse = TfidfEncoder()
        self.logger.info(f"Sparse encoder initialized: {type(self.encoder_sparse).__name__}")
        
        self.logger.info("Initializing reranker")
        self.reranker = Reranker()
        self.logger.info(f"Reranker initialized: {type(self.reranker).__name__}")
        
        self.logger.info("Initialization complete!")
    
    async def get_data_from_theguardian(self, query, date):
        """
        Асинхронное получение данных из The Guardian по запросу.

        :param query: Запрос для поиска новостей.
        :return: Список текстов новостей.
        """
        self.logger.info(f"Fetching data from The Guardian for query: '{query}'")
        self.logger.info("Extracting general topic")
        topic = get_general_topic(query)
        self.logger.info(f"Extracted topic: '{topic}'")
        
        self.logger.info(f"Getting relevant news links (max results: {config.fetching_links_num})")
        links = get_relevant_news_links(topic, max_results=config.fetching_links_num, params={"to-date": date})
        self.logger.info(f"Found {len(links)} news links")
        
        self.logger.info("Fetching text from links asynchronously")
        texts = await get_news_text_from_links_async(links)
        self.logger.info(f"Retrieved {len(texts)} text articles")

        return texts
    
    def get_data_from_local(self, query, articles):
        """
        Получение данных из локального источника по запросу.

        :param query: Запрос для поиска новостей.
        :param articles: Локальные статьи для поиска.
        :return: Список текстов статей.
        """
        self.logger.info(f"Retrieving data from local articles for query: '{query}'")
        texts = [article["text"] for article in articles]
        self.logger.info(f"Extracted {len(texts)} articles from local data")

        return texts
    
    async def run(self, query, data=None, date=None):
        """
        Запуск асинхронного RAG пайплайна для бинарных вопросов.

        :param query: Запрос для обработки.
        :param data: Данные для поиска.
        :return: Ответ от модели.
        """
        self.logger.info(f"Starting hybrid RAG pipeline for query: '{query}'")
        
        # Data acquisition
        self.logger.info("STEP 1: Acquiring data")
        if data is None:
            self.logger.info("No data provided, fetching from The Guardian")
            texts = await self.get_data_from_theguardian(query, date)
        else:
            self.logger.info("Using provided local data")
            texts = self.get_data_from_local(query, data)
        self.logger.info(f"Data acquisition complete. Total texts: {len(texts)}")

        # Text chunking
        self.logger.info("STEP 2: Chunking texts")
        self.logger.info(f"Chunk size: {config.chunk_size}, overlap: {config.chunk_overlap}")
        documents = chunk_texts(texts, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = [doc.page_content for doc in documents]
        self.logger.info(f"Chunking complete. Total chunks: {len(chunks)}")
        
        # Encoding
        self.logger.info("STEP 3: Encoding chunks")
        self.logger.info("Creating dense embeddings with SBERT encoder")
        vectors_dense = self.encoder_dense.encode(chunks)
        query_vector_dense = self.encoder_dense.encode([query])
        self.logger.info(f"Dense encoding complete: {len(vectors_dense)} vectors generated")
        
        self.logger.info("Creating sparse embeddings with TF-IDF encoder")
        vectors_sparse = self.encoder_sparse.fit_transform(chunks)
        query_vector_sparse = self.encoder_sparse.transform([query])
        self.logger.info(f"Sparse encoding complete: {vectors_sparse.shape} vectors generated")

        # Indexing
        self.logger.info("STEP 4: Building indices")
        self.logger.info("Building dense index")
        indexer_dense = Indexer()
        indexer_dense.build(vectors_dense, chunks)
        self.logger.info("Building sparse index")
        indexer_sparse = Indexer()
        indexer_sparse.build(vectors_sparse, chunks)
        self.logger.info("Index building complete")

        # Retrieval
        self.logger.info("STEP 5: Retrieving relevant chunks")
        self.logger.info(f"Searching with dense index (top-k = {config.k})")
        results_dense = indexer_dense.search(query_vector=query_vector_dense[0], top_k=config.k)
        self.logger.info(f"Found {len(results_dense)} chunks from dense search")
        
        self.logger.info(f"Searching with sparse index (top-k = {config.k})")
        results_sparse = indexer_sparse.search(query_vector=query_vector_sparse[0], top_k=config.k)
        self.logger.info(f"Found {len(results_sparse)} chunks from sparse search")
        
        # Merging and reranking
        self.logger.info("STEP 6: Hybrid fusion and reranking")
        added_results = list(set(results_dense) | set(results_sparse))
        self.logger.info(f"Total unique chunks after fusion: {len(added_results)}")
        
        self.logger.info("Applying cross-encoder reranking")
        reranked_results = self.reranker.rerank(query, added_results)
        self.logger.info(f"Reranking complete. Using top {len(reranked_results)} chunks")

        # Context preparation
        self.logger.info("STEP 7: Preparing context for LLM")
        context = "\n\n".join(reranked_results)
        self.logger.info(f"Context size: {len(context)} characters")
        
        # LLM response generation
        self.logger.info("STEP 8: Generating response with LLM")
        self.logger.info("Creating prompt with context")
        messages = PromptFactory.create_probability_prompt_with_context(query, context)
        
        self.logger.info(f"Sending request to LLM ({self.model_name})")
        for i in range(3):
            try:
                response = await self.client.generate_response(messages=messages, response_format={"type": "json_object"})
                parsed_response = ResponseProbJustification.model_validate_json(response)
                self.logger.info("Response received and parsed successfully")

            except Exception as e:
                self.logger.error(f"Error during LLM response generation: {e}")
                if i < 5:
                    self.logger.info("Retrying...")
                else:
                    self.logger.error("Max retries reached. Returning None.")
                    return None
        
        return parsed_response


if __name__ == "__main__":
    # Пример использования
    import pandas as pd
    df_1 = pd.read_json('/GitHub/rag/PROPHET/data_2024-8/dataset_L1.json')
    query = df_1.iloc[0]['question']
    articles = df_1.iloc[0]['articles']
    logger.info(f"Query: {query}")

    import time

    logger.info("Running Async Hybrid RAG with Cross-Encoder")

    async_rag_pipeline = AsyncBinaryQuestionRAG_hybrid_crossencoder()
    start_time = time.time()
    response = asyncio.run(async_rag_pipeline.run(query, data=articles))
    end_time = time.time()
    logger.info(f"Response: {response}")
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")

    logger.info("_" * 50)