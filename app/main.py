from config import Config
from logger import setup_logger
from data_loader.loader import chunk_texts
from encoders.encoders import EncoderFactory, TfidfEncoder, SBERTEncoder
from indexing.index import Indexer
from reranker.reranker import Reranker
from evaluation.metrics import evaluate
import pandas as pd
from utils.helpers import OpenRouterClient, get_general_topic, get_relevant_news_links, get_news_text_from_links, generate_response_based_on_context
import logging

setup_logger()
logger = logging.getLogger(__name__)

config = Config()

def main():
    logger.info("Starting pipeline")
    
    query = config.query
    logger.info(f"Query: {query}")

    general_topic = get_general_topic(query)
    logger.info(f"General topic: {general_topic}")

    links = get_relevant_news_links(general_topic, max_results=config.fetching_links_num)
    logger.info(f"Fetched {len(links)} links")

    texts = get_news_text_from_links(links)
    logger.info(f"Fetched {len(texts)} texts")

    documents = chunk_texts(texts, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = [doc.page_content for doc in documents]
    logger.info(f"Chunked texts into {len(chunks)} chunks")


    encoder_dense = SBERTEncoder()
    logger.info(f"Using encoder: {encoder_dense.__class__.__name__}")
    encoder_sparse = TfidfEncoder()
    logger.info(f"Using encoder: {encoder_sparse.__class__.__name__}")
    
    
    vectors_dense = encoder_dense.encode(chunks)
    logger.info(f"Encoded {len(chunks)} chunks into dense vectors")
    vectors_sparse = encoder_sparse.fit_transform(chunks)
    logger.info(f"Encoded {len(chunks)} chunks into sparse vectors")
    

    indexer_dense = Indexer(index_type=config.index_type)
    logger.info(f"Using indexer: {indexer_dense.__class__.__name__}")
    indexer_sparse = Indexer(index_type=config.index_type)
    logger.info(f"Using indexer: {indexer_sparse.__class__.__name__}")
    

    indexer_dense.build(vectors_dense, chunks)
    logger.info(f"Built index for dense vectors")
    indexer_sparse.build(vectors_sparse, chunks)
    logger.info(f"Built index for sparse vectors")


    query_vector_dense = encoder_dense.encode([query])
    logger.info(f"Encoded query into dense vector")
    query_vector_sparse = encoder_sparse.transform([query])
    logger.info(f"Encoded query into sparse vector")


    # retrieval from dense index
    results_dense = indexer_dense.search(query_vector=query_vector_dense[0], top_k=config.k)
    logger.info(f"Retrieved {len(results_dense)} results from dense index")

    # retrieval from sparse index
    results_sparse = indexer_sparse.search(query_vector=query_vector_sparse[0], top_k=config.k)
    logger.info(f"Retrieved {len(results_sparse)} results from sparse index")

    added_results = list(set(results_dense) | set(results_sparse))
    logger.info(f"Combined results: {len(added_results)}")


    # reranking
    reranker = Reranker()
    logger.info(f"Using reranker: {reranker.__class__.__name__}")
    reranked_results = reranker.rerank(query, added_results)
    logger.info(f"Reranked results: {len(reranked_results)}")

    reranked_results = reranked_results[:config.k]
    logger.info(f"Top {config.k} reranked results")


    # form model context for query
    messages = [
        {
            "role": "user",
            "content": f"Please provide a detailed answer to the following question based on the context: {query} \n\n\n Context: {reranked_results}"
        },
        {
            "role": "system",
            "content": "You are a helpful assistant that provides detailed answers to the questions based on the context."
        }
    ]

    logger.info(f"Formed model context for query")
    logger.info(f"Messages: {messages}")
    logger.info(f"Total context length: {sum([len(msg['content']) for msg in messages])} characters")

    # generate response based on context
    response = generate_response_based_on_context(messages, model="google/gemini-2.5-pro-exp-03-25:free")
    logger.info(f"Generated response: {response}")

    logger.info(f"Final text: {response['choices'][0]['message']['content']}")
    logger.info(f"Final text length: {len(response['choices'][0]['message']['content'])} characters")

    
    

if __name__ == "__main__":
    main()