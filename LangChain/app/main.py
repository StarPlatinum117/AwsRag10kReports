import logging
from typing import Any

from Custom.logging_config import setup_logging
from Custom.app.rag_config import AWS_RAG_CONFIG
from Custom.app.rag_config import RAGConfig

from LangChain.app.loader import load_10k_text_files
from LangChain.app.chunker import chunk_documents

setup_logging()
logger = logging.getLogger(__name__)
logger.info("App started (LangChain version).")


def run_rag_retrieval_pipeline(config: RAGConfig) -> None:
    """
    Executes a LangChain-based Retrieval-Augmented Generation (RAG) pipeline to build a document index.

    Args:
        config: An instance of the RAGConfig class to run all the functions. 
    """
    logger.info("Commencing index generation pipeline.")
    generator_documents = load_10k_text_files(config.doc_dir)
    generator_chunks = chunk_documents(generator_documents, config.chunk_size, config.chunk_overlap)
    

if __name__ == "__main__":
    run_rag_retrieval_pipeline(AWS_RAG_CONFIG)