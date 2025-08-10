import logging
from typing import Any

from langchain_community.document_loaders import S3DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Custom.logging_config import setup_logging
from Custom.app.rag_config import AWS_RAG_CONFIG
from Custom.app.rag_config import RAGConfig
from Custom.aws.s3_utils import parse_s3_uri

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
    
    # Load .txt files from S3 as Documents.
    logger.info("Fetching documents...")
    bucket, prefix = parse_s3_uri(config.doc_dir)
    loader = S3DirectoryLoader(bucket=bucket, prefix=prefix, suffix=".txt")
    documents = loader.load()
    documents = documents[:2]  # for testing, delete later

    # Split into chunks.
    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        
    )


if __name__ == "__main__":
    run_rag_retrieval_pipeline(AWS_RAG_CONFIG)