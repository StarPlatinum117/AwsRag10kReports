# =================== Add sys.path to fix import errors =================
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =======================================================================
import logging
from typing import Any

from app.chunker import split_documents
from app.embedder import embed_chunks
from app.indexer import build_index
from aws.s3_utils import list_s3_files_with_suffix
from aws.s3_utils import transform_s3_uri_to_dict
from logging_config import setup_logging
from rag_config import AWS_RAG_CONFIG
from rag_config import RAGConfig

setup_logging()
logger = logging.getLogger(__name__)
logger.info("App started.")


def run_rag_retrieval_pipeline(config: RAGConfig) -> None:
    """
    Executes a Retrieval-Augmented Generation (RAG) pipeline to build a document index.

    This function reads `.txt` documents from a directory, splits them into overlapping chunks,
    embeds each chunk using a provided model, and writes the resulting index and metadata to disk.

    Args:
        config: An instance of the RAGConfig class to run all the functions. 
    """
    logger.info("Commencing index generation pipeline.")
    logger.info("Fetching documents...")
    # Get the documents. S3 URI is converted to {"filename": ..., "text":}.
    generator_documents = (
        transform_s3_uri_to_dict(s3_uri)
        for s3_uri in list_s3_files_with_suffix(s3_prefix=config.doc_dir, suffix=".txt")
    )
    #ToDo: this step is for quick testing. Delete later.
    generator_documents = [next(generator_documents) for _ in range(2)]

    logger.info("Splitting documents into chunks...")
    # Split documents into chunks.
    generator_chunks = split_documents(
        documents=generator_documents,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )

    logger.info("Generating chunk embeddings...")
    # Create chunk embeddings.
    generator_embeddings, embeddings_dimension = embed_chunks(
        chunks=generator_chunks,
        model=config.embedding_model,
        normalize=config.normalize_embeddings,
    )

    logger.info("Creating FAISS index...")
    # Generate index and metadata files.
    build_index(
        chunks=generator_embeddings,
        dim=embeddings_dimension,
        output_index_path=config.path_to_index,
        output_metadata_path=config.path_to_metadata,
    )


if __name__ == "__main__":
    """
    CLI entry point to build the RAG index using the config file AWS_RAG_CONFIG.
    """
    #run_rag_retrieval_pipeline(AWS_RAG_CONFIG)
    logger.setLevel(logging.DEBUG)
    logger.debug("It's debug time!")
    logger.setLevel(logging.INFO)
    logger.info("It's info time now!")
    retriever = get_rag_retriever(AWS_RAG_CONFIG)
    query = "What are the main financial risks disclosed in the 2022 filings?"
    query_emb = AWS_RAG_CONFIG.embedding_model.encode(query, show_progress_bar=True)
    results = retriever.retrieve(query_emb, print_sample=True)
