# =================== Add sys.path to fix import errors =================
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =======================================================================
import logging
from typing import Any

from aws.s3_utils import list_s3_files_with_suffix
from aws.s3_utils import transform_s3_uri_to_dict
from chunker import split_documents
from embedder import embed_chunks
from indexer import build_index
from logging_config import setup_logging
from rag_config import AWS_RAG_CONFIG
from rag_config import RAGConfig
from retriever import DocumentRetriever

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
    

def get_rag_retriever(config: RAGConfig, build_index: bool = False) -> DocumentRetriever:
    """
    Initializes and returns a DocumentRetriever instance based on RAG configuration.

    If `build_index` is True, the RAG retrieval pipeline will be executed using the
    provided configuration before constructing the retriever. This is necessary when the index
    and metadata files have not yet been generated.

    Parameters:
        config: Configuration object containing all relevant paths, parameters, and embedding model required for retrieval.
        build_index: Whether to run the RAG pipeline before loading the retriever. Defaults to False.

    Returns:
        DocumentRetriever: An instance configured to perform retrieval using the precomputed index and metadata.
    """
    if build_index:
        run_rag_retrieval_pipeline(config)
    retriever = DocumentRetriever(config.path_to_index, config.path_to_metadata)
    return retriever


if __name__ == "__main__":
    """
    CLI entry point to build the RAG index using the config file AWS_RAG_CONFIG.
    """
    run_rag_retrieval_pipeline(AWS_RAG_CONFIG)