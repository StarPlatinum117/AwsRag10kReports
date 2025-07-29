import json
import logging
import tempfile
from typing import Any
from typing import Iterable

import faiss
import numpy as np
from smart_open import open as sopen

from app.main import run_rag_retrieval_pipeline
from rag_config import RAGConfig

logger = logging.getLogger(__name__)


class DocumentRetriever:
    def __init__(self, index_path: str, metadata_path: str):
        """
        Holds the FAISS index and initializes an offset index for fast retrieval of relevant chunks.
        
        Args:
            index_path: S3 URI where the index file is saved.
            metadata_path: S3 URI where the chunks metadata file is saved.
        """ 
        self.index = self.read_index_file_from_S3(index_path)
        self.metadata_path = metadata_path
        self.metadata_offsets = self.build_line_offsets(metadata_path)
    
    @staticmethod
    def read_index_file_from_S3(path: str) -> faiss.Index:
        """
        Loads the FAISS index file from S3 as a temp file and reads it using faiss.
        Args: 
            path: S3 URI where the index file is saved.
        Returns:
            FAISS index.
        """
        try:
            with sopen(path, "rb") as s3_file:
                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    tmp_file.write(s3_file.read())
                    tmp_file.flush()
                    return faiss.read_index(tmp_file.name)
        except Exception as e:
            logger.error(f"Failed to load index file from {path}: {e}")

    @staticmethod
    def build_line_offsets(file_path: str) -> list[int]:
        """
        Builds a list of byte offsets corresponding to the start of each line in a JSONL file.

        This allows efficient random access to any line in the file without loading the entire
        file into memory. Each line in a JSONL (JSON Lines) file is a complete JSON object,
        and lines are separated by newline characters.

        By storing the byte offset of each line, we can later use `seek(offset)` to jump
        directly to a specific line and read it, which is ideal for memory-efficient retrieval.

        Args:
            file_path: Where the file is located.
        Returns:
            A list of integer offsets.
        """
        logger.debug(f"Creating list of byte offsets for metadata file...")
        encoding_type = "utf-8"
        offsets = []
        offset = 0
        try:
            with sopen(file_path, "r", encoding=encoding_type) as f:
                for line in f:
                    offsets.append(offset)
                    # Add the length of the current line (in bytes) to get the next line's offset.
                    offset += len(line.encode(encoding_type))  
                logger.debug("Byte offsets created successfully.")
        except Exception as e:
            logger.error(f"Failed to create byte offsets for {file_path}: {e}")
            raise
        return offsets

    def retrieve(self, query_embedding: np.ndarray, k: int = 3, print_sample: bool = False) -> list[dict[str, Any]]:
        """
        Searches the FAISS index using the query embedding and returns the top-k metadata entries.

        Args:
            query_embedding: The vector of embeddings for the user's query.
            k: The number of relevant results to return.
        
        Returns:
            The top-k relevant chunks as dicts with the corresponding metadata (id, text, source file and score).
        """
        # Enforce query embeddings are of shape (1, dim) and type float32 for the FAISS index to work.
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        query_embedding = query_embedding.astype(np.float32)
        scores, indices = self.index.search(query_embedding, k)

        topk_chunks = self.get_chunk_metadata_by_index(indices.flatten(), scores.flatten())
        
        # Print a small sample of retrieved chunks to console.
        if print_sample:
            self.print_chunk_samples(topk_chunks)
           
        return topk_chunks
    
    def get_chunk_metadata_by_index(self, line_indices: Iterable[int], scores: Iterable[float]) -> list[dict[str, Any]]:
        """
        Extracts chunk metadata corresponding to line indices and FAISS scores.
        
        Args:
            line_indices: Line index where the chunk is located in the metadata file.
            scores: Similarity scores for chunks.
        
        Returns:
            A list of dicts, each containing chunk metadata and its similarity score.
        """
        chunks = []
        try:
            with sopen(self.metadata_path, "r", encoding="utf-8") as f:
                for idx, score in zip(line_indices, scores):
                    offset = self.metadata_offsets[idx]
                    f.seek(offset)
                    chunk = json.loads(f.readline())
                    chunk["score"] = float(score)
                    chunks.append(chunk)
        except Exception as e:
            logger.error(f"Failed to load metadata file from {self.metadata_path}: {e}")
            raise
        return chunks
    
    @staticmethod
    def print_chunk_samples(chunks: list[dict[str, Any]]) -> None:
        logging.info("Chunks retrieved (excerpts): " + "="*30)
        for chunk in chunks:
            idx = chunk["chunk_id"]
            source = chunk["source_file"]
            sample_text = chunk["text"][:200].replace("\n", " ")
            logger.info(f"Source file: {source}. Chunk ID: {idx}. \nRetrieved text: {sample_text}")
        logging.info("="*70)


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