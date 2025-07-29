import json
import logging
from typing import Any
from typing import Iterable

import faiss
import numpy as np
from smart_open import open as sopen


logger = logging.getLogger(__name__)



class DocumentRetriever:
    def __init__(self, index_path: str, metadata_path: str):
        """
        Holds the FAISS index and initializes an offset index for fast retrieval of relevant chunks.
        
        Args:
            index_path: S3 URI where the index file is saved.
            metadata_path: S3 URI where the chunks metadata file is saved.
        """ 
        try:
            with sopen(index_path, "rb") as f:
                self.index = faiss.read_index(f)
                logger.debug(f"FAISS index loaded from {index_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_path}: {e}")
            raise
        self.metadata_path = metadata_path
        self.metadata_offsets = self.build_line_offsets(metadata_path)
    
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

    def retrieve(self, query_embedding: np.ndarray, k: int = 3) -> list[dict[str, Any]]:
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
        
        # Print matched chunks for debugging.
        logger.debug(f"Search results:\n{indices=}\n{scores=}")

        topk_chunks = self.get_chunk_metadata_by_index(indices.flatten(), scores.flatten())
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
