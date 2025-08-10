import logging

import numpy as np

from typing import Any
from typing import Iterable
from typing import Iterator

from langchain_core.documents import Document

from Custom.app.embedder import get_embedding_dimension_from_model

logger = logging.getLogger()

def embed_chunks(
    chunks: Iterable[Document],
    model: Any,
    normalize: bool = False,
) -> tuple[int, Iterator[tuple[np.ndarray, Document]]]:
    """
    Creates embeddings for document chunks using the provided model.
    
    Args:
        chunks: Iterable Document chunks to embed.
        model: The model to generate the embeddings.
        normalize: If True, embeddings are normalized.
    
    Returns:
        Tuple of (embedding_dimension, embedding_generator). 
        Where the generator yields a tuple of (embedding_vector, original Document).
    """
    embedding_dim = get_embedding_dimension_from_model(model)

    def generator():
        for chunk in chunks:
            idx = chunk.metadata["chunk_id"]
            source_file = chunk.metadata["source_file"]
            try:
                embeddings = model.encode(chunk.page_content)
                # Ensure embedding vector is float32 and (1, dim).
                embeddings = np.asarray(embeddings, dtype=np.float32)
                if embeddings.ndim == 1:
                    embeddings = embeddings[np.newaxis, :]

                if normalize:
                    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
                
                yield (embeddings, chunk)

            except:
                logger.error(f"Error embedding chunk {idx} from {source_file}")
    
    return (embedding_dim, generator())
