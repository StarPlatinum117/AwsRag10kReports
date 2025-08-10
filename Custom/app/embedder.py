import logging

import numpy as np

from typing import Any
from typing import Iterable
from typing import Iterator

from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


def embed_chunks(
    *,
    chunks: Iterable[dict[str, str]],
    model: Any,
    normalize: bool = False,
) -> tuple[Iterator[dict[str, Any]], int]:
    """
    Creates embeddings for document chunks using the provided model.
    Args:
        chunks: Iterable of {"chunk_id": ..., "text": ..., "source_file": ...}.
        model: The model to generate the embeddings.
        normalize: If True, embeddings are normalized.
    
    Returns:
        The dimension of the embeddings.
        A generator of original chunks of documents with an additional "embedding" key containing the embeddings vector.
    """
    embedding_dim = get_embedding_dimension_from_model(model)

    def generator():
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            source_file = chunk["source_file"]
            try:
                # Create the embeddings as np.float32 and normalize if needed.
                embeddings = model.encode(chunk["text"], show_progress_bar=False)
                embeddings = np.asarray(embeddings, dtype=np.float32)
                if normalize:
                    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)  #(dim)
                
                # Ensure shape is (1, dim).
                if embeddings.ndim == 1:
                    embeddings = embeddings[np.newaxis, :]

                yield {
                    **chunk,
                    "embedding": embeddings,
                }
                logger.debug(f"Embedded chunk {chunk_id} from {source_file}.")
            except Exception as e:
                logger.error(f"Error embedding chunk {chunk_id} from {source_file}: \n{e}")

    return embedding_dim, generator()


def get_embedding_dimension_from_model(model: Any) -> int:
    """Extracts the embedding dimension based on the type of model used."""
    if isinstance(model, SentenceTransformer):
        return model.get_sentence_embedding_dimension()
    
    elif isinstance(model, PreTrainedModel):
        return model.config.hidden_size

    elif hasattr(model, "embedding_dim"):
        return model.embedding_dim
    
    else:
        raise TypeError(
            f"Unsupported model type {type(model)}."
            "Expected SentenceTransformer, HuggingFace model or custom model with 'embedding_dim' attribute"
        )
