import logging
import numpy as np

from typing import Any
from typing import Iterable

from langchain_core.embeddings import Embeddings

from Custom.app.rag_config import RAGConfig


logger = logging.getLogger()


class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model: Any, normalize: bool):
        self.model = model
        self.normalize = normalize
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if self.normalize:
            embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-10
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        if self.normalize:
            embedding /= np.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-10
        return embedding.tolist()


def embed_chunks(*,
    chunks: Iterable[Document],
    embedding_model: Any,
    normalize: bool,
    batch_size: int = 32
) -> Iterator[tuple[np.ndarray, Document]]:
    """
    Embeds chunks in batches, yielding (embedding, Document) pairs.

    Args:
        chunks: Iterable of Document objects to process.
        embedding_model: The model to create the text embeddings.
        batch_size: Number of documents to embed per batch.

    Yields:
        Tuples of (embedding array, Document) for each chunk.
    """
    # Create batches of text and Document metadata.
    embedder = SentenceTransformersEmbeddings(model, normalize=normalize)
    batch = []
    docs_batch = []
    for chunk in chunks:
        batch.append(chunk.page_content)  # text
        docs_batch.append(chunk)  # metadata
        # Process batch: encode text and yield embedding vector + metadata.
        if len(batch) == batch_size:
            yield from process_batch(batch, docs_batch, embedder)
            # Restart batch.
            batch.clear()
            docs_batch.clear()
    
    # Process possible leftover batch.
    if batch:
        yield from process_batch(batch, docs_batch, embedder)
    

def process_batch(
    texts: list[str],
    metadata: list[Document],
    embedder: SentenceTransformersEmbeddings,
) -> Iterator[tuple[np.ndarray, Document]]:
    """
    Generates embeddings for a batch of texts and pairs them with their corresponding Documents.

    Args:
        texts: A list of raw text to embed.
        metadata: A list of Document objects containing metadata for each text.
        embedder: A wrapper class inheriting LangChain's Embeddings, used to embed documents.

    Yields:
        Tuples of (embedding array, Document), one for each input text.
    """
    embeddings = embedder.embed_documents(texts)
    for emb, doc in zip(embeddings, metadata):
        emb = np.array(emb, dtype=np.float32)
        yield(emb, doc)
