import logging
from typing import Iterable
from typing import Iterator

from smart_open import open as sopen

logger = logging.getLogger(__name__)


def split_documents(
    *,
    documents: Iterable[dict[str, str]],
    chunk_size: int,
    overlap: int,
) -> Iterator[dict[str, str]]:
    """
    Loads .txt files in the documents directory and splits them into overlapping chunks for the RAG pipeline.
    Args:
        document: Iterable of {"filename": ..., "text": ...}.
        chunk_size: The number of characters for each split.
        overlap: The number of overlapped characters between subsequent chunks.

    Yields:
        Chunks of documents of the form {"chunk_id": ..., "text": ..., "source_file": ...}.
    """
    chunks_yielded = 0
    chunks_skipped = 0
    for doc in documents:
        filename = doc["filename"]
        text = doc["text"]
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size].strip()
            if chunk:  # ensure chunk is not empty
                chunk_data = {"chunk_id": i + 1, "text": chunk, "source_file": filename}
                chunks_yielded += 1
                yield chunk_data
            else:
                chunks_skipped += 1
    logger.info(f"Total chunks yielded: {chunks_yielded}")
    logger.info(f"Empty chunks skipped: {chunks_skipped}")
        