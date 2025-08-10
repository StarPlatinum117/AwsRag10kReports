import logging
from typing import Iterable
from typing import Iterator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def chunk_documents(
    documents: Iterator[Document],
    chunk_size: int,
    chunk_overlap: int,    
) -> Iterator[Document]:
    """
    Splits a stream of documents into smaller chunks using LangChain's text splitter.

    Args:
        documents: An iterator of Document objects to be chunked.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Yields:
        Individual Document chunks produced from the original documents.
    """
    # Define the text splitter.
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    is_separator_regex=False,
    )
    # Split documents.
    chunks_yielded = 0
    for doc in documents:
        for idx, chunk in enumerate(text_splitter.split_documents([doc])):
            chunk.metadata["chunk_id"] = idx
            chunks_yielded += 1
            yield chunk
    logging.info(f"Total chunks yielded: {chunks_yielded}")
