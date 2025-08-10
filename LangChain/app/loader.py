from typing import Iterator

from langchain_core.documents import Document

from Custom.aws.s3_utils import list_s3_files_with_suffix
from Custom.aws.s3_utils import transform_s3_uri_to_dict


def load_10k_text_files(doc_dir: str) -> Iterator[Document]:
    """
    Load all .txt files from the given S3 URI as LangChain documents.

    Args: 
        doc_dir: path to the bucket and prefix where the documents reside.
    
    Yields:
        A generator of LangChain Document objects.
    """
    s3_uris = list_s3_files_with_suffix(s3_prefix=doc_dir, suffix=".txt")
    generator_documents = (transform_s3_uri_to_dict(uri) for uri in s3_uris)

    for doc in generator_documents:
        yield Document(page_content=doc["text"], metadata=doc["filename"])
