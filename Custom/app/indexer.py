import io
import json
import logging
import tempfile
from typing import Any
from typing import Iterable
from typing import Iterator

import faiss

from aws.s3_utils import get_s3_client
from aws.s3_utils import parse_s3_uri

s3 = get_s3_client()

logger = logging.getLogger(__name__)


def build_index(
    chunks: Iterable[dict[str, Any]],
    dim: int,
    output_index_path: str,
    output_metadata_path: str,
) -> None:
    """
    Builds a FAISS index from embedded chunks and saves it to the specified path in S3.
    Args:
        chunks: Iterable of {"chunk_id": ..., "text": ..., "source_file": ..., "embedding": ...}.
        dim: Dimensionality of the embedding space.
        output_index_path: Where to save the FAISS index.
        output_metadata_path: Where to save the chunks metadata.
    """
    index = faiss.IndexFlatIP(dim)
    count = 0

    # Metadata is written to memory as JSONL. Lines should be encoded.
    metadata_buffer = io.BytesIO()

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        try:
            # Add chunk to FAISS index.
            embedding = chunk["embedding"]  # (1, dim)
            index.add(embedding)
            # Save current chunk metadata to buffer file. Add newline manually. 
            del chunk["embedding"]
            jsonline = json.dumps(chunk, ensure_ascii=False) +"\n"
            metadata_buffer.write(jsonline.encode("utf-8"))
            count += 1

        except Exception as e:
            logger.error(f"Error adding chunk {chunk_id} to FAISS index: \n{e}")

    # Upload metadata to S3.
    metadata_buffer.seek(0)  # return to beginning of file
    metadata_bucket, metadata_key = parse_s3_uri(output_metadata_path)
    s3.upload_fileobj(metadata_buffer, Bucket=metadata_bucket, Key=metadata_key)
    logger.info(f"Chunks metadata uploaded to S3 at {output_metadata_path}")

    # Index is written to a temp file, then uploaded to S3.
    with tempfile.NamedTemporaryFile(suffix=".index", delete=True) as tmp:
        faiss.write_index(index, tmp.name)
        tmp.flush()  # to ensure all data is written
        index_bucket, index_key = parse_s3_uri(output_index_path)
        s3.upload_fileobj(tmp, Bucket=index_bucket, Key=index_key)
        logger.info(f"FAISS index with {count} vectors uploaded to S3 at {output_index_path}")
    