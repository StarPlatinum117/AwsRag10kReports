import logging
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import boto3
from smart_open import open as sopen

logger = logging.getLogger(__name__)

# Define single boto3 client.
_s3_client = boto3.client("s3")

def get_s3_client():
    return _s3_client


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parses an S3 URI and returns the bucket name and object key.
    Args:
        s3_uri: A URI in the format 's3://<bucket>/<key>'.
    Returns:
        A tuple containing the bucket name and object key.
    Raises:
        ValueError: If the URI does not use the 's3' scheme.
    Example:
        parse_s3_uri("s3://my-bucket/path/to/object.txt")
        → ("my-bucket", "path/to/object.txt")
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")

    
def list_s3_files_with_suffix(*, s3_prefix: str, suffix: str = "") -> Iterator[str]:
    """
    Lists S3 object keys under a given prefix that end with a specific suffix.

    Args:
        s3_prefix: A fully qualified S3 URI pointing to the bucket and prefix, 
                   e.g. 's3://my-bucket/path/to/files/'.
        suffix: An optional string that each returned object's key must end with. 
                Defaults to an empty string (returns all files).

    Yields:
        A generator of S3 URIs (str) that match the given suffix.

    Example:
        list_s3_files_with_suffix("s3://my-bucket/data/", ".json")
        → yields 's3://my-bucket/data/file1.json', etc.
    """
    # Get the shared S3 client.
    s3 = get_s3_client()
    # Split S3 URI into bucket and key prefix.
    bucket, prefix = parse_s3_uri(s3_prefix)
    
    # Use paginator to keep requesting files until all matching objects are retrived.
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    # Iterate through each page of results.
    for page in page_iterator:
        # "Contents" contains the list of objects in the page (may be empty).
        for obj in page.get("Contents", []):
            key = obj["Key"]  # full path within the bucket
            # Yield keys that match the suffix filter.
            if key.endswith(suffix):
                yield f"s3://{bucket}/{key}"


def transform_s3_uri_to_dict(s3_uri: str) -> dict[str, str] | None:
    """
    Reads the contents of a text file stored at the given S3 URI and returns it 
    as a dictionary with the filename and file content.

    Args:
        s3_uri: S3 URI pointing to the file to be read.

    Returns:
        A dictionary with:
            - 'filename': the basename of the file (not the full S3 URI)
            - 'text': the file content as a string
        Returns None if the file cannot be read.
    """
    try:
        with sopen(s3_uri, "r", encoding="utf-8") as f:
            text = f.read()
            return {"filename": Path(s3_uri).name, "text": text}
    except Exception as e:
        logger.warning(f"Could not read {s3_uri}: {e}")
