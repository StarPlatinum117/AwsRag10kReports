BUCKET = "sagemaker-studio-491085416297-d1gkcu6e1fv"

# Dir names.
DATA_DIR = "rag_data"
RAG_10K_DATA_DIR = "10k_clean"

# File names.
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "chunks_metadata.jsonl"

# Full S3 URIs.
DOCUMENTS_DIR = f"s3://{BUCKET}/{DATA_DIR}/{RAG_10K_DATA_DIR}"
PATH_TO_FAISS_INDEX = f"s3://{BUCKET}/{DATA_DIR}/{INDEX_FILE}"
PATH_TO_CHUNKS_METADATA = f"s3://{BUCKET}/{DATA_DIR}/{METADATA_FILE}"