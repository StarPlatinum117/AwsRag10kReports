from dataclasses import dataclass
import logging

from sentence_transformers import SentenceTransformer
# from transformers import AutoModel
# from transformers import AutoTokenizer

from typing import Any

from Custom.aws.s3_paths import DOCUMENTS_DIR
from Custom.aws.s3_paths import PATH_TO_CHUNKS_METADATA
from Custom.aws.s3_paths import PATH_TO_FAISS_INDEX


# --- Select your model here ---
# Option 1: SentenceTransformer (default, recommended)
embedding_model = SentenceTransformer("thenlper/gte-base")

# Option 2: HuggingFace model (slightly more verbose)
# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
# model = AutoModel.from_pretrained("thenlper/gte-base")
# embedding_model = HFModelWrapper(model, tokenizer)  # this wrapper needs to be defined, not implemented.

# Option 3: Custom model  (not implemented)
# embedding_model = CustomEmbeddingModel(...)


@dataclass
class RAGConfig:
    doc_dir: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: Any
    path_to_index: str
    path_to_metadata: str
    normalize_embeddings: bool = False


AWS_RAG_CONFIG = RAGConfig(
    doc_dir=DOCUMENTS_DIR,
    chunk_size=1000,
    chunk_overlap=250,
    embedding_model=embedding_model,
    path_to_index=PATH_TO_FAISS_INDEX,
    path_to_metadata=PATH_TO_CHUNKS_METADATA,
    normalize_embeddings=True,
)