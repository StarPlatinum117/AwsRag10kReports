# Modify sys.path to run file on AWS. ===================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
# =======================================================
import logging

from app.rag_config import AWS_RAG_CONFIG
from app.retriever import get_rag_retriever
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

model = AWS_RAG_CONFIG.embedding_model
retriever = get_rag_retriever(AWS_RAG_CONFIG, build_index=False)


def generate_response(query: str, k: int = 3) -> str:
    query_emb = model.encode(query, show_progress_bar=False)
    retrieved_chunks = retriever.retrieve(query_emb, k=k, print_sample=False)
    response = f"""
    =========================================================
    Query: {query}.
    Number of retrieved chunks: {k}.
    =========================================================
    The following chunks were retrieved:
    """

    for i, chunk in enumerate(retrieved_chunks):
        response += f"""\n
        -------------- Top {i + 1} chunk --------------
        Chunk ID: {chunk["chunk_id"]}. Source file: {chunk["source_file"]}
        Text: {chunk["text"]}\n
        """
    
    logging.info(response)
    return response

generate_response("Is it a good time to invest in this company?")
