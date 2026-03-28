"""
embeddings.py
-------------
BGE-large-en-v1.5 embedding model for the RAG pipeline.
Matches the embedding model used to build the 37,009-vector FAISS index.
"""

import os
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-large-en-v1.5")

_embedder = None  # module-level singleton


def get_embedder():
    """
    Returns a singleton SentenceTransformer embedder.
    Uses BGE-large-en-v1.5 (1024-dim) — same model used to build the index.

    Returns:
        SentenceTransformer instance
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDINGS_MODEL}")
        _embedder = SentenceTransformer(EMBEDDINGS_MODEL)
        logger.info("Embedding model loaded.")
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using BGE-large.

    Args:
        texts: List of strings to embed

    Returns:
        np.ndarray of shape (len(texts), 1024)
    """
    embedder = get_embedder()
    # BGE models benefit from the instruction prefix for retrieval queries
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single retrieval query.
    BGE recommends a prefix for query-side encoding.

    Args:
        query: The search query

    Returns:
        np.ndarray of shape (1024,)
    """
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([prefixed])[0]
