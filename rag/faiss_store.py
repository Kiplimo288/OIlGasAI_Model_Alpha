"""
faiss_store.py
--------------
FAISS vector store for document-grounded retrieval.
Manages the 37,009-vector OilGasAI knowledge base.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./rag/index/oilgas_faiss.index")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))


class FAISSStore:
    """
    Wraps a FAISS index with document metadata for retrieval.

    The index stores 1024-dim BGE-large embeddings.
    Metadata (text chunks, sources) is stored in a parallel JSON file.
    """

    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(".meta.pkl")
        self.index = None
        self.metadata = []  # list of {"text": ..., "source": ..., "chunk_id": ...}

    def load(self):
        """Load existing FAISS index and metadata from disk."""
        import faiss

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run scripts/build_faiss_index.py to build it."
            )

        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        return self

    def build(self, texts: list[str], sources: list[str], embeddings: np.ndarray):
        """
        Build a new FAISS index from text chunks and their embeddings.

        Args:
            texts: List of text chunks
            sources: List of source labels (e.g. 'EPA_SubpartW_2024.pdf')
            embeddings: np.ndarray of shape (n, 1024)
        """
        import faiss

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        dim = embeddings.shape[1]

        logger.info(f"Building FAISS index: {len(texts)} vectors, dim={dim}")
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vecs)
        self.index.add(embeddings.astype(np.float32))

        self.metadata = [
            {"text": t, "source": s, "chunk_id": i}
            for i, (t, s) in enumerate(zip(texts, sources))
        ]

        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Index saved to {self.index_path}")

    def search(self, query_embedding: np.ndarray, top_k: int = RAG_TOP_K) -> list[dict]:
        """
        Retrieve top-k most relevant chunks for a query embedding.

        Args:
            query_embedding: np.ndarray of shape (1024,)
            top_k: Number of results to return

        Returns:
            List of {"text": ..., "source": ..., "score": ...} dicts
        """
        if self.index is None:
            self.load()

        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "score": float(score),
            })

        return results
