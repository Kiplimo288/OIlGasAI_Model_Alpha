"""
build_faiss_index.py
--------------------
Build or rebuild the FAISS vector index for the RAG pipeline.
Run this script whenever you add new documents to the knowledge base.

Usage:
    python scripts/build_faiss_index.py --docs_dir ./docs/knowledge_base
    python scripts/build_faiss_index.py --docs_dir ./docs/knowledge_base --chunk_size 512
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv
from rag.embeddings import embed_texts
from rag.faiss_store import FAISSStore

load_dotenv()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def load_documents(docs_dir: str) -> tuple[list[str], list[str]]:
    """
    Load all .txt and .md documents from a directory.
    Returns (texts, sources) parallel lists.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    texts, sources = [], []
    supported = [".txt", ".md"]

    for fpath in sorted(docs_path.rglob("*")):
        if fpath.suffix.lower() not in supported:
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(content)
            texts.extend(chunks)
            sources.extend([fpath.name] * len(chunks))
            logger.info(f"  {fpath.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")

    return texts, sources


def main():
    parser = argparse.ArgumentParser(description="Build OilGasAI FAISS vector index")
    parser.add_argument(
        "--docs_dir",
        default="./docs/knowledge_base",
        help="Directory containing knowledge base documents",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--output",
        default=os.getenv("FAISS_INDEX_PATH", "./rag/index/oilgas_faiss.index"),
    )
    args = parser.parse_args()

    logger.info(f"Loading documents from: {args.docs_dir}")
    texts, sources = load_documents(args.docs_dir)
    logger.info(f"Total chunks: {len(texts)}")

    logger.info("Generating embeddings with BGE-large-en-v1.5...")
    import numpy as np
    from tqdm import tqdm

    batch_size = 128
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        emb = embed_texts(batch)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    logger.info(f"Building FAISS index → {args.output}")
    store = FAISSStore(index_path=args.output)
    store.build(texts, sources, embeddings)

    logger.info(f"Done. Index contains {store.index.ntotal} vectors.")
    logger.info("Run `uvicorn api.main:app --reload` to start the API.")


if __name__ == "__main__":
    main()
