"""
rag_pipeline.py
---------------
Full RAG chain: retrieve relevant document chunks → inject into prompt → generate.
Mirrors the production pipeline used in the OilGasAI platform.
"""

from loguru import logger
from .embeddings import embed_query
from .faiss_store import FAISSStore
from inference.chat import chat, SYSTEM_PROMPT


RAG_SYSTEM_PROMPT = """{base_system}

You have access to the following retrieved context from official regulations, 
research papers, and engineering documentation. Use this context to ground your 
response in accurate, current information. Always cite the source when using context.

--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---
"""


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for OilGasAI Model Alpha.

    Combines FAISS retrieval with OilGasAI Model Alpha inference for
    document-grounded compliance and technical responses.
    """

    def __init__(self, index_path: str | None = None):
        self.store = FAISSStore(index_path) if index_path else FAISSStore()
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.store.load()
            self._loaded = True

    def query(
        self,
        question: str,
        backend: dict,
        top_k: int = 5,
        return_sources: bool = False,
    ) -> dict:
        """
        Run a RAG query: retrieve context → generate grounded answer.

        Args:
            question: User question
            backend: Dict from get_inference_backend()
            top_k: Number of context chunks to retrieve
            return_sources: Whether to include retrieved sources in output

        Returns:
            dict with 'answer', and optionally 'sources'
        """
        self._ensure_loaded()

        # 1. Embed query
        query_vec = embed_query(question)

        # 2. Retrieve context
        chunks = self.store.search(query_vec, top_k=top_k)
        logger.info(f"Retrieved {len(chunks)} chunks (top score: {chunks[0]['score']:.3f})")

        # 3. Format context block
        context_lines = []
        for i, chunk in enumerate(chunks, 1):
            context_lines.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
        context_str = "\n\n".join(context_lines)

        # 4. Build RAG-augmented system prompt
        augmented_system = RAG_SYSTEM_PROMPT.format(
            base_system=SYSTEM_PROMPT,
            context=context_str,
        )

        # 5. Generate response
        answer = chat(question, backend, system_prompt=augmented_system)

        result = {"answer": answer}
        if return_sources:
            result["sources"] = [
                {"source": c["source"], "score": c["score"], "excerpt": c["text"][:200]}
                for c in chunks
            ]

        return result
