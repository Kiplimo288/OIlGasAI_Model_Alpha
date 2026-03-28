"""
routes/chat.py
--------------
General-purpose domain chat endpoint.
"""

from fastapi import APIRouter, Depends
from api.schemas import ChatRequest, ChatResponse
from api.main import get_backend, get_rag
from inference.chat import chat, chat_with_history

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    backend: dict = Depends(get_backend),
    rag: object = Depends(get_rag),
):
    """
    Chat with OilGasAI-Model-Alpha.

    Optionally uses RAG pipeline for document-grounded responses.
    Set use_rag=false for pure model inference.
    """
    if request.use_rag:
        result = rag.query(
            question=request.message,
            backend=backend,
            top_k=request.top_k,
            return_sources=True,
        )
        return ChatResponse(answer=result["answer"], sources=result.get("sources"))
    else:
        if request.history:
            messages = [{"role": m.role, "content": m.content} for m in request.history]
            messages.append({"role": "user", "content": request.message})
            answer = chat_with_history(messages, backend)
        else:
            answer = chat(request.message, backend)

        return ChatResponse(answer=answer)
