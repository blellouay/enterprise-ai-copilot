
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from app.ai.rag_service import RAGService
from app.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
)


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService()


@router.post(
    "/query",
    response_model=RAGQueryResponse,
)
def query_documents(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank",
        )

    result = rag_service.answer_question(
        question=request.question,
        top_k=request.top_k,
    )

    return result
