
from pydantic import BaseModel, Field

from typing import Literal


class Evidence(BaseModel):
    source_id: str
    quote: str


class RAGDraft(BaseModel):
    answerable: bool
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)

class RAGQueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about AcmeOps documentation",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retrieved document chunks",
    )


class RAGCitation(BaseModel):
    source_id: str
    source: str
    quote: str


class RetrievedSource(BaseModel):
    source_id: str
    source: str
    text: str
    score: float


class RAGQueryResponse(BaseModel):
    question: str
    answer: str

    status: Literal[
        "evidence_verified",
        "insufficient_evidence",
        "validation_failed",
    ]

    citations: list[RAGCitation]
    retrieved_sources: list[RetrievedSource]
