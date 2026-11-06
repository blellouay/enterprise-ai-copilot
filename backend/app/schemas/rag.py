
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source_id: str
    quote: str


class RAGDraft(BaseModel):
    answerable: bool
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
