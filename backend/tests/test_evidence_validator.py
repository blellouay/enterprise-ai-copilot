
from app.ai.evidence_validator import (
    validate_rag_draft,
    ABSTENTION_MESSAGE,
)
from app.schemas.rag import RAGDraft, Evidence


SOURCES = [
    {
        "source_id": "S1",
        "source": "auth.md",
        "text": "Accounts are locked for 15 minutes.",
    }
]


def test_valid_evidence():

    draft = RAGDraft(
        answerable=True,
        answer="Accounts are locked for 15 minutes.",
        evidence=[
            Evidence(
                source_id="S1",
                quote="Accounts are locked for 15 minutes.",
            )
        ],
    )

    result = validate_rag_draft(draft, SOURCES)

    assert result["status"] == "evidence_verified"
    assert len(result["citations"]) == 1


def test_fabricated_quote():

    draft = RAGDraft(
        answerable=True,
        answer="Accounts are locked for 60 minutes.",
        evidence=[
            Evidence(
                source_id="S1",
                quote="Accounts are locked for 60 minutes.",
            )
        ],
    )

    result = validate_rag_draft(draft, SOURCES)

    assert result["status"] == "validation_failed"
    assert result["answer"] == ABSTENTION_MESSAGE


def test_invalid_source():

    draft = RAGDraft(
        answerable=True,
        answer="An account is locked.",
        evidence=[
            Evidence(
                source_id="S99",
                quote="An account is locked.",
            )
        ],
    )

    result = validate_rag_draft(draft, SOURCES)

    assert result["status"] == "validation_failed"


def test_unanswerable_question():

    draft = RAGDraft(
        answerable=False,
        answer="",
        evidence=[],
    )

    result = validate_rag_draft(draft, SOURCES)

    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
