
from app.schemas.rag import RAGDraft


ABSTENTION_MESSAGE = (
    "I cannot determine that from "
    "the available documentation."
)


def validate_rag_draft(
    draft: RAGDraft,
    retrieved_sources: list[dict],
) -> dict:
    """
    Validate source references and exact quotations.

    This verifies evidence existence, not whether
    the quotation logically entails the answer.
    """

    source_lookup = {
        source["source_id"]: source
        for source in retrieved_sources
    }

    if not draft.answerable or not draft.answer.strip():
        return {
            "status": "insufficient_evidence",
            "answer": ABSTENTION_MESSAGE,
            "citations": [],
        }

    if not draft.evidence:
        return {
            "status": "validation_failed",
            "answer": ABSTENTION_MESSAGE,
            "citations": [],
        }

    validated_citations = []

    for item in draft.evidence:

        source = source_lookup.get(item.source_id)

        if source is None:
            return {
                "status": "validation_failed",
                "answer": ABSTENTION_MESSAGE,
                "citations": [],
            }

        quote = item.quote.strip()

        if not quote or quote not in source["text"]:
            return {
                "status": "validation_failed",
                "answer": ABSTENTION_MESSAGE,
                "citations": [],
            }

        validated_citations.append({
            "source_id": item.source_id,
            "source": source["source"],
            "quote": quote,
        })

    return {
        "status": "evidence_verified",
        "answer": draft.answer.strip(),
        "citations": validated_citations,
    }
