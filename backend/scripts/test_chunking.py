
from pathlib import Path

from app.services.document_loader import load_markdown_documents
from app.services.markdown_analyzer import analyze_document

from app.services.document_chunker import (
    document_level_chunking,
    hybrid_chunking,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main():

    documents = load_markdown_documents(DOCUMENTS_DIR)

    if not documents:
        raise RuntimeError("No Markdown documents found")

    analyzed_documents = [
        analyze_document(doc)
        for doc in documents
    ]

    baseline_chunks = []
    hybrid_chunks = []

    for document in analyzed_documents:

        baseline_chunks.extend(
            document_level_chunking(document)
        )

        hybrid_chunks.extend(
            hybrid_chunking(document, max_chars=1000)
        )

    print("DOCUMENT CHUNKING COMPARISON")
    print("-" * 45)

    print(f"Original documents: {len(documents)}")
    print(f"Document-level chunks: {len(baseline_chunks)}")
    print(f"Hybrid chunks: {len(hybrid_chunks)}")

    print("\nCHUNK LENGTHS")

    for name, chunks in [
        ("Document-level", baseline_chunks),
        ("Hybrid", hybrid_chunks),
    ]:
        lengths = [len(c["text"]) for c in chunks]

        print(
            f"{name}: "
            f"min={min(lengths)}, "
            f"max={max(lengths)}, "
            f"avg={sum(lengths) / len(lengths):.1f}"
        )

    print("\nEXAMPLE: AUTHENTICATION DOCUMENT")

    for chunk in hybrid_chunks:
        if chunk["document_id"] == "auth":
            print(f"\nChunk ID: {chunk['chunk_id']}")
            print(f"Section: {chunk['section']}")
            print(f"Text:\n{chunk['text']}")


if __name__ == "__main__":
    main()
