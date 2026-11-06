
from pathlib import Path

from app.services.document_loader import load_markdown_documents
from app.services.markdown_analyzer import analyze_document
from app.services.document_chunker import document_level_chunking

from app.ai.embedding_service import EmbeddingService
from app.ai.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main():

    print("Step 1: Loading documents")

    documents = load_markdown_documents(DOCUMENTS_DIR)

    if len(documents) != 12:
        raise RuntimeError(
            f"Expected 12 documents, found {len(documents)}"
        )

    print(f"Loaded {len(documents)} documents")

    print("\nStep 2: Analyzing documents")

    analyzed_documents = [
        analyze_document(doc)
        for doc in documents
    ]

    print("\nStep 3: Creating chunks")

    chunks = []

    for document in analyzed_documents:
        chunks.extend(document_level_chunking(document))

    print(f"Created {len(chunks)} chunks")

    print("\nStep 4: Generating embeddings")

    embedding_service = EmbeddingService()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_service.encode_texts(texts)

    print(f"Embedding shape: {embeddings.shape}")

    print("\nStep 5: Storing embeddings in Qdrant")

    vector_store = VectorStore()

    try:
        vector_store.create_collection()
        vector_store.insert_chunks(chunks, embeddings)

        info = vector_store.client.get_collection(
            collection_name="acmeops_documents_minilm"
        )

        print(f"Stored points: {info.points_count}")
    finally:
        vector_store.close()

    print("\nIndexing completed!")


if __name__ == "__main__":
    main()
