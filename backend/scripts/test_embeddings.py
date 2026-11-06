
import numpy as np
from pathlib import Path

from app.ai.embedding_service import EmbeddingService
from app.services.document_loader import load_markdown_documents
from app.services.markdown_analyzer import analyze_document
from app.services.document_chunker import document_level_chunking


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main():

    documents = load_markdown_documents(DOCUMENTS_DIR)

    if not documents:
        raise RuntimeError("No documents found")

    analyzed_documents = [
        analyze_document(document)
        for document in documents
    ]

    chunks = []

    for document in analyzed_documents:
        chunks.extend(document_level_chunking(document))

    texts = [chunk["text"] for chunk in chunks]

    print("Loading embedding model...")

    embedding_service = EmbeddingService()

    print("\nGenerating document embeddings...")

    document_embeddings = embedding_service.encode_texts(texts)

    print(f"\nDocuments: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding shape: {document_embeddings.shape}")

    print("\nFirst 10 embedding values:")
    print(document_embeddings[0][:10])

    query = "What happens after multiple failed login attempts?"

    print(f"\nQuery: {query}")

    query_embedding = embedding_service.encode_query(query)

    # Normalized vectors: dot product = cosine similarity
    similarities = document_embeddings @ query_embedding

    ranked_indices = np.argsort(similarities)[::-1]

    print("\nTOP 5 MOST SIMILAR DOCUMENTS\n")

    for rank, index in enumerate(ranked_indices[:5], start=1):

        chunk = chunks[index]
        score = similarities[index]

        print(f"Rank: {rank}")
        print(f"Source: {chunk['source']}")
        print(f"Similarity: {score:.4f}")
        print(f"Text: {chunk['text'][:180]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()
