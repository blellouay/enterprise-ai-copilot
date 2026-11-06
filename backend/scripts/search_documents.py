
from app.ai.embedding_service import EmbeddingService
from app.ai.vector_store import VectorStore


def main():

    query = input("Enter your question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    print("\nLoading embedding model...")

    embedding_service = EmbeddingService()

    print("Generating query embedding...")

    query_vector = embedding_service.encode_query(query)

    print("\nSearching Qdrant...")

    vector_store = VectorStore()

    try:
        results = vector_store.search(
            query_vector=query_vector,
            limit=5,
        )
    finally:
        vector_store.close()

    print("\nTOP SEARCH RESULTS\n")

    for rank, result in enumerate(results, start=1):

        payload = result.payload or {}

        print(f"Rank: {rank}")
        print(f"Source: {payload.get('source')}")
        print(f"Title: {payload.get('title')}")
        print(f"Score: {result.score:.4f}")
        print(f"Text: {payload.get('text')}")
        print("-" * 50)


if __name__ == "__main__":
    main()
