
from qdrant_client import QdrantClient, models

from app.core.config import settings


COLLECTION_NAME = "acmeops_documents_minilm"
VECTOR_SIZE = 384


class VectorStore:

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            timeout=10,
        )

    def create_collection(self):

        if self.client.collection_exists(COLLECTION_NAME):
            print("Collection already exists")
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        print("Collection created successfully")

    def insert_chunks(self, chunks, embeddings):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings must have equal lengths"
            )

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings),
            start=1,
        ):
            point = models.PointStruct(
                id=index,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"Inserted {len(points)} points")

    def search(self, query_vector, limit=5):

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )

        return response.points

    def close(self):
        self.client.close()
