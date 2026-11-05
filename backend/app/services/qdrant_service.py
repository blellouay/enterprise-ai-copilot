
from qdrant_client import QdrantClient

from app.core.config import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        timeout=5,
    )


def check_qdrant_connection() -> dict:
    client = get_qdrant_client()

    try:
        response = client.get_collections()

        return {
            "status": "connected",
            "collections_count": len(response.collections),
            "collections": [
                collection.name
                for collection in response.collections
            ],
        }
    finally:
        client.close()
