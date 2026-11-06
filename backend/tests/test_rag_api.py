
from fastapi.testclient import TestClient

from app.main import app
from app.api.routers.rag import get_rag_service


class FakeRAGService:
    def answer_question(self, question, top_k=3):
        return {
            "question": question,
            "answer": "Deployment failure triggers rollback.",
            "status": "evidence_verified",
            "citations": [
                {
                    "source_id": "S1",
                    "source": "deploy.md",
                    "quote": "Deployment failure triggers rollback.",
                }
            ],
            "retrieved_sources": [],
        }


def override_rag_service():
    return FakeRAGService()


app.dependency_overrides[get_rag_service] = override_rag_service

client = TestClient(app)


def test_rag_query_success():
    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": "What happens when deployment fails?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "evidence_verified"
    assert data["citations"][0]["source"] == "deploy.md"


def test_empty_question():
    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_invalid_top_k():
    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": "What is RAG?",
            "top_k": 100,
        },
    )

    assert response.status_code == 422


def test_missing_question():
    response = client.post(
        "/api/v1/rag/query",
        json={"top_k": 3},
    )

    assert response.status_code == 422
