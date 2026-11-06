
from app.services.advanced_chunker import AdvancedChunker


def test_hybrid_chunking_respects_token_limit():

    chunker = AdvancedChunker(
        max_tokens=220,
        overlap_tokens=30,
    )

    section = {
        "heading": "Authentication",
        "heading_path": [
            "Security",
            "Authentication",
        ],
        "start_line": 10,
        "end_line": 400,
        "content": (
            "Authentication requires valid credentials.\n"
            * 200
        ),
    }

    document = {
        "document_id": "test.md",
        "source": "test.md",
        "title": "Security",
        "sections": [section],
    }

    chunks = chunker.hybrid_chunking(document)

    assert len(chunks) > 1

    assert all(
        chunk["token_count"] <= 220
        for chunk in chunks
    )

    assert all(
        chunk["heading_path"] == [
            "Security",
            "Authentication",
        ]
        for chunk in chunks
    )


def test_short_section_remains_one_chunk():

    chunker = AdvancedChunker()

    document = {
        "document_id": "short.md",
        "source": "short.md",
        "title": "Deployment",
        "sections": [{
            "heading": "Rollback",
            "heading_path": [
                "Deployment",
                "Rollback",
            ],
            "start_line": 1,
            "end_line": 3,
            "content": (
                "## Rollback\n"
                "Failed deployments trigger rollback."
            ),
        }],
    }

    chunks = chunker.hybrid_chunking(document)

    assert len(chunks) == 1
    assert chunks[0]["token_count"] <= 220
    assert chunks[0]["start_line"] == 1
