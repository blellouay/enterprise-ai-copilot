
import uuid

import pytest

from app.services.advanced_chunker import AdvancedChunker


@pytest.fixture(scope="module")
def chunker():
    return AdvancedChunker(
        max_tokens=220,
        overlap_tokens=30,
    )


def make_document(content, source="docs/test.md"):

    return {
        "document_id": source,
        "source": source,
        "title": "Test Documentation",
        "sections": [{
            "heading": "Configuration",
            "heading_path": [
                "Documentation",
                "Configuration",
            ],
            "start_line": 10,
            "end_line": 500,
            "content": content,
        }],
    }


def assert_complete_coverage(chunks, original_text):
    """
    Check that:
    1. Every chunk matches its source character range.
    2. No characters are missing between ranges.
    3. The final range reaches the end of the source.
    """

    assert chunks

    cursor = 0

    for chunk in chunks:

        start = chunk["start_char"]
        end = chunk["end_char"]

        assert 0 <= start < end <= len(original_text)

        assert chunk["source_text"] == (
            original_text[start:end]
        )

        # Overlap is allowed, but gaps are not.
        assert start <= cursor

        cursor = max(cursor, end)

    assert cursor == len(original_text)


def test_fixed_chunks_preserve_text(chunker):

    content = (
        "Configure the editor using the settings panel.\n"
        "Authentication requires valid credentials.  \n"
        "\n"
    ) * 100

    document = make_document(content)

    chunks = chunker.fixed_chunking(
        document,
        content,
    )

    assert_complete_coverage(
        chunks,
        content,
    )


def test_hybrid_chunks_preserve_text(chunker):

    content = (
        "## Configuration\n\n"
        "Configure the editor using the settings panel.\n"
        "Authentication requires valid credentials.\n\n"
    ) * 100

    document = make_document(content)

    chunks = chunker.hybrid_chunking(document)

    assert_complete_coverage(
        chunks,
        content,
    )


def test_code_and_whitespace_preserved(chunker):

    content = (
        "## Python Example\n\n"
        "```python\n"
        "def authenticate(user):\n"
        "    return user.is_active\n"
        "```\n\n"
        "Settings panel:  Ctrl + Shift + P\n"
    )

    document = make_document(content)

    chunks = chunker.hybrid_chunking(document)

    assert_complete_coverage(
        chunks,
        content,
    )

    assert len(chunks) == 1

    assert chunks[0]["source_text"] == content


def test_ids_are_unique_across_documents(chunker):

    content = (
        "Configure authentication securely.\n" * 100
    )

    doc_a = make_document(
        content,
        source="docs/python/auth.md",
    )

    doc_b = make_document(
        content,
        source="docs/nodejs/auth.md",
    )

    chunks_a = chunker.hybrid_chunking(doc_a)
    chunks_b = chunker.hybrid_chunking(doc_b)

    ids_a = {c["chunk_id"] for c in chunks_a}
    ids_b = {c["chunk_id"] for c in chunks_b}

    assert len(ids_a) == len(chunks_a)
    assert len(ids_b) == len(chunks_b)

    assert ids_a.isdisjoint(ids_b)


def test_ids_are_deterministic(chunker):

    content = "Deployment requires approval.\n" * 50

    document = make_document(content)

    first_run = chunker.hybrid_chunking(document)
    second_run = chunker.hybrid_chunking(document)

    first_ids = [c["chunk_id"] for c in first_run]
    second_ids = [c["chunk_id"] for c in second_run]

    assert first_ids == second_ids

    for chunk_id in first_ids:
        uuid.UUID(chunk_id)


def test_fixed_and_hybrid_respect_token_limits(chunker):

    content = (
        "Configure the debugger and inspect variables.\n"
        * 150
    )

    document = make_document(content)

    fixed = chunker.fixed_chunking(
        document,
        content,
    )

    hybrid = chunker.hybrid_chunking(document)

    for chunk in fixed + hybrid:
        assert chunk["token_count"] <= 220


def test_heading_metadata_is_normalized(chunker):

    content = "Configure editor settings."

    document = make_document(content)

    document["sections"][0]["heading_path"] = [
        "{% data variables.product.prodname_vscode_shortname %}",
        "Configuration",
    ]

    chunks = chunker.hybrid_chunking(document)

    assert chunks[0]["heading_path"] == [
        "VS Code",
        "Configuration",
    ]


def test_source_line_numbers(chunker):

    content = (
        "Line one.\n"
        "Line two.\n"
        "Line three."
    )

    document = make_document(content)

    chunks = chunker.hybrid_chunking(document)

    assert len(chunks) == 1

    assert chunks[0]["start_line"] == 10
    assert chunks[0]["end_line"] == 12
