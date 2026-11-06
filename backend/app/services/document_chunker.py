
import re


def create_chunk(
    document: dict,
    text: str,
    chunk_index: int,
    section: str | None = None,
) -> dict:
    """Create a chunk while preserving source metadata."""

    return {
        "chunk_id": f"{document['document_id']}_{chunk_index}",
        "document_id": document["document_id"],
        "source": document["filename"],
        "title": document["title"],
        "section": section,
        "text": text,
        "character_count": len(text),
    }


def document_level_chunking(
    document: dict,
) -> list[dict]:
    """Keep a short document together as one chunk."""

    return [
        create_chunk(
            document=document,
            text=document["content"].strip(),
            chunk_index=0,
        )
    ]


def split_long_text(
    text: str,
    max_chars: int = 1000,
) -> list[str]:
    """
    Split long text into bounded passages.

    Prefer whitespace boundaries.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    words = text.split()
    chunks = []
    current = ""

    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(
                word[i:i + max_chars]
                for i in range(0, len(word), max_chars)
            )
            continue

        candidate = f"{current} {word}".strip()

        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks


def hybrid_chunking(
    document: dict,
    max_chars: int = 1000,
) -> list[dict]:
    """
    Split at Markdown headings, then split
    sections that exceed max_chars.
    """

    content = document["content"]

    sections = re.split(
        r"(?=^#{1,6}\s+)",
        content,
        flags=re.MULTILINE,
    )

    chunks = []

    for section_text in sections:
        section_text = section_text.strip()

        if not section_text:
            continue

        first_line = section_text.splitlines()[0]

        if first_line.startswith("#"):
            section_title = first_line.lstrip("#").strip()
        else:
            section_title = document["title"]

        passages = split_long_text(
            section_text,
            max_chars=max_chars,
        )

        for passage in passages:
            chunks.append(
                create_chunk(
                    document=document,
                    text=passage,
                    chunk_index=len(chunks),
                    section=section_title,
                )
            )

    return chunks
