
import re


HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$",
    re.MULTILINE,
)


def extract_headings(content: str) -> list[dict]:
    """
    Extract Markdown headings with their levels.
    """

    headings = []

    for match in HEADING_PATTERN.finditer(content):

        level = len(match.group(1))
        title = match.group(2).strip()

        headings.append({
            "level": level,
            "title": title,
        })

    return headings


def analyze_document(document: dict) -> dict:
    """
    Add structural metadata to a document.
    """

    content = document["content"]

    headings = extract_headings(content)

    main_title = (
        headings[0]["title"]
        if headings
        else document["filename"]
    )

    return {
        **document,
        "document_type": "technical_documentation",
        "title": main_title,
        "headings": headings,
        "heading_count": len(headings),
        "word_count": len(content.split()),
        "character_count": len(content),
    }
