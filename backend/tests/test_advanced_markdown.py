
from pathlib import Path

from app.services.advanced_markdown_processor import (
    AdvancedMarkdownProcessor,
)


def test_markdown_processing(tmp_path: Path):
    """
    Test normal Markdown processing:
    - YAML front matter
    - Markdown headings
    - Code blocks
    - Heading hierarchy
    - No empty introduction
    """

    content = """---
title: Example Documentation
---

# Main Heading

Introduction text.

## Python Example

```python
# Not a Markdown heading
print("Hello")
```

More explanation.
"""

    file_path = tmp_path / "example.md"

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    processor = AdvancedMarkdownProcessor()

    result = processor.process_document(
        file_path=file_path,
        dataset_root=tmp_path,
    )

    assert result["title"] == "Example Documentation"

    # Two meaningful sections, no empty introduction
    assert result["section_count"] == 2

    assert result["code_block_count"] == 1

    assert len(result["headings"]) == 2

    assert result["headings"][0]["title"] == "Main Heading"

    assert result["headings"][1]["title"] == "Python Example"

    assert (
        "# Not a Markdown heading"
        in result["sections"][1]["content"]
    )

    assert result["sections"][1]["heading_path"] == [
        "Main Heading",
        "Python Example",
    ]


def test_invalid_yaml_frontmatter(tmp_path: Path):
    """
    Test that GitHub template expressions
    do not stop document processing.
    """

    content = """---
title: Example
related:
  - {% data variables.copilot.agents_example %}
---

# Authentication

This document explains authentication.
"""

    file_path = tmp_path / "invalid.md"

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    processor = AdvancedMarkdownProcessor()

    result = processor.process_document(
        file_path=file_path,
        dataset_root=tmp_path,
    )

    assert "_parse_error" in result["metadata"]

    assert "_raw_frontmatter" in result["metadata"]

    # No empty introduction from blank lines
    assert result["section_count"] == 1

    assert result["headings"][0]["title"] == "Authentication"

    assert (
        "This document explains authentication."
        in result["sections"][0]["content"]
    )


def test_document_without_frontmatter(tmp_path: Path):
    """
    Test Markdown without YAML metadata.
    """

    content = """# Deployment

## Rollback

Failed deployments trigger rollback.
"""

    file_path = tmp_path / "deploy.md"

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    processor = AdvancedMarkdownProcessor()

    result = processor.process_document(
        file_path=file_path,
        dataset_root=tmp_path,
    )

    assert result["metadata"] == {}

    assert result["title"] == "Deployment"

    assert result["section_count"] == 2

    assert result["code_block_count"] == 0


def test_real_introduction(tmp_path: Path):
    """
    Verify that meaningful introductory content
    is preserved as its own section.
    """

    content = """This document explains the system.

# Main Heading

Some information.
"""

    file_path = tmp_path / "introduction.md"

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    processor = AdvancedMarkdownProcessor()

    result = processor.process_document(
        file_path=file_path,
        dataset_root=tmp_path,
    )

    assert result["section_count"] == 2

    assert result["sections"][0]["heading"] == "Introduction"

    assert (
        "This document explains the system."
        in result["sections"][0]["content"]
    )


def test_document_without_headings(tmp_path: Path):
    """
    Verify that documents containing only
    paragraphs are still processed.
    """

    content = """This document has no headings.

It contains two paragraphs.
"""

    file_path = tmp_path / "no_headings.md"

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    processor = AdvancedMarkdownProcessor()

    result = processor.process_document(
        file_path=file_path,
        dataset_root=tmp_path,
    )

    assert result["section_count"] == 1

    assert result["headings"] == []

    assert result["sections"][0]["heading"] == "no_headings"


def test_empty_document(tmp_path: Path):
    """
    An empty Markdown document should
    not create meaningless sections.
    """

    file_path = tmp_path / "empty.md"

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    processor = AdvancedMarkdownProcessor()

    result = processor.process_document(
        file_path=file_path,
        dataset_root=tmp_path,
    )

    assert result["section_count"] == 0

    assert result["headings"] == []

    assert result["code_block_count"] == 0
