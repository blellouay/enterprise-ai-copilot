
from pathlib import Path

import yaml
from markdown_it import MarkdownIt


class AdvancedMarkdownProcessor:

    def __init__(self):
        self.parser = MarkdownIt("commonmark")

    def extract_frontmatter(self, content: str):
        """
        Extract YAML metadata while tolerating
        invalid GitHub template expressions.

        Returns:
            metadata, markdown_body, line_offset
        """

        lines = content.splitlines(keepends=True)

        if not lines or lines[0].strip() != "---":
            return {}, content, 0

        closing_index = None

        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                closing_index = index
                break

        if closing_index is None:
            return {}, content, 0

        metadata_text = "".join(lines[1:closing_index])

        try:
            metadata = yaml.safe_load(metadata_text) or {}

            if not isinstance(metadata, dict):
                raise ValueError(
                    "YAML front matter must contain a mapping"
                )

        except (yaml.YAMLError, ValueError) as exc:
            metadata = {
                "_parse_error": str(exc),
                "_raw_frontmatter": metadata_text,
            }

        body = "".join(lines[closing_index + 1:])

        # Number of removed front matter lines
        offset = closing_index + 1

        return metadata, body, offset

    def process_document(
        self,
        file_path: Path,
        dataset_root: Path,
    ) -> dict:
        """
        Process one Markdown file.

        Extracts:
        - YAML metadata
        - Document title
        - Headings and hierarchy
        - Sections
        - Line numbers
        - Code block counts
        """

        content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        metadata, body, line_offset = (
            self.extract_frontmatter(content)
        )

        tokens = self.parser.parse(body)

        lines = body.splitlines(keepends=True)

        # ----------------------------------
        # 1. Extract headings
        # ----------------------------------

        headings = []

        for index, token in enumerate(tokens):

            if token.type != "heading_open":
                continue

            level = int(token.tag[1:])

            title = ""

            if index + 1 < len(tokens):
                inline_token = tokens[index + 1]

                if inline_token.type == "inline":
                    title = inline_token.content

            line_number = (
                token.map[0] + line_offset + 1
                if token.map
                else None
            )

            headings.append({
                "level": level,
                "title": title,
                "line": line_number,
            })

        # ----------------------------------
        # 2. Count code blocks
        # ----------------------------------

        code_blocks = [
            token
            for token in tokens
            if token.type in ("fence", "code_block")
        ]

        # ----------------------------------
        # 3. Identify heading tokens
        # ----------------------------------

        sections = []

        heading_tokens = [
            (index, token)
            for index, token in enumerate(tokens)
            if (
                token.type == "heading_open"
                and token.map is not None
            )
        ]

        # ----------------------------------
        # 4. Documents without headings
        # ----------------------------------

        if not heading_tokens:

            # Avoid creating empty sections
            if body.strip():

                sections.append({
                    "heading": str(
                        metadata.get("title")
                        or file_path.stem
                    ),
                    "heading_path": [],
                    "level": 0,
                    "start_line": line_offset + 1,
                    "end_line": line_offset + len(lines),
                    "content": body,
                })

        # ----------------------------------
        # 5. Documents containing headings
        # ----------------------------------

        else:

            first_start = heading_tokens[0][1].map[0]

            introduction_text = "".join(
                lines[:first_start]
            )

            # FIX:
            # Ignore introductions containing
            # only blank lines or whitespace.
            if introduction_text.strip():

                sections.append({
                    "heading": "Introduction",
                    "heading_path": [],
                    "level": 0,
                    "start_line": line_offset + 1,
                    "end_line": line_offset + first_start,
                    "content": introduction_text,
                })

            hierarchy = []

            for position, (token_index, token) in enumerate(
                heading_tokens
            ):

                level = int(token.tag[1:])

                inline_token = tokens[token_index + 1]
                title = inline_token.content

                # Remove headings at the same
                # or deeper hierarchy level.
                hierarchy = [
                    item
                    for item in hierarchy
                    if item["level"] < level
                ]

                hierarchy.append({
                    "level": level,
                    "title": title,
                })

                start = token.map[0]

                if position + 1 < len(heading_tokens):

                    end = heading_tokens[
                        position + 1
                    ][1].map[0]

                else:
                    end = len(lines)

                section_content = "".join(
                    lines[start:end]
                )

                if not section_content.strip():
                    continue

                sections.append({
                    "heading": title,
                    "heading_path": [
                        item["title"]
                        for item in hierarchy
                    ],
                    "level": level,
                    "start_line": start + line_offset + 1,
                    "end_line": end + line_offset,
                    "content": section_content,
                })

        # ----------------------------------
        # 6. Build document metadata
        # ----------------------------------

        source = file_path.relative_to(
            dataset_root
        ).as_posix()

        document_title = str(
            metadata.get("title")
            or (
                headings[0]["title"]
                if headings
                else file_path.stem
            )
        )

        return {
            "document_id": source,
            "source": source,
            "title": document_title,
            "metadata": metadata,
            "headings": headings,
            "sections": sections,
            "section_count": len(sections),
            "code_block_count": len(code_blocks),
            "character_count": len(content),
            "word_count": len(content.split()),
        }

