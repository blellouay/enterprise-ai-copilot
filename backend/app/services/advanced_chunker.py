
from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class AdvancedChunker:

    def __init__(
        self,
        max_tokens: int = 220,
        overlap_tokens: int = 30,
    ):
        if max_tokens < 16:
            raise ValueError("max_tokens must be >= 16")

        if overlap_tokens < 0:
            raise ValueError("Overlap cannot be negative")

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )

        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required for offsets"
            )

    def token_count(self, text: str) -> int:
        """Count tokens including model special tokens."""

        return len(self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        ))

    def split_token_windows(
        self,
        text: str,
        budget: int,
    ) -> list[tuple[str, int, int]]:
        """
        Return (text, start_char, end_char) windows.

        Character offsets refer to the original input.
        """

        if budget <= self.overlap_tokens:
            raise ValueError(
                "Budget must exceed overlap_tokens"
            )

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )

        offsets = [
            (start, end)
            for start, end in encoded["offset_mapping"]
            if end > start
        ]

        if not offsets:
            return []

        windows = []
        start_index = 0

        while start_index < len(offsets):

            end_index = min(
                start_index + budget,
                len(offsets),
            )

            start_char = offsets[start_index][0]
            end_char = offsets[end_index - 1][1]

            passage = text[start_char:end_char]

            if passage.strip():
                windows.append((
                    passage,
                    start_char,
                    end_char,
                ))

            if end_index == len(offsets):
                break

            start_index = (
                end_index - self.overlap_tokens
            )

        return windows

    def make_chunk(
        self,
        document: dict,
        text: str,
        strategy: str,
        index: int,
        section: dict | None = None,
        start_offset: int = 0,
    ) -> dict:

        if section is not None:
            source_start = section["start_line"]
            source_start += section["content"][
                :start_offset
            ].count("\n")

            source_end = source_start + (
                text.count("\n")
                - int(text.endswith("\n"))
            )
        else:
            source_start = None
            source_end = None

        return {
            "chunk_id": f"{strategy}_{index}",
            "document_id": document["document_id"],
            "source": document["source"],
            "title": document["title"],
            "strategy": strategy,
            "heading_path": (
                section["heading_path"]
                if section else []
            ),
            "start_line": source_start,
            "end_line": source_end,
            "text": text,
            "token_count": self.token_count(text),
        }

    def fixed_chunking(
        self,
        document: dict,
        original_text: str,
    ) -> list[dict]:

        # Reserve 2 tokens for MiniLM special tokens.
        budget = self.max_tokens - 2

        windows = self.split_token_windows(
            original_text,
            budget=budget,
        )

        return [
            self.make_chunk(
                document=document,
                text=passage,
                strategy="fixed",
                index=index,
            )
            for index, (passage, _, _) in enumerate(windows)
        ]

    def section_chunking(
        self,
        document: dict,
    ) -> list[dict]:
        """One chunk per meaningful Markdown section."""

        chunks = []

        for section in document["sections"]:

            text = section["content"]

            if not text.strip():
                continue

            chunks.append(
                self.make_chunk(
                    document=document,
                    text=text,
                    strategy="section",
                    index=len(chunks),
                    section=section,
                )
            )

        return chunks

    def hybrid_chunking(
        self,
        document: dict,
    ) -> list[dict]:
        """Split oversized sections with heading context."""

        chunks = []

        for section in document["sections"]:

            content = section["content"]

            if not content.strip():
                continue

            heading_path = section["heading_path"]

            prefix = (
                "Section: " + " > ".join(heading_path) + "\n\n"
                if heading_path else ""
            )

            # Keep enough room for content, including
            # the model's special tokens.
            prefix_tokens = self.token_count(prefix) - 2

            budget = (
                self.max_tokens
                - prefix_tokens
                - 2
                - 4
            )

            if budget <= self.overlap_tokens:
                raise ValueError(
                    "Heading path leaves insufficient "
                    "space for chunk content"
                )

            windows = self.split_token_windows(
                content,
                budget=budget,
            )

            for passage, start_char, _ in windows:

                chunk_text = prefix + passage

                # Defensive verification against
                # tokenizer boundary effects.
                if self.token_count(chunk_text) > self.max_tokens:
                    raise RuntimeError(
                        "Generated chunk exceeds token limit"
                    )

                chunk = self.make_chunk(
                    document=document,
                    text=chunk_text,
                    strategy="hybrid",
                    index=len(chunks),
                    section=section,
                    start_offset=start_char,
                )

                # The prefix is generated metadata,
                # not part of the original source lines.
                source_text = passage
                chunk["end_line"] = (
                    chunk["start_line"]
                    + source_text.count("\n")
                    - int(source_text.endswith("\n"))
                )

                chunks.append(chunk)

        return chunks
