
import uuid

from transformers import AutoTokenizer

from app.services.template_normalizer import TemplateNormalizer


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
                "A fast tokenizer is required"
            )

        self.special_tokens = (
            self.tokenizer.num_special_tokens_to_add(
                pair=False
            )
        )

        if max_tokens <= (
            self.special_tokens + overlap_tokens
        ):
            raise ValueError(
                "Not enough room for tokens and overlap"
            )

        self.template_normalizer = TemplateNormalizer()

    def normalize_templates(self, text: str) -> str:
        return self.template_normalizer.normalize(text)

    def find_unresolved_templates(
        self,
        text: str,
    ) -> list[str]:
        return self.template_normalizer.unresolved(text)

    def token_count(self, text: str) -> int:

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )

        return len(encoded["input_ids"])

    def generate_chunk_id(
        self,
        document_id: str,
        strategy: str,
        index: int,
        start_char: int,
        end_char: int,
    ) -> str:
        """
        Generate a deterministic, globally namespaced UUID.
        """

        key = (
            f"{document_id}|{strategy}|{index}|"
            f"{start_char}|{end_char}"
        )

        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, key)
        )

    def split_token_windows(
        self,
        text: str,
        budget: int,
        prefix: str = "",
    ) -> list[tuple[str, int, int]]:
        """
        Split text without losing whitespace.

        Returns:
            (passage, start_char, end_char)

        Each interval is [start_char, end_char).

        Character offsets refer to the normalized text
        passed to this method.
        """

        if budget <= self.overlap_tokens:
            raise ValueError(
                "Budget must exceed overlap"
            )

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )

        offsets = [
            (start, end)
            for start, end in encoded["offset_mapping"]
            if end > start
        ]

        if not offsets:
            if text.strip():
                raise ValueError(
                    "Tokenizer returned no usable offsets"
                )
            return []

        windows = []
        start_index = 0

        while start_index < len(offsets):

            end_index = min(
                start_index + budget,
                len(offsets),
            )

            # Include whitespace before the first token.
            start_char = (
                0 if start_index == 0
                else offsets[start_index][0]
            )

            # Find the end of the next token window.
            # Using the next token's start position
            # preserves separating whitespace.
            while end_index > start_index:

                end_char = (
                    offsets[end_index][0]
                    if end_index < len(offsets)
                    else len(text)
                )

                passage = text[start_char:end_char]

                if (
                    self.token_count(prefix + passage)
                    <= self.max_tokens
                ):
                    break

                end_index -= 1

            if end_index <= start_index:
                raise RuntimeError(
                    "Unable to fit a token window "
                    "within max_tokens"
                )

            if not passage.strip():
                raise RuntimeError(
                    "Unexpected whitespace-only token window"
                )

            windows.append((
                passage,
                start_char,
                end_char,
            ))

            if end_index == len(offsets):
                break

            # Overlap is measured in tokens.
            # Guarantee forward progress.
            start_index = max(
                start_index + 1,
                end_index - self.overlap_tokens,
            )

        return windows

    def make_chunk(
        self,
        document: dict,
        strategy: str,
        index: int,
        text: str,
        source_text: str,
        start_char: int,
        end_char: int,
        section: dict | None = None,
    ) -> dict:

        count = self.token_count(text)

        if (
            strategy != "section"
            and count > self.max_tokens
        ):
            raise RuntimeError(
                f"Token limit exceeded: {count}"
            )

        if section is not None:

            # The normalized text preserves line breaks.
            normalized_section = self.normalize_templates(
                section["content"]
            )

            start_line = (
                section["start_line"]
                + normalized_section[
                    :start_char
                ].count("\n")
            )

            end_line = (
                section["start_line"]
                + normalized_section[
                    :max(end_char - 1, start_char)
                ].count("\n")
            )

        else:
            start_line = None
            end_line = None

        return {
            "chunk_id": self.generate_chunk_id(
                document["document_id"],
                strategy,
                index,
                start_char,
                end_char,
            ),
            "document_id": document["document_id"],
            "source": document["source"],
            "title": document["title"],
            "strategy": strategy,
            "heading_path": (
                [
                    self.normalize_templates(heading)
                    for heading in section["heading_path"]
                ]
                if section else []
            ),
            "start_char": start_char,
            "end_char": end_char,
            "start_line": start_line,
            "end_line": end_line,
            "source_text": source_text,
            "text": text,
            "token_count": count,
            "unresolved_templates": (
                self.find_unresolved_templates(text)
            ),
        }

    def fixed_chunking(
        self,
        document: dict,
        original_text: str,
    ) -> list[dict]:

        text = self.normalize_templates(original_text)

        budget = (
            self.max_tokens - self.special_tokens
        )

        windows = self.split_token_windows(
            text=text,
            budget=budget,
        )

        chunks = []

        for index, (
            passage,
            start_char,
            end_char,
        ) in enumerate(windows):

            chunks.append(
                self.make_chunk(
                    document=document,
                    strategy="fixed",
                    index=index,
                    text=passage,
                    source_text=passage,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return chunks

    def section_chunking(
        self,
        document: dict,
    ) -> list[dict]:

        chunks = []

        for section in document["sections"]:

            text = self.normalize_templates(
                section["content"]
            )

            if not text.strip():
                continue

            chunks.append(
                self.make_chunk(
                    document=document,
                    strategy="section",
                    index=len(chunks),
                    text=text,
                    source_text=text,
                    start_char=0,
                    end_char=len(text),
                    section=section,
                )
            )

        return chunks

    def hybrid_chunking(
        self,
        document: dict,
    ) -> list[dict]:

        chunks = []

        for section in document["sections"]:

            content = self.normalize_templates(
                section["content"]
            )

            if not content.strip():
                continue

            heading_path = [
                self.normalize_templates(heading)
                for heading in section["heading_path"]
            ]

            prefix = (
                "Section: "
                + " > ".join(heading_path)
                + "\n\n"
                if heading_path
                else ""
            )

            prefix_tokens = (
                self.token_count(prefix)
                - self.special_tokens
            )

            budget = (
                self.max_tokens
                - self.special_tokens
                - prefix_tokens
                - 4
            )

            if budget <= self.overlap_tokens:
                raise ValueError(
                    "Heading path is too long "
                    "for the token budget"
                )

            windows = self.split_token_windows(
                text=content,
                budget=budget,
                prefix=prefix,
            )

            for passage, start_char, end_char in windows:

                chunk = self.make_chunk(
                    document=document,
                    strategy="hybrid",
                    index=len(chunks),
                    text=prefix + passage,
                    source_text=passage,
                    start_char=start_char,
                    end_char=end_char,
                    section=section,
                )

                chunks.append(chunk)

        return chunks


