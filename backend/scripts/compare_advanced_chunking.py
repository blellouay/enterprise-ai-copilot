
from pathlib import Path

import pandas as pd

from app.services.advanced_markdown_processor import (
    AdvancedMarkdownProcessor,
)
from app.services.advanced_chunker import AdvancedChunker


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data" / "external" / "vscode-docs"
DOCS_DIR = DATASET_ROOT / "docs"

OUTPUT_FILE = (
    ROOT / "data" / "evaluation"
    / "advanced_chunking_statistics.csv"
)


def main():

    processor = AdvancedMarkdownProcessor()

    chunker = AdvancedChunker(
        max_tokens=220,
        overlap_tokens=30,
    )

    files = sorted(DOCS_DIR.rglob("*.md"))

    if not files:
        raise RuntimeError("No VS Code documents found")

    # Sample documents throughout the sorted list
    sample_size = min(30, len(files))

    indices = [
        i * len(files) // sample_size
        for i in range(sample_size)
    ]

    selected_files = [files[i] for i in indices]

    results = []
    examples = {}

    for file_path in selected_files:

        document = processor.process_document(
            file_path,
            DATASET_ROOT,
        )

        _, body, _ = processor.extract_frontmatter(
            file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

        strategies = {
            "fixed": chunker.fixed_chunking(
                document, body
            ),
            "section": chunker.section_chunking(
                document
            ),
            "hybrid": chunker.hybrid_chunking(
                document
            ),
        }

        for name, chunks in strategies.items():

            if name not in examples and chunks:
                examples[name] = chunks[0]

            for chunk in chunks:

                results.append({
                    "source": document["source"],
                    "strategy": name,
                    "chunk_id": chunk["chunk_id"],
                    "tokens": chunk["token_count"],
                    "characters": len(chunk["text"]),
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "exceeds_256": (
                        chunk["token_count"] > 256
                    ),
                })

    df = pd.DataFrame(results)

    print("\nADVANCED CHUNKING COMPARISON")
    print("=" * 55)

    print(f"Documents analyzed: {len(selected_files)}")

    summary = df.groupby("strategy").agg(
        chunks=("tokens", "count"),
        min_tokens=("tokens", "min"),
        median_tokens=("tokens", "median"),
        mean_tokens=("tokens", "mean"),
        max_tokens=("tokens", "max"),
        over_limit=("exceeds_256", "sum"),
    )

    print("\nTOKEN STATISTICS\n")
    print(summary.round(1).to_string())

    print("\nEXAMPLE CHUNKS")

    for strategy, chunk in examples.items():

        print("\n" + "-" * 50)
        print(f"Strategy: {strategy}")
        print(f"Source: {chunk['source']}")
        print(f"Tokens: {chunk['token_count']}")
        print(f"Heading path: {chunk['heading_path']}")
        print(f"Text preview:\n{chunk['text'][:350]}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
