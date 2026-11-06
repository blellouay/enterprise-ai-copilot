
from pathlib import Path

import pandas as pd

from app.services.document_loader import (
    load_markdown_documents,
)
from app.services.markdown_analyzer import (
    analyze_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main():

    documents = load_markdown_documents(DOCUMENTS_DIR)

    analyzed_documents = [
        analyze_document(document)
        for document in documents
    ]

    summary = []

    for document in analyzed_documents:

        summary.append({
            "filename": document["filename"],
            "title": document["title"],
            "word_count": document["word_count"],
            "character_count": document["character_count"],
            "heading_count": document["heading_count"],
        })

    df = pd.DataFrame(summary)

    print("\nDOCUMENT ANALYSIS\n")
    print(df.to_string(index=False))

    print("\nWORD COUNT STATISTICS\n")
    print(df["word_count"].describe())

    print("\nHEADINGS BY DOCUMENT\n")

    for document in analyzed_documents:

        print(f"\n{document['filename']}")

        for heading in document["headings"]:
            indentation = "  " * (heading["level"] - 1)

            print(
                f"{indentation}- {heading['title']}"
            )


if __name__ == "__main__":
    main()
