
from pathlib import Path
import statistics
import subprocess
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "external" / "vscode-docs"
)

DOCS_DIR = DATASET_ROOT / "docs"

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "evaluation"
    / "vscode_document_inventory.csv"
)


def inspect_document(file_path: Path) -> dict:

    content = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    headings = re.findall(
        r"^#{1,6}\s+(.+)$",
        content,
        flags=re.MULTILINE,
    )

    code_blocks = len(
        re.findall(r"^\s*```", content, re.MULTILINE)
    ) // 2

    return {
        "source": file_path.relative_to(
            DATASET_ROOT
        ).as_posix(),
        "filename": file_path.name,
        "characters": len(content),
        "words": len(content.split()),
        "headings": len(headings),
        "code_blocks": code_blocks,
        "empty": not content.strip(),
    }


def main():

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Documentation directory missing: {DOCS_DIR}"
        )

    files = sorted(DOCS_DIR.rglob("*.md"))

    if not files:
        raise RuntimeError(
            "No Markdown files found"
        )

    print("Inspecting VS Code documentation...")

    rows = [
        inspect_document(path)
        for path in files
    ]

    df = pd.DataFrame(rows)

    commit = subprocess.check_output(
        [
            "git", "-C", str(DATASET_ROOT),
            "rev-parse", "HEAD"
        ],
        text=True,
    ).strip()

    print(f"\nDataset commit: {commit}")
    print(f"Total documents: {len(df)}")
    print(f"Empty documents: {df['empty'].sum()}")
    print(f"Total words: {df['words'].sum():,}")

    print("\nDOCUMENT LENGTH STATISTICS\n")

    print(
        df["words"].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        )
    )

    print("\n10 LONGEST DOCUMENTS\n")

    print(
        df.nlargest(10, "words")[
            ["source", "words", "headings", "code_blocks"]
        ].to_string(index=False)
    )

    print("\nDOCUMENTS BY TOPIC DIRECTORY\n")

    df["topic"] = df["source"].apply(
        lambda path: (
            Path(path).parts[1]
            if len(Path(path).parts) > 2
            else "other"
        )
    )

    print(
        df.groupby("topic")
        .agg(
            documents=("source", "count"),
            total_words=("words", "sum"),
        )
        .sort_values("documents", ascending=False)
        .to_string()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved inventory: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
