
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ai.embedding_service import EmbeddingService
from app.ai.vector_store import VectorStore, COLLECTION_NAME
from app.ai.retrieval_metrics import (
    recall_at_k,
    hit_at_k,
    reciprocal_rank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_FILE = (
    PROJECT_ROOT / "data" / "evaluation"
    / "retrieval_questions.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "evaluation"
    / "retrieval_results.csv"
)


def main():

    # 1. Load evaluation questions
    with open(EVALUATION_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        raise RuntimeError("Evaluation dataset is empty")

    print(f"Loaded {len(questions)} evaluation questions")

    # 2. Load embedding model
    embedding_service = EmbeddingService()

    # 3. Generate query embeddings in a batch
    query_texts = [item["question"] for item in questions]

    query_embeddings = embedding_service.encode_texts(
        query_texts
    )

    # 4. Connect to Qdrant
    vector_store = VectorStore()

    results = []

    try:
        collection_info = vector_store.client.get_collection(
            COLLECTION_NAME
        )

        point_count = collection_info.points_count

        if point_count is None or point_count <= 0:
            raise RuntimeError("Collection has no indexed points")

        # Full collection retrieval for this small dataset
        search_limit = point_count

        for item, query_vector in zip(
            questions, query_embeddings
        ):

            search_results = vector_store.search(
                query_vector=query_vector,
                limit=search_limit,
            )

            retrieved = [
                result.payload["source"]
                for result in search_results
            ]

            relevant = item["relevant_documents"]

            row = {
                "question_id": item["id"],
                "question": item["question"],
                "expected_sources": ", ".join(relevant),
                "top_1": retrieved[0] if retrieved else None,
                "recall@1": recall_at_k(retrieved, relevant, 1),
                "recall@3": recall_at_k(retrieved, relevant, 3),
                "recall@5": recall_at_k(retrieved, relevant, 5),
                "hit@3": hit_at_k(retrieved, relevant, 3),
                "reciprocal_rank": reciprocal_rank(
                    retrieved, relevant
                ),
            }

            results.append(row)

    finally:
        vector_store.close()

    # 5. Convert results to DataFrame
    df = pd.DataFrame(results)

    # 6. Calculate aggregate metrics
    metrics = [
        "recall@1",
        "recall@3",
        "recall@5",
        "hit@3",
        "reciprocal_rank",
    ]

    averages = df[metrics].mean()

    print("\nPER-QUESTION RESULTS\n")

    print(
        df[
            [
                "question_id",
                "expected_sources",
                "top_1",
                "recall@3",
                "reciprocal_rank",
            ]
        ].to_string(index=False)
    )

    print("\nAGGREGATE METRICS\n")

    for metric in metrics:
        label = (
            "MRR"
            if metric == "reciprocal_rank"
            else metric
        )

        print(f"{label}: {averages[metric]:.4f}")

    # 7. Save reproducible results
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
