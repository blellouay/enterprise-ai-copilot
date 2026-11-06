
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from app.ai.rag_service import RAGService


ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = (
    ROOT / "data" / "evaluation" / "rag_questions.json"
)

OUTPUT_FILE = (
    ROOT / "data" / "evaluation" / "rag_results.csv"
)


def main():

    with open(EVAL_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    if not questions:
        raise RuntimeError("No evaluation questions found")

    print("Loading RAG pipeline...")

    rag = RAGService()

    results = []

    for item in questions:

        print(f"\nEvaluating {item['id']}")
        print(f"Question: {item['question']}")

        # Measure complete RAG response time
        start = perf_counter()

        result = rag.answer_question(
            question=item["question"],
            top_k=3,
        )

        elapsed = perf_counter() - start

        status = result["status"]

        # A verified answer is treated as answerable
        predicted_answerable = (
            status == "evidence_verified"
        )

        expected_answerable = item["answerable"]

        abstention_decision_correct = (
            predicted_answerable == expected_answerable
        )

        cited_sources = [
            citation["source"]
            for citation in result["citations"]
        ]

        expected_source = item["expected_source"]

        source_cited = (
            expected_source in cited_sources
            if expected_source is not None
            else None
        )

        results.append({
            "question_id": item["id"],
            "question": item["question"],
            "expected_answerable": expected_answerable,
            "predicted_answerable": predicted_answerable,
            "status": status,
            "abstention_decision_correct": (
                abstention_decision_correct
            ),
            "expected_source": expected_source,
            "source_cited": source_cited,
            "answer": result["answer"],
            "citations": json.dumps(
                result["citations"],
                ensure_ascii=False,
            ),
            "expected_facts": " | ".join(
                item["expected_facts"]
            ),
            "latency_seconds": round(elapsed, 3),
            "answer_correct": "",
            "citation_support_correct": "",
        })

        print(f"Status: {status}")
        print(f"Latency: {elapsed:.2f} seconds")
        print(f"Answer: {result['answer']}")

    df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nEVALUATION SUMMARY")
    print("-" * 45)

    print(f"Questions: {len(df)}")

    print(
        "Abstention decision accuracy: "
        f"{df['abstention_decision_correct'].mean():.2%}"
    )

    print(
        "Mean response latency: "
        f"{df['latency_seconds'].mean():.2f} seconds"
    )

    print(
        "Median response latency: "
        f"{df['latency_seconds'].median():.2f} seconds"
    )

    print(
        "Maximum response latency: "
        f"{df['latency_seconds'].max():.2f} seconds"
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(
        "Answer correctness and citation support "
        "still require manual review."
    )


if __name__ == "__main__":
    main()
