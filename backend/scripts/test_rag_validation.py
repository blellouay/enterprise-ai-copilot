
from app.ai.rag_service import RAGService


def main():

    rag = RAGService()

    questions = [
        "How long is an account locked after five failed login attempts?",
        "What happens when deployment fails?",
        "What is AcmeOps's annual employee salary budget?",
        "Which employee received the highest bonus in 2025?",
        "How long are audit logs retained?",
    ]

    for question in questions:

        print("\n" + "=" * 65)
        print(f"QUESTION: {question}")
        print("=" * 65)

        result = rag.answer_question(question, top_k=3)

        print(f"\nSTATUS: {result['status']}")
        print(f"\nANSWER:\n{result['answer']}")

        print("\nVALIDATED CITATIONS:")

        for citation in result["citations"]:
            print(
                f"[{citation['source_id']}: "
                f"{citation['source']}]"
            )
            print(f"Evidence: {citation['quote']}")

        if not result["citations"]:
            print("No validated citations")

        print("\nRETRIEVED SOURCES:")

        for source in result["retrieved_sources"]:
            print(
                f"{source['source_id']}: "
                f"{source['source']} "
                f"({source['score']:.4f})"
            )


if __name__ == "__main__":
    main()
