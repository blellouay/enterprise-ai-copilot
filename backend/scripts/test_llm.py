
from app.ai.llm_service import LLMService


def main():

    print("Initializing LLM...")

    llm = LLMService()

    system_prompt = """
    You are a technical AI assistant.

    Explain concepts clearly and accurately.
    If you are uncertain, acknowledge it.
    Keep your answers concise.
    """

    questions = [
        "What is Retrieval-Augmented Generation?",
        "What is the difference between a vector database and a relational database?",
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        answer = llm.generate_answer(
            question=question,
            system_prompt=system_prompt,
        )

        print(f"\nANSWER:\n{answer}\n")


if __name__ == "__main__":
    main()
