
from ollama import Client

from app.core.config import settings

from app.schemas.rag import RAGDraft


class LLMService:

    def __init__(self):
        self.client = Client(
            host=settings.ollama_url,
            timeout=120,
        )

        self.model = settings.ollama_model

    def generate_answer(
        self,
        question: str,
        system_prompt: str | None = None,
    ) -> str:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": question,
        })

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": 0,
            },
        )

        return response.message.content or ""
    
    def generate_rag_draft(
        self,
        prompt: str,
        system_prompt: str,
    ) -> RAGDraft:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            format=RAGDraft.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        return RAGDraft.model_validate_json(
            response.message.content or ""
        )

