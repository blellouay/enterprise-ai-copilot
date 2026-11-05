
from fastapi import FastAPI

from app.api.routers import health


app = FastAPI(
    title="Enterprise AI Copilot",
    description="Enterprise AI platform powered by RAG and LLMs",
    version="0.1.0"
)

app.include_router(health.router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Welcome to Enterprise AI Copilot"
    }
