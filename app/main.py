"""API FastAPI : POST /ask et GET /health."""

import logging
import time
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.ingest import load_chunks
from app.llm import generate_answer
from app.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = load_chunks(
        config.docs_dir(), 
        config.chunk_size(), 
        config.chunk_overlap()
)
    app.state.retriever = Retriever(chunks, min_score=config.min_score())
    logger.info("Index construit : %d chunks", len(chunks))
    yield


app = FastAPI(title="RAG Assistant", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    source: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    mode: str
    sources: list[Source]
    latency_ms: int


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "chunks": len(request.app.state.retriever.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    start = time.perf_counter()
    hits = request.app.state.retriever.search(body.question, k=body.top_k)
    try:
        answer, mode = generate_answer(body.question, hits)
    except anthropic.APIError as exc:
        logger.error("Erreur du fournisseur LLM : %s", exc)
        raise HTTPException(status_code=502, detail="Le service LLM est indisponible")
    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info("ask mode=%s hits=%d latency_ms=%d", mode, len(hits), latency_ms)
    return AskResponse(
        answer=answer,
        mode=mode,
        sources=[
            Source(source=h.chunk.source, score=round(h.score, 3), snippet=h.chunk.text[:200])
            for h in hits
        ],
        latency_ms=latency_ms,
    )
