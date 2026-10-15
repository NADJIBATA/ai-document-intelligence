from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class SearchRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        max_length=100,
        description="Restrict the search to these documents (all my documents by default).",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Number of passages (default: RERANK_TOP_K)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "What is the notice period to terminate the contract?"}]
        }
    }


class SearchResultRead(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    section: str | None
    chunk_index: int
    text: str
    score: float = Field(description="Reranker score when available, RRF score otherwise.")
    rerank_score: float | None
    rrf_score: float
    bm25_rank: int | None = Field(description="Rank in the BM25 results (1 = best).")
    dense_rank: int | None = Field(description="Rank in the dense results (1 = best).")


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultRead]
    candidate_count: int = Field(description="Candidates after fusion, before reranking.")
    degraded: list[str] = Field(
        description="Stages skipped because a dependency was unavailable (dense, rerank)."
    )
    timings_ms: dict[str, float]
