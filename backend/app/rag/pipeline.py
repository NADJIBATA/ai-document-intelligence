"""Hybrid retrieval pipeline.

    query -> preprocessing
          -> BM25 (top N)  +  dense / Qdrant (top N)        [in parallel]
          -> Reciprocal Rank Fusion (<= max candidates)
          -> hydration from PostgreSQL, filtered on the user (defence in depth)
          -> cross-encoder reranking
          -> top K passages

Every stage receives the ``user_id``: there is no code path searching across users.
If the embedding server, the vector store or the reranker is down, the pipeline
degrades gracefully (lexical-only results, fusion order) and reports it.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import structlog

from app.rag.embeddings import EmbeddingError
from app.rag.fusion import ReciprocalRankFusion
from app.rag.reranker import Reranker, RerankerError
from app.rag.types import (
    ChunkStore,
    LexicalRetriever,
    RetrievalResult,
    RetrievedChunk,
    SearchHit,
    VectorRetriever,
)
from app.rag.vector_store import VectorStoreError

logger = structlog.get_logger(__name__)

_WHITESPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def preprocess_query(query: str, max_chars: int = 2000) -> str:
    """Normalise a user question: control characters, whitespace, maximum length."""
    cleaned = _WHITESPACE.sub(" ", _CONTROL.sub(" ", query)).strip()
    return cleaned[:max_chars].strip()


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    bm25_top_k: int = 30
    dense_top_k: int = 30
    rerank_top_k: int = 5
    max_query_chars: int = 2000
    # Passages scored below this value by the reranker are discarded (0 = keep all).
    min_rerank_score: float = 0.0


class RAGPipeline:
    def __init__(
        self,
        *,
        lexical: LexicalRetriever,
        dense: VectorRetriever,
        fusion: ReciprocalRankFusion,
        reranker: Reranker,
        chunk_store: ChunkStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._lexical = lexical
        self._dense = dense
        self._fusion = fusion
        self._reranker = reranker
        self._chunk_store = chunk_store
        self._config = config or RetrievalConfig()
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval")

    def retrieve(
        self,
        query: str,
        user_id: uuid.UUID,
        *,
        document_ids: Sequence[uuid.UUID] | None = None,
        top_k: int | None = None,
        use_lexical: bool = True,
        use_dense: bool = True,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        """Retrieve the best passages of ``user_id`` for ``query``.

        The ``use_*`` switches exist for evaluation (BM25 only, dense only, hybrid,
        hybrid + reranker); the API always uses the full pipeline.
        """
        config = self._config
        top_k = top_k or config.rerank_top_k
        cleaned = preprocess_query(query, config.max_query_chars)
        result = RetrievalResult(query=cleaned, chunks=[])
        if not cleaned:
            return result

        total_start = time.perf_counter()

        # 1. Lexical and dense retrieval in parallel.
        lexical_future = (
            self._executor.submit(
                self._timed, self._lexical.search, cleaned, user_id, config.bm25_top_k, document_ids
            )
            if use_lexical
            else None
        )
        dense_future = (
            self._executor.submit(
                self._timed, self._dense.search, cleaned, user_id, config.dense_top_k, document_ids
            )
            if use_dense
            else None
        )

        rankings: list[list[SearchHit]] = []
        lexical_hits: list[SearchHit] = []
        dense_hits: list[SearchHit] = []
        if lexical_future is not None:
            lexical_hits, result.timings_ms["bm25"] = lexical_future.result()
            rankings.append(lexical_hits)
        if dense_future is not None:
            try:
                dense_hits, result.timings_ms["dense"] = dense_future.result()
                rankings.append(dense_hits)
            except (EmbeddingError, VectorStoreError) as exc:
                logger.warning("dense_retrieval_unavailable", error=str(exc))
                result.degraded.append("dense")

        # 2. Fusion.
        start = time.perf_counter()
        fused = self._fusion.fuse(rankings)
        result.timings_ms["fusion"] = _elapsed_ms(start)
        result.candidate_count = len(fused)
        if not fused:
            result.timings_ms["total"] = _elapsed_ms(total_start)
            return result

        # 3. Hydration: PostgreSQL is the source of truth and re-applies the user filter.
        start = time.perf_counter()
        loaded = self._chunk_store.get_many(user_id, [hit.chunk_id for hit in fused])
        result.timings_ms["hydration"] = _elapsed_ms(start)

        bm25_ranks = {hit.chunk_id: rank for rank, hit in enumerate(lexical_hits, start=1)}
        dense_ranks = {hit.chunk_id: rank for rank, hit in enumerate(dense_hits, start=1)}
        candidates: list[RetrievedChunk] = []
        dropped = 0
        for hit in fused:
            chunk = loaded.get(hit.chunk_id)
            if chunk is None:
                # Stale vector (deleted document) or an id not owned by the user.
                dropped += 1
                continue
            chunk.rrf_score = hit.score
            chunk.bm25_rank = bm25_ranks.get(hit.chunk_id)
            chunk.dense_rank = dense_ranks.get(hit.chunk_id)
            candidates.append(chunk)
        if dropped:
            logger.warning("retrieval_candidates_dropped", count=dropped)

        # 4. Reranking.
        if use_reranker and candidates:
            start = time.perf_counter()
            try:
                candidates = self._reranker.rerank(cleaned, candidates, top_k)
                if config.min_rerank_score > 0.0:
                    candidates = [
                        chunk
                        for chunk in candidates
                        if chunk.rerank_score is None
                        or chunk.rerank_score >= config.min_rerank_score
                    ]
            except RerankerError as exc:
                logger.warning("reranker_unavailable", error=str(exc))
                result.degraded.append("rerank")
                candidates = candidates[:top_k]
            result.timings_ms["rerank"] = _elapsed_ms(start)
        else:
            candidates = candidates[:top_k]

        result.chunks = candidates
        result.timings_ms["total"] = _elapsed_ms(total_start)
        return result

    @staticmethod
    def _timed(
        search: Callable[..., list[SearchHit]],
        query: str,
        user_id: uuid.UUID,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None,
    ) -> tuple[list[SearchHit], float]:
        start = time.perf_counter()
        hits = search(query, user_id, limit=limit, document_ids=document_ids)
        return hits, _elapsed_ms(start)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
