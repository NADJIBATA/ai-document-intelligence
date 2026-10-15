"""Lexical retrieval with Okapi BM25.

``BM25Index`` is a small inverted index:

    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

``BM25Retriever`` keeps **one index per user**, built from that user's chunks only
and cached in memory. The cache entry is rebuilt as soon as the corpus signature
changes (a document was indexed or deleted), including from another process.

To move to Elasticsearch / OpenSearch later, implement the ``LexicalRetriever``
protocol with another class: nothing else changes.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.rag.analyzer import TextAnalyzer
from app.rag.types import CorpusEntry, CorpusSource, SearchHit


@dataclass(frozen=True, slots=True)
class _Postings:
    doc_indices: npt.NDArray[np.int32]
    term_frequencies: npt.NDArray[np.float32]


class BM25Index:
    """Immutable BM25 index over a list of chunks."""

    def __init__(
        self,
        entries: Sequence[CorpusEntry],
        analyzer: TextAnalyzer,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._analyzer = analyzer
        self._k1 = k1
        self._b = b
        self._chunk_ids = [entry.chunk_id for entry in entries]
        self._document_ids = np.array([entry.document_id.int for entry in entries], dtype=object)

        lengths = np.zeros(len(entries), dtype=np.float32)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for index, entry in enumerate(entries):
            tokens = analyzer.tokenize(entry.text)
            lengths[index] = len(tokens)
            for term, frequency in Counter(tokens).items():
                docs, freqs = postings.setdefault(term, ([], []))
                docs.append(index)
                freqs.append(frequency)

        self._lengths = lengths
        self._average_length = float(lengths.mean()) if len(entries) else 0.0
        self._postings = {
            term: _Postings(np.asarray(docs, dtype=np.int32), np.asarray(freqs, dtype=np.float32))
            for term, (docs, freqs) in postings.items()
        }

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def idf(self, term: str) -> float:
        postings = self._postings.get(term)
        if postings is None:
            return 0.0
        df = len(postings.doc_indices)
        return math.log(1.0 + (len(self) - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        *,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[SearchHit]:
        if not len(self) or limit <= 0:
            return []
        scores = np.zeros(len(self), dtype=np.float64)
        # A term repeated in the query counts once: BM25 has no query-term saturation here.
        for term in set(self._analyzer.tokenize(query)):
            postings = self._postings.get(term)
            if postings is None:
                continue
            tf = postings.term_frequencies
            norm = (
                1.0
                - self._b
                + self._b * self._lengths[postings.doc_indices] / (self._average_length or 1.0)
            )
            scores[postings.doc_indices] += (
                self.idf(term) * tf * (self._k1 + 1.0) / (tf + self._k1 * norm)
            )

        if document_ids is not None:
            allowed = {document_id.int for document_id in document_ids}
            mask = np.fromiter(
                (doc in allowed for doc in self._document_ids), dtype=bool, count=len(self)
            )
            scores = np.where(mask, scores, 0.0)

        matching = np.flatnonzero(scores > 0.0)
        if matching.size == 0:
            return []
        # Highest score first; ties broken by position for deterministic results.
        order = matching[np.lexsort((matching, -scores[matching]))][:limit]
        return [SearchHit(self._chunk_ids[i], float(scores[i])) for i in order]


class BM25Retriever:
    """Per-user BM25 search with an in-memory, signature-checked LRU cache."""

    def __init__(
        self,
        corpus: CorpusSource,
        analyzer: TextAnalyzer | None = None,
        *,
        max_cached_users: int = 64,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._corpus = corpus
        self._analyzer = analyzer or TextAnalyzer()
        self._max_cached_users = max_cached_users
        self._k1 = k1
        self._b = b
        self._cache: OrderedDict[uuid.UUID, tuple[str, BM25Index]] = OrderedDict()
        self._lock = threading.Lock()

    def search(
        self,
        query: str,
        user_id: uuid.UUID,
        *,
        limit: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[SearchHit]:
        return self._index_for(user_id).search(query, limit=limit, document_ids=document_ids)

    def invalidate(self, user_id: uuid.UUID) -> None:
        with self._lock:
            self._cache.pop(user_id, None)

    def _index_for(self, user_id: uuid.UUID) -> BM25Index:
        signature = self._corpus.signature(user_id)
        with self._lock:
            cached = self._cache.get(user_id)
            if cached is not None and cached[0] == signature:
                self._cache.move_to_end(user_id)
                return cached[1]

        # Built outside the lock: two concurrent builds for the same user are harmless
        # (same result), blocking every other user during a build is not.
        index = BM25Index(self._corpus.load(user_id), self._analyzer, k1=self._k1, b=self._b)
        with self._lock:
            self._cache[user_id] = (signature, index)
            self._cache.move_to_end(user_id)
            while len(self._cache) > self._max_cached_users:
                self._cache.popitem(last=False)
        return index
