"""Hybrid retrieval engine: BM25 + dense search, Reciprocal Rank Fusion, reranking.

The package does not depend on FastAPI. Its only link with the database is the pair of
protocols ``CorpusSource`` / ``ChunkStore`` implemented in ``app.repositories.corpus``.
"""
