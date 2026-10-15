from __future__ import annotations

import math
import uuid

import pytest

from app.rag.analyzer import TextAnalyzer
from app.rag.bm25 import BM25Index, BM25Retriever
from app.rag.types import CorpusEntry, SearchHit

DOC_A = uuid.UUID(int=1)
DOC_B = uuid.UUID(int=2)


def entry(number: int, text: str, document_id: uuid.UUID = DOC_A) -> CorpusEntry:
    return CorpusEntry(chunk_id=uuid.UUID(int=1000 + number), document_id=document_id, text=text)


def ids(hits: list[SearchHit]) -> list[int]:
    return [hit.chunk_id.int - 1000 for hit in hits]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index(
        [
            entry(1, "The contract may be terminated with a notice period of thirty days."),
            entry(2, "Invoices are payable within thirty days of receipt."),
            entry(3, "Late payment penalties apply to unpaid invoices. Invoices must be dated."),
            entry(4, "The supplier guarantees the confidentiality of the data.", DOC_B),
        ],
        TextAnalyzer(),
    )


def test_ranks_the_most_relevant_chunk_first(index: BM25Index) -> None:
    hits = index.search("contract termination notice", limit=10)
    assert ids(hits)[0] == 1
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_only_matching_chunks_are_returned(index: BM25Index) -> None:
    assert ids(index.search("confidentiality", limit=10)) == [4]
    assert index.search("blockchain", limit=10) == []
    assert index.search("", limit=10) == []
    assert index.search("the of a", limit=10) == []  # stop words only


def test_limit(index: BM25Index) -> None:
    assert len(index.search("invoices thirty days", limit=2)) == 2
    assert index.search("invoices", limit=0) == []


def test_term_frequency_and_plural_matching(index: BM25Index) -> None:
    # "invoice" (singular) matches "Invoices"; chunk 3 contains it twice.
    assert ids(index.search("invoice", limit=10)) == [3, 2]


def test_idf_formula(index: BM25Index) -> None:
    # idf() takes an analysed term. "day" is in 2 of 4 chunks, "supplier" in 1 of 4.
    assert index.idf("day") == pytest.approx(math.log(1 + (4 - 2 + 0.5) / (2 + 0.5)))
    assert index.idf("supplier") > index.idf("day")
    assert index.idf("unknown") == 0.0


def test_score_matches_the_okapi_formula() -> None:
    analyzer = TextAnalyzer()
    index = BM25Index([entry(1, "alpha beta beta"), entry(2, "gamma delta")], analyzer)
    hit = index.search("beta", limit=1)[0]
    k1, b, tf, length, avgdl = 1.5, 0.75, 2.0, 3.0, 2.5
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))
    assert hit.score == pytest.approx(expected, rel=1e-6)


def test_document_filter(index: BM25Index) -> None:
    assert ids(index.search("thirty days", limit=10, document_ids=[DOC_B])) == []
    assert ids(index.search("confidentiality data", limit=10, document_ids=[DOC_B])) == [4]
    assert ids(index.search("confidentiality", limit=10, document_ids=[])) == []


def test_ties_are_broken_by_position() -> None:
    index = BM25Index([entry(2, "same text"), entry(1, "same text")], TextAnalyzer())
    assert ids(index.search("same", limit=10)) == [2, 1]


def test_empty_index() -> None:
    index = BM25Index([], TextAnalyzer())
    assert len(index) == 0
    assert index.search("anything", limit=5) == []


# ---------------------------------------------------------------- BM25Retriever
class FakeCorpus:
    """Corpus keyed by user, counting how many times each user's chunks were loaded."""

    def __init__(self) -> None:
        self.chunks: dict[uuid.UUID, list[CorpusEntry]] = {}
        self.loads: dict[uuid.UUID, int] = {}

    def signature(self, user_id: uuid.UUID) -> str:
        return str(len(self.chunks.get(user_id, [])))

    def load(self, user_id: uuid.UUID) -> list[CorpusEntry]:
        self.loads[user_id] = self.loads.get(user_id, 0) + 1
        return list(self.chunks.get(user_id, []))


ALICE = uuid.UUID(int=11)
BOB = uuid.UUID(int=12)


def test_retriever_never_returns_another_users_chunks() -> None:
    corpus = FakeCorpus()
    corpus.chunks[ALICE] = [entry(1, "alice secret salary report")]
    corpus.chunks[BOB] = [entry(2, "bob secret merger plan")]
    retriever = BM25Retriever(corpus)

    assert ids(retriever.search("secret", ALICE, limit=10)) == [1]
    assert ids(retriever.search("secret", BOB, limit=10)) == [2]
    assert retriever.search("merger", ALICE, limit=10) == []
    assert retriever.search("secret", uuid.UUID(int=99), limit=10) == []


def test_index_is_cached_until_the_corpus_changes() -> None:
    corpus = FakeCorpus()
    corpus.chunks[ALICE] = [entry(1, "first document")]
    retriever = BM25Retriever(corpus)

    retriever.search("document", ALICE, limit=5)
    retriever.search("document", ALICE, limit=5)
    assert corpus.loads[ALICE] == 1

    corpus.chunks[ALICE].append(entry(2, "second document"))  # signature changes
    assert ids(retriever.search("second", ALICE, limit=5)) == [2]
    assert corpus.loads[ALICE] == 2


def test_invalidate_forces_a_rebuild() -> None:
    corpus = FakeCorpus()
    corpus.chunks[ALICE] = [entry(1, "first document")]
    retriever = BM25Retriever(corpus)
    retriever.search("document", ALICE, limit=5)
    retriever.invalidate(ALICE)
    retriever.search("document", ALICE, limit=5)
    assert corpus.loads[ALICE] == 2


def test_cache_is_bounded() -> None:
    corpus = FakeCorpus()
    users = [uuid.UUID(int=100 + i) for i in range(3)]
    for number, user in enumerate(users):
        corpus.chunks[user] = [entry(number, "some text")]
    retriever = BM25Retriever(corpus, max_cached_users=2)
    for user in users:
        retriever.search("text", user, limit=5)
    retriever.search("text", users[0], limit=5)  # evicted (least recently used): rebuilt
    assert corpus.loads[users[0]] == 2
    retriever.search("text", users[2], limit=5)  # still cached
    assert corpus.loads[users[2]] == 1
