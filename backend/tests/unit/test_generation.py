from __future__ import annotations

import uuid

import pytest

from app.llm.base import ChatMessage, LLMError
from app.llm.fake import FakeLLMProvider
from app.rag.generation import AnswerGenerator, make_excerpt
from app.rag.prompts import REFUSAL_MESSAGES
from app.rag.types import RetrievedChunk

QUESTION = "What is the notice period?"


def chunk(number: int, text: str, page: int = 1, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.UUID(int=number),
        document_id=uuid.UUID(int=100 + number),
        filename=f"doc-{number}.pdf",
        text=text,
        page=page,
        section="Termination" if number == 1 else None,
        chunk_index=number,
        rerank_score=score,
    )


CHUNKS = [
    chunk(1, "The notice period is thirty days.", page=14, score=0.93),
    chunk(2, "Invoices are payable within sixty days.", page=3, score=0.41),
    chunk(3, "The supplier keeps the data confidential.", page=7, score=0.12),
]


def test_sources_come_from_the_chunk_metadata() -> None:
    llm = FakeLLMProvider(responses=["The notice period is thirty days [S1]."])
    answer = AnswerGenerator(llm).generate(QUESTION, CHUNKS)

    assert answer.answered is True
    assert answer.text == "The notice period is thirty days [S1]."
    assert len(answer.sources) == 1
    source = answer.sources[0]
    assert source.ref == "S1"
    assert source.chunk_id == CHUNKS[0].chunk_id
    assert source.document_id == CHUNKS[0].document_id
    assert (source.filename, source.page, source.section) == ("doc-1.pdf", 14, "Termination")
    assert source.excerpt == "The notice period is thirty days."
    assert source.score == 0.93
    assert answer.model == "fake-llm"
    assert answer.citation_fallback is False
    assert answer.prompt_tokens is not None


def test_the_model_cannot_invent_a_page_or_a_source() -> None:
    llm = FakeLLMProvider(
        responses=["It is thirty days (contract.pdf, page 99) [S1][S8]. See also [S42]."]
    )
    answer = AnswerGenerator(llm).generate(QUESTION, CHUNKS)
    assert [source.ref for source in answer.sources] == ["S1"]
    assert answer.sources[0].page == 14  # from the index, whatever the model wrote
    assert "[S8]" not in answer.text
    assert "[S42]" not in answer.text


def test_cited_sources_are_renumbered_in_order_of_appearance() -> None:
    llm = FakeLLMProvider(responses=["Data stays confidential [S3]. Notice: thirty days [S1]."])
    answer = AnswerGenerator(llm).generate(QUESTION, CHUNKS)
    assert answer.text == "Data stays confidential [S1]. Notice: thirty days [S2]."
    assert [(s.ref, s.chunk_id) for s in answer.sources] == [
        ("S1", CHUNKS[2].chunk_id),
        ("S2", CHUNKS[0].chunk_id),
    ]


@pytest.mark.parametrize("output", ["[NO_ANSWER]", "NO_ANSWER", " [no answer] ", "[NO_ANSWER]."])
def test_refusal_sentinel_becomes_a_clear_message(output: str) -> None:
    llm = FakeLLMProvider(responses=[output])
    answer = AnswerGenerator(llm).generate(QUESTION, CHUNKS)
    assert answer.answered is False
    assert answer.sources == []
    assert answer.text == REFUSAL_MESSAGES["en"]


def test_refusal_is_localised() -> None:
    llm = FakeLLMProvider(responses=["[NO_ANSWER]"])
    answer = AnswerGenerator(llm).generate("Quelle est la durée du préavis ?", CHUNKS)
    assert answer.text == REFUSAL_MESSAGES["fr"]


def test_no_passage_means_no_model_call() -> None:
    llm = FakeLLMProvider()
    answer = AnswerGenerator(llm).generate(QUESTION, [])
    assert answer.answered is False
    assert answer.text == REFUSAL_MESSAGES["en"]
    assert answer.model is None
    assert llm.calls == []


def test_answer_without_marker_lists_every_passage() -> None:
    llm = FakeLLMProvider(responses=["The notice period is thirty days."])
    answer = AnswerGenerator(llm).generate(QUESTION, CHUNKS)
    assert answer.answered is True
    assert answer.citation_fallback is True
    assert [source.ref for source in answer.sources] == ["S1", "S2", "S3"]


def test_sentinel_mixed_with_a_cited_answer_is_treated_as_an_answer() -> None:
    llm = FakeLLMProvider(responses=["[NO_ANSWER] Actually it is thirty days [S1]."])
    answer = AnswerGenerator(llm).generate(QUESTION, CHUNKS)
    assert answer.answered is True
    assert answer.text == "Actually it is thirty days [S1]."


def test_history_is_limited_to_the_last_turns() -> None:
    history = [
        ChatMessage("user" if i % 2 == 0 else "assistant", f"message {i}") for i in range(10)
    ]
    llm = FakeLLMProvider(responses=["Thirty days [S1]."])
    AnswerGenerator(llm, history_turns=2).generate(QUESTION, CHUNKS, history)
    sent = llm.calls[0]
    assert [m.content for m in sent[1:-1]] == [f"message {i}" for i in range(6, 10)]

    llm = FakeLLMProvider(responses=["Thirty days [S1]."])
    AnswerGenerator(llm, history_turns=0).generate(QUESTION, CHUNKS, history)
    assert len(llm.calls[0]) == 2  # system + question


def test_model_errors_propagate() -> None:
    llm = FakeLLMProvider()
    llm.fail_with = LLMError("llm: HTTP 503", status_code=503)
    with pytest.raises(LLMError):
        AnswerGenerator(llm).generate(QUESTION, CHUNKS)


def test_default_fake_model_gives_a_grounded_answer() -> None:
    answer = AnswerGenerator(FakeLLMProvider()).generate(QUESTION, CHUNKS)
    assert answer.text == "The notice period is thirty days. [S1]"
    assert answer.sources[0].chunk_id == CHUNKS[0].chunk_id


def test_make_excerpt() -> None:
    assert make_excerpt("short   text\nhere") == "short text here"
    long = make_excerpt("word " * 200)
    assert len(long) <= 301
    assert long.endswith("…")
    assert not long[:-1].endswith(" ")
