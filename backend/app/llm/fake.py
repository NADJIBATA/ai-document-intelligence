"""Deterministic provider (``LLM_PROVIDER=fake``): tests, CI and demos without any model.

By default it behaves like a very literal RAG model: it reads the ``<sources>`` and
``<question>`` blocks of the prompt, picks the passage sharing the most words with the
question and quotes its most relevant sentence followed by the citation marker - or
returns the "no answer" sentinel when no passage shares a word with the question.

For structured outputs it returns a small, clearly artificial document built from the
prompt (first sentences as summary, most frequent words as keywords, first passages of
each document for a comparison), so that every feature can be demonstrated offline.

Tests can script exact outputs with ``responses`` or plug a ``handler``.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from app.llm.base import ChatMessage, JsonSchema, LLMError, LLMResponse
from app.rag.analyzer import TextAnalyzer

Handler = Callable[[Sequence[ChatMessage], JsonSchema | None], str]

_SOURCE = re.compile(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_QUESTION = re.compile(r"<question>\n(.*?)\n</question>", re.DOTALL)
_DOCUMENT = re.compile(r"<document [^>]*>\n(.*?)\n</document>", re.DOTALL)
_COMPARED = re.compile(r'<source id="([AB]1)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
NO_ANSWER = "[NO_ANSWER]"


class FakeLLMProvider:
    def __init__(
        self,
        *,
        model: str = "fake-llm",
        responses: Sequence[str] | None = None,
        handler: Handler | None = None,
    ) -> None:
        self._model = model
        self._responses = list(responses or [])
        self._handler = handler
        self._analyzer = TextAnalyzer()
        self.calls: list[list[ChatMessage]] = []
        self.fail_with: LLMError | None = None

    def enqueue(self, *responses: str) -> None:
        """Script the next outputs (used before the default behaviour)."""
        self._responses.extend(responses)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: JsonSchema | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self.fail_with is not None:
            raise self.fail_with
        if self._responses:
            content = self._responses.pop(0)
        elif self._handler is not None:
            content = self._handler(messages, json_schema)
        elif json_schema is not None:
            content = self._structured_answer(messages[-1].content if messages else "", json_schema)
        else:
            content = self._grounded_answer(messages[-1].content if messages else "")
        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=sum(len(message.content) for message in messages) // 4,
            completion_tokens=len(content) // 4,
            finish_reason="stop",
        )

    def _grounded_answer(self, prompt: str) -> str:
        question_match = _QUESTION.search(prompt)
        question_terms = set(
            self._analyzer.tokenize(html.unescape(question_match.group(1)))
            if question_match
            else []
        )
        best_ref, best_text, best_overlap = "", "", 0
        for ref, text in _SOURCE.findall(prompt):
            passage = html.unescape(text)
            overlap = len(question_terms & set(self._analyzer.tokenize(passage)))
            if overlap > best_overlap:
                best_ref, best_text, best_overlap = ref, passage, overlap
        if not best_ref:
            return NO_ANSWER
        # Quote the sentence of that passage that is the closest to the question.
        sentences = [part for part in _SENTENCE_END.split(best_text.strip()) if part.strip()]
        best_sentence = max(
            sentences,
            key=lambda sentence: len(question_terms & set(self._analyzer.tokenize(sentence))),
        )
        return f"{best_sentence.strip()} [{best_ref}]"

    def _structured_answer(self, prompt: str, json_schema: JsonSchema) -> str:
        properties = json_schema.schema.get("properties", {})
        payload: dict[str, Any] = {}
        compared = {ref: html.unescape(text) for ref, text in _COMPARED.findall(prompt)}
        if "differences" in properties and {"A1", "B1"} <= set(compared):
            payload["summary"] = "Fake model: the first passage of each document is quoted."
            payload["differences"] = [
                {
                    "aspect": "other",
                    "title": "First passage of each document",
                    "significance": "low",
                    "document_a": {
                        "value": self._first_sentences(compared["A1"], 1),
                        "refs": ["A1"],
                    },
                    "document_b": {
                        "value": self._first_sentences(compared["B1"], 1),
                        "refs": ["B1"],
                    },
                }
            ]
            return json.dumps(payload)

        document = _DOCUMENT.search(prompt)
        text = html.unescape(document.group(1)) if document else ""
        if "summary" in properties:
            payload["summary"] = self._first_sentences(text, 2)
        if "keywords" in properties:
            counts = Counter(token for token in self._analyzer.tokenize(text) if len(token) > 3)
            payload["keywords"] = [word for word, _ in counts.most_common(5)]
        return json.dumps(payload)

    @staticmethod
    def _first_sentences(text: str, count: int) -> str:
        sentences = [part.strip() for part in _SENTENCE_END.split(text.strip()) if part.strip()]
        return " ".join(sentences[:count])
