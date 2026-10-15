"""Evaluation dataset: one JSON object per line.

    {"question": "What is the notice period?", "relevant_chunk_ids": ["..."]}

Optional field: ``"id"`` (defaults to ``q1``, ``q2``...). A relevant chunk is given
either by its id (UUID, as returned by the API) or, for convenience, by a reference
``"<file name>#<chunk index>"`` resolved against the corpus being evaluated. Empty lines
and lines starting with ``#`` are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class DatasetError(ValueError):
    """The dataset file is malformed or references an unknown chunk."""


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    question: str
    # References as written in the dataset (UUIDs or "file#index").
    relevant: tuple[str, ...]


def parse_line(line: str, number: int) -> EvalQuestion | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        item = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"line {number}: invalid JSON ({exc.msg})") from exc
    if not isinstance(item, dict):
        raise DatasetError(f"line {number}: expected a JSON object")
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"line {number}: 'question' must be a non-empty string")
    relevant = item.get("relevant_chunk_ids")
    if (
        not isinstance(relevant, list)
        or not relevant
        or not all(isinstance(ref, str) and ref.strip() for ref in relevant)
    ):
        raise DatasetError(
            f"line {number}: 'relevant_chunk_ids' must be a non-empty list of strings"
        )
    identifier = item.get("id", f"q{number}")
    return EvalQuestion(
        id=str(identifier),
        question=question.strip(),
        relevant=tuple(dict.fromkeys(ref.strip() for ref in relevant)),
    )


def load_dataset(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            parsed = parse_line(line, number)
            if parsed is None:
                continue
            if parsed.id in seen:
                raise DatasetError(f"line {number}: duplicate id '{parsed.id}'")
            seen.add(parsed.id)
            questions.append(parsed)
    if not questions:
        raise DatasetError(f"{path}: the dataset contains no question")
    return questions
