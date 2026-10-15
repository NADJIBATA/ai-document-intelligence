"""Document Intelligence: what the LLM is asked to extract, and what the API returns.

``InsightsDraft`` is the structured output requested from the model. It is deliberately
forgiving (every field has a default, unknown categories fall back to "other", dates
that are not ISO dates are kept as text only): one malformed value must not discard a
whole extraction. Nothing in it is trusted for *locating* information: pages are
resolved afterwards by finding the quoted text in the indexed passages.
"""

from __future__ import annotations

import datetime
import re
import uuid
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from app.models.enums import JobStatus
from app.schemas.llm_types import Description, OptionalText, ShortText, Text, TextList, choice

DocumentType = Literal[
    "contract",
    "invoice",
    "report",
    "letter",
    "resume",
    "scientific_paper",
    "manual",
    "policy",
    "other",
]
EntityType = Literal["person", "organization", "location", "other"]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")


def _iso_date(value: object) -> object:
    """Keep real ISO dates only ("2025-03-31"); anything else becomes ``None``."""
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str) and _ISO_DATE.match(value.strip()):
        try:
            return datetime.date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _number(value: object) -> object:
    """Accept 1200.5, "1200.50", "1,200.50" or "1 200,50"; anything else -> ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        compact = _WHITESPACE.sub("", value)  # includes non-breaking spaces
        if "," in compact and "." in compact:
            # The last separator is the decimal one.
            if compact.rfind(",") > compact.rfind("."):
                compact = compact.replace(".", "").replace(",", ".")
            else:
                compact = compact.replace(",", "")
        elif "," in compact:
            head, _, tail = compact.rpartition(",")
            compact = f"{head}.{tail}" if len(tail) != 3 else f"{head}{tail}"
        match = _NUMBER.fullmatch(compact)
        return float(match.group()) if match else None
    return None


# ------------------------------------------------------------------ LLM output
class EntityDraft(BaseModel):
    name: ShortText = Field(min_length=1)
    type: Annotated[EntityType, choice(get_args(EntityType))] = "other"


class KeyDateDraft(BaseModel):
    date: Annotated[datetime.date | None, BeforeValidator(_iso_date)] = Field(
        default=None, description="ISO 8601 date (YYYY-MM-DD) when it can be determined."
    )
    text: ShortText = Field(
        min_length=1, description="The date exactly as written in the document."
    )
    description: Description = ""


class AmountDraft(BaseModel):
    value: Annotated[float | None, BeforeValidator(_number)] = None
    currency: OptionalText = Field(default=None, description="ISO 4217 code, e.g. EUR.")
    text: ShortText = Field(
        min_length=1, description="The amount exactly as written in the document."
    )
    description: Description = ""


class InsightsDraft(BaseModel):
    """Information extracted from one part of a document."""

    model_config = ConfigDict(extra="ignore")

    document_type: Annotated[DocumentType, choice(get_args(DocumentType))] = "other"
    title: OptionalText = None
    authors: TextList = Field(default_factory=list)
    language: OptionalText = Field(default=None, description="ISO 639-1 code, e.g. fr or en.")
    summary: Text = ""
    keywords: TextList = Field(default_factory=list)
    entities: list[EntityDraft] = Field(default_factory=list)
    important_dates: list[KeyDateDraft] = Field(default_factory=list)
    monetary_amounts: list[AmountDraft] = Field(default_factory=list)
    key_points: TextList = Field(default_factory=list)

    @field_validator("entities", "important_dates", "monetary_amounts", mode="before")
    @classmethod
    def _drop_malformed_items(cls, value: object) -> object:
        """A null list or an item without its mandatory text must not fail everything."""
        if value is None:
            return []
        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, dict)
                and str(item.get("name") or item.get("text") or "").strip()
            ]
        return value


class SummaryDraft(BaseModel):
    """Consolidation of the partial summaries of a long document."""

    summary: Text = ""
    key_points: TextList = Field(default_factory=list)


# ----------------------------------------------------------------- API output
class EntityRead(BaseModel):
    name: str
    type: EntityType


class KeyDateRead(BaseModel):
    date: datetime.date | None
    text: str
    description: str
    page: int | None = Field(description="Page where the quoted text was found, if any.")
    chunk_id: uuid.UUID | None


class AmountRead(BaseModel):
    value: float | None
    currency: str | None
    text: str
    description: str
    page: int | None
    chunk_id: uuid.UUID | None


class InsightRead(BaseModel):
    document_id: uuid.UUID
    status: JobStatus
    document_type: DocumentType | None
    title: str | None
    language: str | None
    summary: str | None
    authors: list[str]
    keywords: list[str]
    entities: list[EntityRead]
    important_dates: list[KeyDateRead]
    monetary_amounts: list[AmountRead]
    key_points: list[str]
    truncated: bool = Field(
        description="True when the document was too long to be analysed entirely."
    )
    model: str | None
    error_message: str | None
    updated_at: datetime.datetime
