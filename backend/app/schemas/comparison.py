"""Document comparison: request, structured LLM output, and API response."""

from __future__ import annotations

import datetime
import uuid
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import JobStatus
from app.schemas.llm_types import OptionalText, ShortText, Text, TextList, choice

Aspect = Literal["pricing", "dates", "conditions", "clauses", "deadlines", "obligations"]
DEFAULT_ASPECTS: tuple[str, ...] = get_args(Aspect)
Significance = Literal["high", "medium", "low"]


# -------------------------------------------------------------------- request
class CompareRequest(BaseModel):
    document_a_id: uuid.UUID
    document_b_id: uuid.UUID
    aspects: list[Aspect] | None = Field(
        default=None,
        min_length=1,
        description="Aspects to compare (all of them by default).",
    )

    @field_validator("aspects")
    @classmethod
    def _unique_aspects(cls, value: list[Aspect] | None) -> list[Aspect] | None:
        return list(dict.fromkeys(value)) if value else value

    @model_validator(mode="after")
    def _different_documents(self) -> CompareRequest:
        if self.document_a_id == self.document_b_id:
            raise ValueError("document_a_id and document_b_id must be different documents")
        return self


# ------------------------------------------------------------------ LLM output
class SideDraft(BaseModel):
    value: OptionalText = Field(
        default=None, description="What this document says; null if it does not address it."
    )
    refs: TextList = Field(
        default_factory=list, description="Identifiers of the supporting passages, e.g. A2."
    )


class DifferenceDraft(BaseModel):
    aspect: Annotated[str, choice((*DEFAULT_ASPECTS, "other"))] = "other"
    title: ShortText = Field(min_length=1)
    document_a: SideDraft = Field(default_factory=SideDraft)
    document_b: SideDraft = Field(default_factory=SideDraft)
    significance: Annotated[Significance, choice(get_args(Significance), "medium")] = "medium"

    @field_validator("document_a", "document_b", mode="before")
    @classmethod
    def _null_side(cls, value: object) -> object:
        return {} if value is None else value


class ComparisonDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: Text = ""
    differences: list[DifferenceDraft] = Field(default_factory=list)
    similarities: TextList = Field(default_factory=list)

    @field_validator("differences", mode="before")
    @classmethod
    def _drop_malformed_items(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, dict) and str(item.get("title") or "").strip()
            ]
        return value


# ----------------------------------------------------------------- API output
class ComparisonSourceRead(BaseModel):
    """A passage supporting one side of a difference (built from the index)."""

    ref: str = Field(description="`A1`, `A2`... for document A, `B1`... for document B.")
    document_id: uuid.UUID
    document: str
    page: int | None
    section: str | None
    chunk_id: uuid.UUID
    excerpt: str


class ComparisonSideRead(BaseModel):
    value: str | None
    sources: list[ComparisonSourceRead]


class DifferenceRead(BaseModel):
    aspect: str
    title: str
    significance: Significance
    document_a: ComparisonSideRead
    document_b: ComparisonSideRead


class ComparisonResultRead(BaseModel):
    summary: str
    differences: list[DifferenceRead]
    similarities: list[str]
    model: str | None = None


class ComparedDocumentRead(BaseModel):
    id: uuid.UUID | None = Field(description="None if the document was deleted since.")
    name: str


class ComparisonSummaryRead(BaseModel):
    """History entry (without the result)."""

    id: uuid.UUID
    status: JobStatus
    document_a: ComparedDocumentRead
    document_b: ComparedDocumentRead
    aspects: list[str]
    error_message: str | None
    created_at: datetime.datetime
    completed_at: datetime.datetime | None


class ComparisonRead(ComparisonSummaryRead):
    result: ComparisonResultRead | None
