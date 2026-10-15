from __future__ import annotations

import datetime

from app.schemas.comparison import ComparisonDraft
from app.schemas.insight import InsightsDraft, SummaryDraft


def test_empty_object_is_a_valid_extraction() -> None:
    draft = InsightsDraft.model_validate({})
    assert draft.document_type == "other"
    assert draft.title is None
    assert draft.summary == ""
    assert draft.entities == []
    assert SummaryDraft.model_validate({}).summary == ""


def test_categories_are_normalised() -> None:
    draft = InsightsDraft.model_validate(
        {
            "document_type": "Scientific Paper",
            "entities": [
                {"name": "ACME Corp", "type": "Organization"},
                {"name": "Jane Doe", "type": "PERSON"},
                {"name": "Zorg", "type": "alien"},
            ],
        }
    )
    assert draft.document_type == "scientific_paper"
    assert [entity.type for entity in draft.entities] == ["organization", "person", "other"]
    assert InsightsDraft.model_validate({"document_type": "memo"}).document_type == "other"
    assert InsightsDraft.model_validate({"document_type": None}).document_type == "other"


def test_text_fields_are_cleaned() -> None:
    draft = InsightsDraft.model_validate(
        {
            "title": "   ",
            "summary": "  A   summary\nover two lines. ",
            "authors": "Jane Doe",
            "keywords": ["contract", "", "  notice  period ", None],
            "key_points": None,
            "unexpected_field": 1,
        }
    )
    assert draft.title is None
    assert draft.summary == "A summary over two lines."
    assert draft.authors == ["Jane Doe"]
    assert draft.keywords == ["contract", "notice period"]
    assert draft.key_points == []


def test_only_real_iso_dates_are_kept_as_dates() -> None:
    draft = InsightsDraft.model_validate(
        {
            "important_dates": [
                {"date": "2025-03-31", "text": "31 March 2025", "description": "End of term"},
                {"date": "March 2025", "text": "March 2025"},
                {"date": "2025-02-30", "text": "30/02/2025"},
                {"date": None, "text": "next spring"},
                {"date": "2025-01-01"},  # no quote: dropped
                "garbage",
            ]
        }
    )
    assert [(item.date, item.text) for item in draft.important_dates] == [
        (datetime.date(2025, 3, 31), "31 March 2025"),
        (None, "March 2025"),
        (None, "30/02/2025"),
        (None, "next spring"),
    ]


def test_amount_values_accept_common_number_formats() -> None:
    draft = InsightsDraft.model_validate(
        {
            "monetary_amounts": [
                {"value": 1200.5, "currency": "EUR", "text": "1 200,50 EUR"},
                {"value": "1 200,50", "text": "1 200,50"},
                {"value": "1,200.50", "text": "$1,200.50"},
                {"value": "1.200,50", "text": "1.200,50"},
                {"value": "1,200", "text": "1,200"},
                {"value": "12,5", "text": "12,5 %"},
                {"value": "about a thousand", "text": "about a thousand"},
                {"value": True, "currency": "", "text": "true"},
            ]
        }
    )
    assert [item.value for item in draft.monetary_amounts] == [
        1200.5,
        1200.5,
        1200.5,
        1200.5,
        1200.0,
        12.5,
        None,
        None,
    ]
    assert draft.monetary_amounts[0].currency == "EUR"
    assert draft.monetary_amounts[-1].currency is None


def test_over_long_values_are_truncated_not_rejected() -> None:
    draft = InsightsDraft.model_validate(
        {"entities": [{"name": "x" * 500}], "important_dates": [{"text": "d" * 900}]}
    )
    assert len(draft.entities[0].name) == 200
    assert len(draft.important_dates[0].text) == 200


def test_comparison_draft_is_forgiving() -> None:
    draft = ComparisonDraft.model_validate(
        {
            "summary": " Prices differ. ",
            "similarities": "Same governing law",
            "differences": [
                {
                    "aspect": "Pricing",
                    "title": "Monthly fee",
                    "significance": "CRITICAL",
                    "document_a": {"value": "100 EUR", "refs": "A1"},
                    "document_b": None,
                },
                {"aspect": "astrology", "title": "Other", "document_a": {"value": ""}},
                {"aspect": "dates"},  # no title: dropped
            ],
        }
    )
    assert draft.summary == "Prices differ."
    assert draft.similarities == ["Same governing law"]
    first, second = draft.differences
    assert (first.aspect, first.significance) == ("pricing", "medium")
    assert first.document_a.refs == ["A1"]
    assert first.document_b.value is None
    assert first.document_b.refs == []
    assert second.aspect == "other"
    assert second.document_a.value is None
    assert ComparisonDraft.model_validate({}).differences == []
