from __future__ import annotations

from app.rag.citations import parse_citations, strip_citations

VALID = ["S1", "S2", "S3", "S4", "S5"]


def test_valid_markers_are_kept_and_listed_in_order_of_appearance() -> None:
    parsed = parse_citations("The notice period is thirty days [S1]. Fees apply [S2].", VALID)
    assert parsed.text == "The notice period is thirty days [S1]. Fees apply [S2]."
    assert parsed.refs == ["S1", "S2"]


def test_markers_are_renumbered_without_gaps() -> None:
    parsed = parse_citations("Fees apply [S4]. Notice is thirty days [S2]. Again [S4].", VALID)
    assert parsed.text == "Fees apply [S1]. Notice is thirty days [S2]. Again [S1]."
    assert parsed.refs == ["S4", "S2"]  # S4 is displayed as S1, S2 as S2


def test_grouped_and_lowercase_markers_are_normalised() -> None:
    parsed = parse_citations("Both documents agree [S1, S3]. Also [s2] and [ S3 ; S1 ].", VALID)
    assert parsed.text == "Both documents agree [S1][S2]. Also [S3] and [S2][S1]."
    assert parsed.refs == ["S1", "S3", "S2"]


def test_unknown_markers_are_removed() -> None:
    parsed = parse_citations("Invented fact [S9]. Real fact [S2]. Mixed [S2, S7].", VALID)
    assert parsed.text == "Invented fact. Real fact [S1]. Mixed [S1]."
    assert parsed.refs == ["S2"]


def test_answer_without_valid_marker() -> None:
    parsed = parse_citations("No source at all [S12].", ["S1"])
    assert parsed.text == "No source at all."
    assert parsed.refs == []


def test_french_typography_and_other_brackets_are_preserved() -> None:
    answer = "Quelle surprise ! Le délai est de 30 jours [S1] ; voir l'annexe [A] (page [3])."
    parsed = parse_citations(answer, VALID)
    assert parsed.text == answer
    assert parsed.refs == ["S1"]


def test_strip_citations() -> None:
    assert strip_citations("Thirty days [S1]. Fees apply [S2][S3].") == "Thirty days. Fees apply."
    assert strip_citations("Nothing to strip.") == "Nothing to strip."


def test_markdown_structure_is_preserved() -> None:
    # Lists and tables (rendered by the frontend) keep their line breaks, markers and
    # pipes; markers are renumbered and invalid ones removed inside them as well.
    answer = (
        "Two obligations:\n\n"
        "- Pay within **45 days** [S3]\n"
        "- Report incidents [S9]\n\n"
        "| Aspect | Value |\n"
        "|---|---|\n"
        "| Notice | 90 days [S2] |"
    )
    parsed = parse_citations(answer, VALID)
    assert parsed.text == (
        "Two obligations:\n\n"
        "- Pay within **45 days** [S1]\n"
        "- Report incidents\n\n"
        "| Aspect | Value |\n"
        "|---|---|\n"
        "| Notice | 90 days [S2] |"
    )
    assert parsed.refs == ["S3", "S2"]
