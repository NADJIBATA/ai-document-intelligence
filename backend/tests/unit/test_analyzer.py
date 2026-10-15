from __future__ import annotations

from app.rag.analyzer import TextAnalyzer, fold, light_stem


def test_fold_lowercases_and_strips_accents() -> None:
    assert fold("Résiliation ANTICIPÉE") == "resiliation anticipee"
    assert fold("ŒUVRE naïve") == "oeuvre naive"


def test_french_sentence() -> None:
    tokens = TextAnalyzer().tokenize("La résiliation du contrat est possible après 30 jours.")
    assert tokens == ["resiliation", "contrat", "possible", "apre", "30", "jour"]


def test_stopwords_depend_on_the_languages() -> None:
    assert TextAnalyzer(languages=["en"]).tokenize("le chat the cat") == ["le", "chat", "cat"]
    assert TextAnalyzer(languages=["fr"]).tokenize("le chat the cat") == ["chat", "the", "cat"]


def test_month_names_are_not_stopwords() -> None:
    assert "may" in TextAnalyzer().tokenize("Payments are due in May")


def test_light_stemmer_conflates_regular_plurals() -> None:
    assert light_stem("contracts") == "contract"
    assert light_stem("clauses") == "clause"
    assert light_stem("taxes") == "tax"
    assert light_stem("party") == light_stem("parties") == light_stem("partie")
    # Words that only look like plurals are left alone.
    for word in ("bus", "class", "analysis", "des"):
        assert light_stem(word) == word


def test_query_and_document_share_the_same_analysis() -> None:
    analyzer = TextAnalyzer()
    assert analyzer.tokenize("PÉNALITÉS de retard") == analyzer.tokenize("pénalité de retards")


def test_stemmer_can_be_disabled() -> None:
    assert TextAnalyzer(stemmer=None).tokenize("Contracts") == ["contracts"]


def test_punctuation_and_numbers() -> None:
    assert TextAnalyzer().tokenize("Article 12.3 : l'indemnité (10 000 EUR)") == [
        "article",
        "12",
        "3",
        "indemnite",
        "10",
        "000",
        "eur",
    ]
