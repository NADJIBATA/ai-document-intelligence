"""Text analysis for lexical search: tokenisation, accent folding, stop words and a
light plural stemmer for French and English.

The same analyzer is applied to documents and to queries. It is deliberately small
and dependency-free; a heavier stemmer (Snowball, lemmatiser) can be plugged in by
passing another callable to :class:`TextAnalyzer`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

_TOKEN = re.compile(r"[a-z0-9]+")
_VOWELS = frozenset("aeiou")

STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        """a an and are as at be been but by can could did do does for from had has have he her
        his how i if in into is it its me my no not of on or our she should so than that the
        their them then there these they this to was we were what when where which who why will
        with would you your""".split()  # noqa: SIM905
    ),
    "fr": frozenset(
        """a au aux avec ce ces cet cette ceux chez comme dans de des du elle elles en est et
        etait ete etre eu il ils je la le les leur leurs lui ma mais me mes moi mon ne nos notre
        nous on ont ou par pas pour qu que quel quelle quelles quels qui sa sans se ses soit son
        sont sur ta te tes toi ton tu un une vos votre vous y c d j l m n s t""".split()  # noqa: SIM905
    ),
}


# Ligatures have no Unicode decomposition: without this, "œuvre" would lose its first letter.
_LIGATURES = str.maketrans({"\u0153": "oe", "\u00e6": "ae", "\u00df": "ss"})


def fold(text: str) -> str:
    """Lower-case and strip accents: "Résiliation" -> "resiliation"."""
    decomposed = unicodedata.normalize("NFKD", text.lower().translate(_LIGATURES))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def light_stem(token: str) -> str:
    """Conservative singular/plural conflation shared by French and English.

    Only regular, unambiguous endings are handled:
    "contracts" -> "contract", "clauses" -> "clause", "taxes" -> "tax",
    "party" / "parties" / "partie" -> "partie".
    """
    if len(token) <= 3 or token[-1].isdigit():
        return token
    if token.endswith("y") and token[-2] not in _VOWELS:
        return token[:-1] + "ie"
    if token.endswith(("sses", "shes", "ches", "xes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


class TextAnalyzer:
    def __init__(
        self,
        languages: Iterable[str] = ("fr", "en"),
        stemmer: Callable[[str], str] | None = light_stem,
    ) -> None:
        self._stopwords: frozenset[str] = frozenset().union(
            *(STOPWORDS.get(language, frozenset()) for language in languages)
        )
        self._stemmer = stemmer

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in _TOKEN.findall(fold(text)):
            if token in self._stopwords:
                continue
            tokens.append(self._stemmer(token) if self._stemmer else token)
        return tokens
