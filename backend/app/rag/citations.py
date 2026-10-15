"""Citation markers in the model output.

The model cites passages with ``[S1]``, ``[S2]``... Its output is never trusted for
anything else: document names, pages and excerpts shown to the user are built from the
metadata of the chunks that were actually provided, so a page number cannot be invented.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

# "[S1]", "[s1]", "[S1, S3]", "[S1; S2]" - with the whitespace that precedes the marker.
_MARKER = re.compile(r"([ \t]*)\[\s*(S\d+(?:\s*[,;&]\s*S\d+)*)\s*\]", re.IGNORECASE)
_REF = re.compile(r"S\d+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    # Answer whose markers are valid, normalised to "[S1][S2]" and renumbered in order of
    # first appearance.
    text: str
    # Original identifiers of the cited passages; ``refs[i]`` is shown as ``S{i + 1}``.
    refs: list[str]


def parse_citations(answer: str, valid_refs: Collection[str]) -> ParsedAnswer:
    """Keep the markers that reference a provided passage, drop the others.

    The remaining markers are renumbered (the first cited passage becomes ``[S1]``) so
    that the list of sources displayed under the answer has no gaps.
    """
    valid = {ref.upper() for ref in valid_refs}
    order: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kept: list[str] = []
        for ref in _REF.findall(match.group(2)):
            ref = ref.upper()
            if ref not in valid:
                continue
            if ref not in order:
                order.append(ref)
            marker = f"[S{order.index(ref) + 1}]"
            if marker not in kept:
                kept.append(marker)
        if not kept:
            return ""  # invalid marker: removed together with the space before it
        return match.group(1) + "".join(kept)

    text = _MARKER.sub(replace, answer)
    return ParsedAnswer(text=text.strip(), refs=order)


def strip_citations(text: str) -> str:
    """Remove every marker (used for the conversation history sent back to the model,
    where the markers of a previous turn would point to the wrong passages)."""
    return _MARKER.sub("", text).strip()
