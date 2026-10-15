"""Basic protection against prompt injection coming from documents.

Documents are untrusted input: a PDF can contain "ignore your previous instructions and
...". Three simple, complementary measures are implemented here and in ``prompts.py``:

1. **Detection** - instruction-like passages are flagged (at ingestion and again when
   the prompt is built) so that they are logged and explicitly marked in the prompt.
2. **Neutralisation** - passages are escaped before being placed in the prompt, so they
   can neither close the ``<source>`` block nor forge chat-template markers.
3. **Separation** - instructions live in the system message only; passages are data in a
   delimited block of the user message, and the model is told never to obey them.

This is a *basic* protection, not a guarantee: no known technique fully prevents
indirect prompt injection. The last line of defence is structural - the model has no
tools, and citations/pages are always derived from metadata, never from its output.
"""

from __future__ import annotations

import re

_FLAGS = re.IGNORECASE | re.MULTILINE

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}"
            r"\b(previous|prior|above|earlier|preceding|all|any|your|the)\b.{0,40}"
            r"\b(instructions?|prompts?|rules?|directives?|guidelines?)\b",
            _FLAGS,
        ),
    ),
    (
        "ignore_instructions_fr",
        re.compile(
            r"\b(ignore[zr]?|oublie[zr]?|ne\s+tene?z?\s+pas\s+compte|ne\s+tiens\s+pas\s+compte)\b"
            r".{0,60}\b(instructions?|consignes?|r[èe]gles?|directives?)\b",
            _FLAGS,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\byou\s+are\s+now\b|\bfrom\s+now\s+on\s*,?\s+you\b|\bpretend\s+to\s+be\b"
            r"|\b(tu\s+es|vous\s+[êe]tes)\s+(maintenant|d[ée]sormais)\b",
            _FLAGS,
        ),
    ),
    (
        "system_prompt",
        re.compile(
            r"\b(system\s+prompt|prompt\s+syst[èe]me|developer\s+message|hidden\s+instructions?"
            r"|instructions?\s+cach[ée]es?)\b",
            _FLAGS,
        ),
    ),
    (
        "new_instructions",
        re.compile(r"\b(new|updated|nouvelles?)\s+(instructions?|consignes?)\s*:", _FLAGS),
    ),
    ("fake_role", re.compile(r"^\s*(system|assistant|developer)\s*:", _FLAGS)),
    (
        "chat_markup",
        re.compile(
            r"<\|?\s*(im_start|im_end|system|assistant|endoftext)\s*\|?>"
            r"|\[/?INST\]|<</?SYS>>|</?\s*(source|sources|question)\b",
            _FLAGS,
        ),
    ),
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def detect_injection(text: str) -> list[str]:
    """Names of the injection patterns found in ``text`` (empty list: nothing suspect)."""
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]


def is_suspicious(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in INJECTION_PATTERNS)


def escape_for_prompt(text: str) -> str:
    """Make ``text`` inert inside the prompt.

    Angle brackets are escaped, so a passage cannot close its ``<source>`` block, open a
    fake one or inject chat-template markers; control characters are removed.
    """
    cleaned = _CONTROL_CHARS.sub("", text)
    return cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_attribute(value: str) -> str:
    """Escape a metadata value (file name, section title) used as an XML-like attribute."""
    single_line = " ".join(escape_for_prompt(value).split())
    return single_line.replace('"', "&quot;")
