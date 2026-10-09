"""Small text helpers shared by the schema, the dedup index, and the charts."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

import hashlib
import re

# Punctuation the teacher sometimes emits. Saved files stay plain ASCII.
_PUNCTUATION = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
    "\u2026": "...",
}

_WORD = re.compile(r"[a-z0-9]+")


def to_ascii(text: str) -> str:
    """Fold common punctuation to ASCII and drop anything that is still not ASCII."""
    converted = text or ""
    # First map "smart" punctuation to its ASCII look-alike, so meaning is kept (e.g. an en dash becomes "-").
    for source, target in _PUNCTUATION.items():
        converted = converted.replace(source, target)
    # Then drop whatever is left outside ASCII (code points >= 128), e.g. emoji or accented letters.
    return "".join(ch for ch in converted if ord(ch) < 128)


def word_count(text: str) -> int:
    """Whitespace-separated words. Used as the scenario length gate."""
    return len((text or "").split())


def scenario_key(text: str) -> str:
    """Hash of the scenario after case and whitespace are normalized. Exact-dup key."""
    # split() + join collapses any run of spaces, tabs or newlines into one space, so two scenarios
    # that differ only in case or spacing produce the same hash.
    normalized = " ".join((text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def tokens(text: str) -> list:
    """Lowercase word tokens for the keyword chart."""
    return _WORD.findall((text or "").lower())
