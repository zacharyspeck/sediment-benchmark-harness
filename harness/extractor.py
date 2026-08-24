"""Pull the graded answer out of a model response.

THE CONTRACT
------------
Every question instructs both arms to answer in two parts:

    ANSWER: <single value, verdict, or comma-separated list>
    <then normal prose reasoning>

The primary extractor reads that line and nothing else. When the line is
missing or empty, a type-aware fallback tries to recover the answer from the
prose and the item is flagged `extraction_uncertain`.

WHY THE FLAG MATTERS
--------------------
An uncertain extraction means the harness guessed. If the uncertain rate rises
above the configured threshold (default 5%), the honest fix is to strengthen
the format instruction in the prompt -- NOT to make the fallback cleverer. A
smarter fallback silently converts a formatting failure into a correctness
score, which is exactly the kind of measurement error this benchmark exists to
avoid. The report prints the rate per arm per family for that reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["Extraction", "extract_answer", "looks_like_refusal", "split_sentences"]

# `ANSWER:` possibly wrapped in markdown emphasis/headers/bullets/backticks.
_ANSWER_LINE = re.compile(
    r"""^[ \t]*
        (?:[-*+>]\s*)?                 # bullet or blockquote
        (?:\#{1,6}\s*)?                # markdown header
        (?:[*_`]{0,3})\s*              # emphasis / code ticks
        ANSWER
        \s*(?:[*_`]{0,3})\s*
        [:\-=—：]             # : - = em-dash fullwidth-colon
        [ \t]*(?P<val>.*)$
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*$", re.MULTILINE)

# number with optional currency / magnitude / unit suffix
_NUM = re.compile(
    r"""(?P<neg>[-−]\s*)?
        \$?\s*
        (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
        \s*
        (?P<suffix>%|percent|pct|x\b|×|bps|mm|m\b|bn\b|b\b|k\b|
                   million|billion|thousand|years?|yrs?|months?|mos?|days?)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_REFUSAL_CUES = (
    "i cannot", "i can't", "i can not", "i'm unable", "i am unable", "unable to answer",
    "i do not have", "i don't have", "no information", "not able to determine",
    "cannot determine", "can't determine", "insufficient information",
    "not enough information", "i'm sorry", "i am sorry", "as an ai",
    "i don't know", "i do not know", "unclear from the", "no record",
)

_STRIP_EDGES = " \t\r\n.,;:!*_`\"'()[]{}"


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter. Good enough to attach a flag to its context."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[$])|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def looks_like_refusal(text: str) -> bool:
    low = (text or "").casefold()
    return any(cue in low for cue in _REFUSAL_CUES)


@dataclass
class Extraction:
    """What the harness believes the model answered, and how it decided."""

    value: str | None                    # cleaned answer text, None if nothing found
    raw_line: str | None = None          # the literal ANSWER line, if present
    method: str = "none"                 # answer_line | answer_line_next | fallback_* | none
    uncertain: bool = False
    flags: list[str] = field(default_factory=list)
    notes: str = ""
    candidates: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.value is not None and self.value != ""

    def to_trace(self) -> dict:
        return {
            "value": self.value,
            "raw_line": self.raw_line,
            "method": self.method,
            "extraction_uncertain": self.uncertain,
            "flags": list(self.flags),
            "notes": self.notes,
            "candidates": self.candidates[:8],
        }


def _clean(val: str) -> str:
    v = (val or "").strip()
    v = re.sub(r"^\**\s*", "", v)
    v = re.sub(r"\s*\**$", "", v)
    v = v.strip(_STRIP_EDGES)
    v = re.sub(r"\s+", " ", v)
    return v.strip()


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text or "")


def _prose_after_answer(text: str, answer_line: str | None) -> str:
    """Everything except the ANSWER line -- what the fabrication scorer reads."""
    if not text:
        return ""
    if not answer_line:
        return text
    idx = text.find(answer_line)
    if idx < 0:
        return text
    return (text[:idx] + text[idx + len(answer_line):]).strip()


def _fallback_numeric(text: str) -> tuple[str | None, list[str], str]:
    """Recover a number from prose.

    Preference order: a sentence carrying an answer cue, then the first number
    anywhere. Both are guesses, hence the uncertainty flag upstream.
    """
    cues = ("answer is", "the answer", "is approximately", "works out to", "equals",
            "comes to", "average is", "median is", "total is", "count is", "we get",
            "therefore", "so the", "= ")
    sentences = split_sentences(text)
    cands: list[str] = []
    for s in sentences:
        low = s.casefold()
        if any(c in low for c in cues):
            for m in _NUM.finditer(s):
                cands.append(m.group(0).strip())
            if cands:
                return cands[0], cands, "number taken from a sentence containing an answer cue"
    for m in _NUM.finditer(text or ""):
        cands.append(m.group(0).strip())
    if cands:
        return cands[0], cands, "first number found in the response"
    return None, [], "no number found in the response"


def _fallback_vocabulary(text: str, vocabulary: Sequence[str]) -> tuple[str | None, list[str], str]:
    """Recover a categorical answer by looking for one of the allowed labels."""
    if not vocabulary:
        return None, [], "no vocabulary supplied for categorical fallback"
    low = (text or "").casefold()
    hits: list[tuple[int, str]] = []
    for term in vocabulary:
        t = str(term).casefold().strip()
        if not t:
            continue
        pat = r"\b" + re.escape(t) + r"\b" if t.isalnum() or " " in t else re.escape(t)
        for m in re.finditer(pat, low):
            hits.append((m.start(), str(term)))
    if not hits:
        return None, [], "no vocabulary term appeared in the response"
    hits.sort()
    ordered: list[str] = []
    for _, term in hits:
        if term not in ordered:
            ordered.append(term)
    # A first-line verdict is the house convention, so prefer an early hit.
    return ordered[0], ordered, f"vocabulary term found in prose at offset {hits[0][0]}"


def extract_answer(
    text: str,
    answer_type: str = "string",
    vocabulary: Sequence[str] | None = None,
) -> Extraction:
    """Extract the answer. `answer_type` only steers the fallback path."""
    if text is None:
        return Extraction(None, method="none", uncertain=True,
                          flags=["empty_response"], notes="response was None")
    if not str(text).strip():
        return Extraction(None, method="none", uncertain=True,
                          flags=["empty_response"], notes="response was empty")

    body = _strip_fences(str(text))
    flags: list[str] = []

    matches = list(_ANSWER_LINE.finditer(body))
    if matches:
        if len(matches) > 1:
            flags.append("multiple_answer_lines")
        m = matches[0]
        raw_line = m.group(0)
        val = _clean(m.group("val"))
        if val:
            return Extraction(
                value=val,
                raw_line=raw_line,
                method="answer_line",
                uncertain=False,
                flags=flags,
                notes=f"{len(matches)} ANSWER line(s); used the first",
            )
        # `ANSWER:` with the value on the following line
        tail = body[m.end():].lstrip("\n")
        nxt = tail.split("\n", 1)[0] if tail else ""
        nxt_clean = _clean(nxt)
        if nxt_clean:
            flags.append("answer_on_following_line")
            return Extraction(
                value=nxt_clean,
                raw_line=raw_line,
                method="answer_line_next",
                uncertain=False,
                flags=flags,
                notes="ANSWER label was empty; used the line beneath it",
            )
        flags.append("empty_answer_line")

    # ---- fallback ----
    flags.append("missing_answer_line" if not matches else "unusable_answer_line")
    at = (answer_type or "string").lower()
    if at in ("numeric", "number", "float", "int", "percent", "multiple", "money", "count"):
        val, cands, note = _fallback_numeric(body)
        method = "fallback_numeric"
    elif at in ("categorical", "verdict", "boolean", "enum"):
        val, cands, note = _fallback_vocabulary(body, vocabulary or [])
        method = "fallback_vocabulary"
    elif at in ("list", "set"):
        val, cands, note = _fallback_vocabulary(body, vocabulary or [])
        if val is not None and cands:
            val = ", ".join(cands)
            note = "assembled from every vocabulary term present, in order of appearance"
        method = "fallback_list"
    else:
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        val, cands, note = (_clean(first) or None), ([first] if first else []), "first non-empty line"
        method = "fallback_first_line"

    if looks_like_refusal(body):
        flags.append("looks_like_refusal")

    return Extraction(
        value=val,
        raw_line=matches[0].group(0) if matches else None,
        method=method if val is not None else "none",
        uncertain=True,
        flags=flags,
        notes=note,
        candidates=cands,
    )


def prose_only(text: str, extraction: Extraction) -> str:
    """The response minus its ANSWER line -- input to the fabrication scorer."""
    return _prose_after_answer(str(text or ""), extraction.raw_line)
