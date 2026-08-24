"""Numeric and categorical comparison primitives shared by the graders.

Everything here is deterministic string/number handling. The hard part is not
the arithmetic, it is deciding what a model *meant* by "$28.4M" vs "28.4" vs
"28,400,000" without quietly turning a wrong answer into a right one. The rules
below are conservative and every normalization step is recorded in the trace.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

__all__ = [
    "Quantity", "parse_quantity", "compare_numeric", "NumericComparison",
    "normalize_categorical", "compare_categorical", "parse_list", "compare_list",
]

# Suffix alternatives are ordered longest-first: an alternation is first-match,
# so a bare `m` listed before `million` would match the "m" of "million" and
# leave "illion" behind. The trailing lookahead keeps `m` from matching inside
# "margin", and sits INSIDE the optional group so it only applies when a suffix
# was actually present.
_SUFFIX = (
    r"(?:%|percentage\s+points?|percent|pct|bps"
    r"|million|billion|thousand"
    r"|years?|yrs?|months?|mos?|days?"
    r"|mm|bn|[xX×]|[mMbBkK])(?![A-Za-z])"
)

_NUM_RE = re.compile(
    r"""(?P<neg>[-−–]\s*)?
        (?P<cur>[$€£])?\s*
        (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
        \s*
        (?P<suffix>""" + _SUFFIX + r""")?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# unit families the harness understands
UNITS = ("multiple", "percent", "money_m", "count", "years", "months", "ratio", "none")

# Guard against binary floating-point noise at a tolerance boundary. See
# compare_numeric.
_TOL_EPS = 1e-9

_MAG = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
}


@dataclass
class Quantity:
    value: float | None
    unit: str = "none"           # unit as detected/assumed
    raw: str = ""
    detected_unit: str | None = None
    flags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_trace(self) -> dict:
        return {
            "value": self.value, "unit": self.unit, "raw": self.raw,
            "detected_unit": self.detected_unit, "flags": list(self.flags),
            "notes": self.notes,
        }


def _clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.replace("−", "-").replace("–", "-")
    return s.strip()


def parse_quantity(text: str, expected_unit: str = "none") -> Quantity:
    """Parse the first quantity out of `text`, interpreted for `expected_unit`.

    A bare number is assumed to carry the expected unit -- that is the whole
    point of asking for a single value on the ANSWER line. An explicit unit
    that contradicts the expectation is converted where the conversion is
    unambiguous (months->years, dollars->$M) and flagged where it is not.
    """
    raw = _clean_text(text)
    if not raw:
        return Quantity(None, expected_unit, raw, notes="empty answer")

    m = _NUM_RE.search(raw)
    if not m:
        return Quantity(None, expected_unit, raw, notes="no number present in the answer")

    numtxt = m.group("num").replace(",", "")
    try:
        val = float(numtxt)
    except ValueError:
        return Quantity(None, expected_unit, raw, notes=f"unparseable number {numtxt!r}")
    if m.group("neg"):
        val = -val

    suf = (m.group("suffix") or "").lower().strip()
    cur = m.group("cur")
    flags: list[str] = []
    notes = ""
    detected: str | None = None

    # ---- what did the model actually write? ----
    if suf in ("%", "percent", "pct", "percentage point", "percentage points"):
        detected = "percent"
    elif suf == "bps":
        detected = "percent"
        val = val / 100.0
        notes = "converted basis points to percent"
    elif suf in ("x", "×"):
        detected = "multiple"
    elif suf in ("year", "years", "yr", "yrs"):
        detected = "years"
    elif suf in ("month", "months", "mo", "mos"):
        detected = "months"
    elif suf in ("day", "days", "d"):
        detected = "days"
    elif suf in _MAG:
        detected = "money_m" if cur or expected_unit == "money_m" else "scaled"
    elif cur:
        detected = "money_m"

    # ---- convert into the expected unit ----
    if expected_unit == "money_m":
        if suf in _MAG:
            dollars = val * _MAG[suf]
            val = dollars / 1e6
            notes = f"read '{m.group(0).strip()}' as ${val:.4g}M"
        elif cur and val >= 1e5:
            val = val / 1e6
            flags.append("scale_normalized")
            notes = "answer looked like whole dollars; divided by 1e6 to reach $M"
        elif not cur and val >= 1e5:
            val = val / 1e6
            flags.append("scale_normalized")
            notes = "bare number too large for $M; divided by 1e6"
    elif expected_unit == "years":
        if detected == "months":
            val = val / 12.0
            notes = "converted months to years"
        elif detected == "days":
            val = val / 365.25
            notes = "converted days to years"
    elif expected_unit == "months":
        if detected == "years":
            val = val * 12.0
            notes = "converted years to months"
        elif detected == "days":
            val = val / 30.4375
            notes = "converted days to months"
    elif expected_unit == "percent":
        # 0.212 for a gold of 21.2 is a units slip, not a different answer.
        # Rescale, but flag it loudly so the trace shows what happened.
        if detected != "percent" and 0 < abs(val) <= 1.0:
            val = val * 100.0
            flags.append("scale_normalized")
            notes = "answer given as a fraction; multiplied by 100 to reach percent"
    elif expected_unit == "multiple":
        if detected == "percent":
            flags.append("unit_mismatch")
            notes = "expected a multiple, answer carried a percent sign"

    if detected and expected_unit not in ("none", detected) and "unit_mismatch" not in flags:
        if not (
            (expected_unit == "money_m" and detected in ("money_m", "scaled"))
            or (expected_unit in ("years", "months") and detected in ("years", "months", "days"))
            or (expected_unit == "percent" and detected == "percent")
            or (expected_unit == "multiple" and detected == "multiple")
            or (expected_unit == "count" and detected in ("scaled",))
        ):
            flags.append("unit_unexpected")

    trailing = raw[m.end():].strip(" .,;:")
    if trailing and len(trailing.split()) > 6:
        flags.append("answer_line_verbose")

    return Quantity(val, expected_unit, raw, detected, flags, notes)


@dataclass
class NumericComparison:
    passed: bool
    answer: float | None
    gold: float | None
    abs_diff: float | None
    rel_diff: float | None
    tol_abs: float
    tol_rel: float
    statement: str
    flags: list[str] = field(default_factory=list)

    def to_trace(self) -> dict:
        return {
            "passed": self.passed, "answer": self.answer, "gold": self.gold,
            "abs_diff": self.abs_diff, "rel_diff": self.rel_diff,
            "tol_abs": self.tol_abs, "tol_rel": self.tol_rel,
            "statement": self.statement, "flags": list(self.flags),
        }


def compare_numeric(answer: Quantity, gold_value, tol: dict) -> NumericComparison:
    """Pass when |a-g| <= abs tolerance OR |a-g| <= rel * |g|."""
    ta = float(tol.get("abs", 0.0) or 0.0)
    tr = float(tol.get("rel", 0.0) or 0.0)
    try:
        g = float(gold_value)
    except (TypeError, ValueError):
        return NumericComparison(
            False, answer.value, None, None, None, ta, tr,
            f"gold value {gold_value!r} is not numeric; cannot compare",
            ["gold_not_numeric"],
        )
    if answer.value is None:
        return NumericComparison(
            False, None, g, None, None, ta, tr,
            f"no number could be read from the answer; gold was {g:g}",
            ["no_numeric_answer"],
        )
    a = float(answer.value)
    ad = abs(a - g)
    rd = ad / abs(g) if g != 0 else (0.0 if ad == 0 else float("inf"))
    # Compare with a tiny epsilon. A stated tolerance of 0.05 must include a
    # difference a human would call 0.05: in binary floating point 7.15 - 7.1 is
    # 0.05000000000000071, which would otherwise fail an item for a rounding
    # artifact sixteen decimal places down. The epsilon is far below any
    # tolerance this benchmark uses, so it cannot widen a real one.
    ok = (ad <= ta + _TOL_EPS) or (tr > 0 and rd <= tr + _TOL_EPS)
    how = []
    if ta:
        how.append(f"|{a:g} - {g:g}| = {ad:g} vs abs tol {ta:g}")
    if tr:
        how.append(f"rel diff {rd:.4%} vs rel tol {tr:.4%}")
    if not how:
        how.append(f"exact match required: {a:g} vs {g:g}")
    return NumericComparison(
        ok, a, g, ad, rd, ta, tr,
        ("PASS " if ok else "FAIL ") + "; ".join(how),
        list(answer.flags),
    )


# --------------------------------------------------------------------------
# categorical / list
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s%$.-]+")
_WS = re.compile(r"\s+")


def normalize_categorical(s) -> str:
    t = unicodedata.normalize("NFKC", str(s if s is not None else "")).casefold()
    t = t.replace("_", " ").replace("-", " ")
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    # trim polite padding that does not change the verdict
    for lead in ("we would ", "we ", "the answer is ", "answer is ", "verdict is ", "it is "):
        if t.startswith(lead):
            t = t[len(lead):]
    return t.strip()


# Tokens that negate a label appearing after them. Written in the form they
# survive normalization in: normalize_categorical strips apostrophes, so
# "don't" arrives as "don t" and the stem is what we can test.
_NEGATORS = {
    "not", "no", "never", "cannot", "without", "neither", "nor",
    "dont", "don", "doesnt", "doesn", "didnt", "didn", "wont", "won",
    "wouldnt", "wouldn", "couldnt", "couldn", "shouldnt", "shouldn",
    "isnt", "isn", "arent", "aren", "wasnt", "wasn", "cant",
}
_NEG_WINDOW = 3   # tokens of context inspected before a match


def _label_forms(label: str, accept: dict) -> set[str]:
    forms = {normalize_categorical(label)}
    forms |= {normalize_categorical(x) for x in (accept.get(label) or [])}
    return {f for f in forms if f}


def _is_negated(text: str, start: int, form: str) -> bool:
    """Is this match preceded by a negation that flips its meaning?

    A form that already contains a negator ("do not advance") is not negated by
    it -- that phrasing IS the label. Only a bare form under a negator counts,
    which is what makes "not a pass" different from "pass".
    """
    if any(tok in _NEGATORS for tok in form.split()):
        return False
    prefix = text[:start].split()[-_NEG_WINDOW:]
    if any(tok in _NEGATORS for tok in prefix):
        return True
    # "X rather than Y" asserts X and rejects Y
    return len(prefix) >= 2 and prefix[-2:] == ["rather", "than"]


def compare_categorical(answer: str, gold: str, accept: dict | None = None) -> tuple[bool, str, list[str]]:
    """Compare a categorical answer against gold, honouring a synonym map.

    `accept` maps a canonical label -> list of accepted surface forms.

    TWO RULES, BOTH LEARNED THE HARD WAY
    ------------------------------------
    1. **Longest match wins, across every label.** Naive containment scored
       "we would not advance" as `advance`, because "advance" is a substring of
       it. Collecting matches from ALL labels and letting the most specific one
       decide makes "would not advance" (a `pass` form) beat "advance".
    2. **A negated bare form does not match.** "not a pass" contains "pass" and
       no accept list will ever enumerate every negation. A form preceded by a
       negator is dropped unless the form itself contains one.

    Without these, an arm that answers the exact opposite of the gold scores as
    correct -- which would quietly invalidate every categorical family.
    """
    flags: list[str] = []
    a = normalize_categorical(answer)
    g = normalize_categorical(gold)
    if not a:
        return False, "answer was empty", ["no_categorical_answer"]

    accept = dict(accept or {})
    accept.setdefault(gold, [])

    # collect every whole-word match of every form of every label
    cands: list[dict] = []
    for label in accept:
        for form in _label_forms(label, accept):
            for m in re.finditer(rf"(?<!\w){re.escape(form)}(?!\w)", a):
                cands.append({"label": label, "form": form,
                              "start": m.start(), "len": len(form)})
    if not cands:
        flags.append("categorical_unrecognized")
        return False, f"answer {a!r} matched no accepted form of gold {g!r}", flags

    live = [c for c in cands if not _is_negated(a, c["start"], c["form"])]
    if not live:
        flags.append("negated_match")
        forms = sorted({c["form"] for c in cands})
        return False, (
            f"answer {a!r} mentions {forms} but every match is negated; "
            f"scored as not matching gold {g!r}"
        ), flags

    if len(live) < len(cands):
        flags.append("negation_filtered")

    best = max(live, key=lambda c: (c["len"], -c["start"]))
    if best["label"] == gold:
        if best["form"] != a:
            flags.append("categorical_substring_match")
        return True, (
            f"answer {a!r} matched gold {g!r} via form {best['form']!r}"
            + (f" (most specific of {len(live)} matches)" if len(live) > 1 else "")
        ), flags

    return False, (
        f"answer {a!r} matched competing label {best['label']!r} via form "
        f"{best['form']!r}, gold was {g!r}"
    ), flags


def parse_list(answer: str) -> list[str]:
    """Split a comma/semicolon/newline separated answer into normalized parts."""
    if answer is None:
        return []
    parts = re.split(r"[,;\n]| and (?=\S)", str(answer))
    out = []
    for p in parts:
        n = normalize_categorical(p)
        if n and n not in out:
            out.append(n)
    return out


def compare_list(answer: str, gold_list, strict: bool = True) -> tuple[bool, float, str, dict]:
    """Set comparison. Returns (passed, f1, statement, detail).

    Scoring is the strict exact-set result so that every family-level score
    stays a Bernoulli trial and the Wilson interval remains valid. F1 is
    computed and recorded for diagnosis but does not drive the score unless
    `strict` is False.
    """
    a = set(parse_list(answer))
    g = {normalize_categorical(x) for x in (gold_list or [])}
    g = {x for x in g if x}
    inter = a & g
    prec = len(inter) / len(a) if a else 0.0
    rec = len(inter) / len(g) if g else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    exact = a == g
    detail = {
        "answer_set": sorted(a), "gold_set": sorted(g),
        "missing": sorted(g - a), "extra": sorted(a - g),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "exact_set_match": exact,
    }
    stmt = (
        f"{'PASS' if (exact if strict else f1 >= 1.0) else 'FAIL'} set compare: "
        f"{len(inter)}/{len(g)} gold members recovered, {len(a - g)} extra; F1={f1:.3f}"
    )
    return (exact if strict else f1 >= 0.999), (1.0 if exact else (f1 if not strict else 0.0)), stmt, detail
