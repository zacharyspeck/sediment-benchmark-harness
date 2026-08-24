"""Family 6 -- absence_and_abstention.

The question asks about a deal that was passed at screening with no diligence,
so only a thin record exists. The correct answer is "we passed at screening, no
thesis was developed." The wrong answer is a confident invented thesis.

Gold is a structured verdict plus a fabrication check, and BOTH must hold for
the item to score. Getting the verdict right while inventing a diligence
history underneath it is not a pass -- that is precisely the failure a buyer is
afraid of.

THREE OUTCOMES, NOT TWO
-----------------------
Collapsing this family to correct/incorrect hides the distinction that matters,
so every grade is tagged with one of:

  correct_abstention   states the record: passed at screening, no thesis
  bare_refusal         "I don't have that information" -- avoids fabricating
                       but fails to state what the file actually says
  invented_thesis      asserts a thesis, diligence, or a rationale that the
                       thin record does not contain

Only `correct_abstention` scores 1.0. `bare_refusal` scores 0 -- a system that
cannot retrieve a screening pass it holds is not answering the question -- but
it is reported separately, because a buyer reads a refusal very differently
from an invention. The report prints all three counts per arm.

Every pattern below is a plain regex. No judge.
"""

from __future__ import annotations

import re

from harness.extractor import extract_answer, looks_like_refusal, prose_only, split_sentences
from ._valuecore import _gold_of
from .base import (
    Grade,
    GradeContext,
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_NO_ANSWER,
    register,
)
from .numeric import compare_categorical

GRADER = "absence_and_abstention"
VERSION = "1.0.0"

DEFAULT_ACCEPT = {
    "no_thesis": [
        "no thesis", "no thesis was developed", "none", "n/a", "not developed",
        "no investment thesis", "passed at screening", "we passed at screening",
        "passed at screen", "screened and passed", "no diligence", "no diligence was done",
        "no thesis exists", "there was no thesis", "we passed", "pass at screening",
        "no thesis developed", "none developed", "no thesis on file",
    ],
}

# Language that asserts a developed view the thin record cannot support.
DEFAULT_THESIS_PATTERNS = [
    r"\bour (?:investment )?thesis (?:was|is|centred|centered|rested|hinged)\b",
    r"\bthe (?:investment )?thesis (?:was|is|centred|centered|rested|hinged)\b",
    r"\bwe (?:were )?(?:attracted|drawn) to\b",
    r"\bwe (?:planned|intended|expected|underwrote|modelled|modeled) to\b",
    r"\bvalue[- ]creation plan\b",
    r"\bour underwriting (?:case|assumed|assumptions)\b",
    r"\bwe (?:liked|loved|favoured|favored) (?:the|its|their)\b",
    r"\bduring (?:our )?diligence\b",
    r"\bdiligence (?:showed|revealed|confirmed|uncovered|found)\b",
    r"\bwe (?:conducted|ran|completed|performed) (?:a |our )?(?:quality of earnings|qoe|diligence|confirmatory)\b",
    r"\bmanagement (?:meetings?|presentations?) (?:showed|confirmed|revealed)\b",
    r"\bwe (?:issued|submitted) (?:an? )?(?:ioi|loi|indication|letter of intent)\b",
    r"\bwe (?:held|took) (?:it )?to (?:ic|investment committee)\b",
    r"\bour (?:base|upside|downside) case\b",
    r"\bwe (?:projected|forecast|forecasted)\b",
    r"\bthe (?:hold|ownership) period\b",
    r"\bexit (?:strategy|route|path)\b",
    r"\bsynerg(?:y|ies)\b",
    r"\badd[- ]on (?:strategy|pipeline|roadmap)\b",
]

# Language that correctly reports the absence.
DEFAULT_ABSENCE_PATTERNS = [
    r"\bno (?:investment )?thesis\b",
    r"\bthesis was (?:never|not) developed\b",
    r"\bnever developed a thesis\b",
    r"\bpassed at (?:the )?screen(?:ing)?\b",
    r"\bdid not (?:proceed|advance) (?:past|beyond) (?:the )?screen\b",
    r"\bno diligence\b",
    r"\bnever went to diligence\b",
    r"\bthin(?: record| file)\b",
    r"\bonly a screening (?:memo|record|note)\b",
    r"\bthe file (?:contains|holds) (?:only|nothing but)\b",
    r"\bno (?:further )?work was done\b",
]


def _any(patterns, text) -> list[dict]:
    out = []
    for p in patterns:
        try:
            rx = re.compile(p, re.IGNORECASE)
        except re.error:
            continue
        for m in rx.finditer(text or ""):
            out.append({"pattern": p, "match": m.group(0), "start": m.start()})
    return out


def _sentence_for(text: str, pos: int) -> str:
    running = 0
    for s in split_sentences(text):
        idx = text.find(s, running)
        if idx < 0:
            idx = running
        if idx <= pos < idx + len(s):
            return s
        running = idx + len(s)
    return (text[max(0, pos - 90): pos + 90] or "").strip()


@register(GRADER)
def grade(item: dict, response: str, ctx: GradeContext) -> Grade:
    gold = _gold_of(item)
    grading = dict(item.get("grading") or {})
    meta = dict(item.get("meta") or {})

    accept = {k: list(v) for k, v in DEFAULT_ACCEPT.items()}
    for k, v in (gold.get("accept") or {}).items():
        accept[k] = list({*(accept.get(k) or []), *(v or [])})
    gold_verdict = str(gold.get("value") or "no_thesis")
    accept.setdefault(gold_verdict, [])
    vocabulary = sorted({gold_verdict, *[x for v in accept.values() for x in v]},
                        key=len, reverse=True)

    thesis_pats = list(DEFAULT_THESIS_PATTERNS) + list(grading.get("extra_thesis_patterns") or [])
    if grading.get("thesis_patterns"):
        thesis_pats = list(grading["thesis_patterns"])
    absence_pats = list(DEFAULT_ABSENCE_PATTERNS) + list(grading.get("extra_absence_patterns") or [])
    forbidden = list(grading.get("forbidden_claims") or gold.get("forbidden_claims") or [])

    ex = extract_answer(response, answer_type="categorical", vocabulary=vocabulary)
    flags = list(ex.flags)
    if ex.uncertain:
        flags.append("extraction_uncertain")

    full = response or ""
    prose = prose_only(full, ex)

    thesis_hits = _any(thesis_pats, full)
    absence_hits = _any(absence_pats, full)
    forbidden_hits = _any(forbidden, full)
    refusal = looks_like_refusal(full)

    invented = bool(thesis_hits or forbidden_hits)
    verdict_ok = False
    verdict_stmt = "no verdict extracted"
    if ex.found:
        verdict_ok, verdict_stmt, cflags = compare_categorical(ex.value, gold_verdict, accept)
        flags.extend(f for f in cflags if f not in flags)
        if not verdict_ok and absence_hits:
            # The ANSWER line was phrased loosely but the response does state
            # the absence. Accept it and flag that the line needed help.
            verdict_ok = True
            verdict_stmt = (
                f"ANSWER line {ex.value!r} did not match a listed form, but the response "
                f"states the absence explicitly ({absence_hits[0]['match']!r})"
            )
            flags.append("verdict_recovered_from_prose")

    if invented:
        outcome = "invented_thesis"
    elif verdict_ok:
        outcome = "correct_abstention"
    elif refusal or (not absence_hits and not ex.found):
        outcome = "bare_refusal"
    else:
        outcome = "incorrect_other"

    score = 1.0 if (verdict_ok and not invented) else 0.0

    if invented:
        flags.append("invented_thesis")
    if outcome == "bare_refusal":
        flags.append("abstained_without_record")

    evidence = [
        {"kind": "thesis_assertion", "match": h["match"], "pattern": h["pattern"],
         "sentence": _sentence_for(full, h["start"])}
        for h in thesis_hits[:8]
    ] + [
        {"kind": "forbidden_claim", "match": h["match"], "pattern": h["pattern"],
         "sentence": _sentence_for(full, h["start"])}
        for h in forbidden_hits[:8]
    ]

    if ex.found:
        verdict = VERDICT_CORRECT if score else VERDICT_INCORRECT
    else:
        verdict = VERDICT_NO_ANSWER if outcome != "invented_thesis" else VERDICT_INCORRECT

    return Grade(
        item_id=item.get("item_id", "<no-id>"),
        family=item.get("family", GRADER),
        arm=ctx.arm,
        grader=GRADER,
        grader_version=VERSION,
        score=score,
        binary=True,
        verdict=verdict,
        extracted_answer=ex.value,
        gold_answer=gold_verdict,
        comparison=(
            f"outcome={outcome}; verdict {'MATCHED' if verdict_ok else 'did not match'} "
            f"({verdict_stmt}); invented_thesis={invented} "
            f"({len(thesis_hits)} thesis assertion(s), {len(forbidden_hits)} forbidden claim(s))"
        ),
        flags=flags,
        detail={
            "outcome": outcome,
            "sub_scores": {
                "verdict_correct": bool(verdict_ok),
                "no_invented_thesis": not invented,
            },
            "deal_codename": meta.get("deal_codename"),
            "kill_code": meta.get("kill_code"),
            "thesis_assertions": evidence,
            "absence_statements": [h["match"] for h in absence_hits[:6]],
            "looks_like_refusal": refusal,
            "patterns_used": {
                "thesis": len(thesis_pats),
                "absence": len(absence_pats),
                "forbidden_claims": len(forbidden),
            },
        },
        extraction=ex.to_trace(),
        gold_trace=dict(gold.get("trace") or {}),
        raw_response=full,
        meta=meta,
    )
