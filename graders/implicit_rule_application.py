"""Family 4 -- implicit_rule_application.

A holdout target that appears in NO document. Given its facts, does it pass the
screen? The written criteria say one thing; the firm's actual behaviour says
another. Gold is the outcome under the implicit rules, taken from
holdout_targets[]. Grading is a categorical match on the verdict.

WHY THE RAW SCORE HERE IS MISLEADING ON ITS OWN
-----------------------------------------------
The holdout set mixes two kinds of target:

  discriminating   the written criteria and the firm's actual behaviour give
                   DIFFERENT answers. Only these separate a system that
                   absorbed how the firm behaves from one that read the
                   published criteria and applied them.
  clean            both give the SAME answer. A system that does nothing but
                   apply the written screens scores 100% on these.

Roughly half the holdout set is clean, so a pure written-criteria reasoner
scores near 50% overall while getting every interesting item wrong. Reporting
one blended number would hide exactly the effect this family exists to measure.

So every grade records:
  - `discriminates`             is this a separating item
  - `written_criteria_verdict`  what the published screens alone would answer
  - `matched_written_criteria`  did the model answer that instead of gold

and the report breaks accuracy out by discriminating vs clean, and prints the
written-criteria baseline next to each arm. An arm whose accuracy sits at the
baseline learned the documents, not the behaviour.

The score itself stays a plain binary verdict match, so the Wilson interval
remains valid.
"""

from __future__ import annotations

import re

from harness.extractor import extract_answer, prose_only
from ._valuecore import _gold_of
from .base import (
    Grade,
    GradeContext,
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_NO_ANSWER,
    register,
)
from .numeric import compare_categorical, normalize_categorical

GRADER = "implicit_rule_application"
VERSION = "1.0.0"

# The two verdicts the holdout set uses, with the surface forms a model may
# reasonably produce for each. Templates can extend this via gold.accept.
DEFAULT_ACCEPT = {
    "pass": ["pass", "we pass", "passes", "decline", "declines", "no", "reject",
             "rejects", "do not advance", "does not advance", "would not advance",
             "not advance", "would pass", "pass on it", "pass on this", "turn down"],
    "advance": ["advance", "advances", "would advance", "proceed", "proceeds",
                "yes", "pursue", "pursues", "move forward", "moves forward",
                "take it forward", "advance to diligence", "advance it"],
}


def _rule_citation_check(prose: str, rule_ids, ledger) -> dict:
    """Did the response name the behaviour behind the verdict?

    Pure string matching against the `paraphrases` list each implicit rule
    carries in the ledger. This is a DIAGNOSTIC ONLY -- it never moves the
    score. Getting the verdict right for the wrong reason still counts as
    right, because that is what a buyer experiences.
    """
    out = {"checked": False, "rule_ids": list(rule_ids or []), "cited": [], "hits": []}
    if not ledger or not rule_ids:
        return out
    # Collapse whitespace on both sides. A paraphrase like "half the stated
    # screen" must still match when the model's line wrapping happens to break
    # it across two lines -- otherwise the diagnostic reports "did not cite the
    # rule" for a response that cited it perfectly well.
    low = re.sub(r"\s+", " ", (prose or "")).casefold()
    if not low.strip():
        return out
    out["checked"] = True
    for rid in rule_ids:
        rule = ledger.implicit_rule(rid) if hasattr(ledger, "implicit_rule") else None
        if not rule:
            continue
        for phrase in (rule.get("paraphrases") or []):
            p = re.sub(r"\s+", " ", str(phrase)).casefold().strip()
            if p and p in low:
                out["cited"].append(rid)
                out["hits"].append({"rule_id": rid, "paraphrase": phrase})
                break
    out["cited"] = sorted(set(out["cited"]))
    out["cited_any"] = bool(out["cited"])
    return out


@register(GRADER)
def grade(item: dict, response: str, ctx: GradeContext) -> Grade:
    gold = _gold_of(item)
    meta = dict(item.get("meta") or {})

    accept = dict(DEFAULT_ACCEPT)
    for k, v in (gold.get("accept") or {}).items():
        accept[k] = list({*(accept.get(k) or []), *(v or [])})
    vocabulary = sorted({*accept.keys(), *[x for v in accept.values() for x in v]}, key=len, reverse=True)

    ex = extract_answer(response, answer_type="categorical", vocabulary=vocabulary)
    flags = list(ex.flags)
    if ex.uncertain:
        flags.append("extraction_uncertain")

    gold_verdict = str(gold.get("value") or meta.get("ground_truth_outcome") or "").strip()
    written = str(meta.get("written_criteria_verdict") or "").strip()
    discriminates = meta.get("discriminates")
    if discriminates is None and written and gold_verdict:
        discriminates = normalize_categorical(written) != normalize_categorical(gold_verdict)

    prose = prose_only(response or "", ex)
    citation = _rule_citation_check(prose, meta.get("driving_rule_ids"), ctx.ledger)

    base = dict(
        item_id=item.get("item_id", "<no-id>"),
        family=item.get("family", GRADER),
        arm=ctx.arm,
        grader=GRADER,
        grader_version=VERSION,
        binary=True,
        extracted_answer=ex.value,
        gold_answer=gold_verdict,
        extraction=ex.to_trace(),
        gold_trace=dict(gold.get("trace") or {}),
        raw_response=response or "",
        meta=meta,
    )

    detail = {
        "holdout_id": meta.get("holdout_id"),
        "discriminates": bool(discriminates),
        "bucket": meta.get("bucket"),
        "written_criteria_verdict": written or None,
        "written_criteria_failures": meta.get("written_criteria_failures"),
        "driving_rule_ids": meta.get("driving_rule_ids"),
        "rule_citation": citation,
        "accept_map": accept,
    }

    if not ex.found:
        return Grade(
            score=0.0, verdict=VERDICT_NO_ANSWER,
            comparison="no verdict could be extracted from the response",
            flags=flags,
            detail={**detail, "matched_written_criteria": None,
                    "extraction_notes": ex.notes},
            **base,
        )

    passed, stmt, cflags = compare_categorical(ex.value, gold_verdict, accept)
    flags.extend(f for f in cflags if f not in flags)

    matched_written = None
    if written:
        mw, _, _ = compare_categorical(ex.value, written, accept)
        matched_written = bool(mw)
        if discriminates and matched_written and not passed:
            flags.append("answered_the_written_criteria")

    detail["matched_written_criteria"] = matched_written

    comparison = (
        f"{stmt} | item {'DISCRIMINATES' if discriminates else 'is clean'} "
        f"(written criteria say {written or 'n/a'!r}, actual behaviour says {gold_verdict!r})"
    )

    return Grade(
        score=1.0 if passed else 0.0,
        verdict=VERDICT_CORRECT if passed else VERDICT_INCORRECT,
        comparison=comparison,
        flags=flags,
        detail=detail,
        **base,
    )
