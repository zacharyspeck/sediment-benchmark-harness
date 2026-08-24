"""Shared machinery for the families whose gold answer is a single value.

single_fact_recall, multi_doc_aggregation and corpus_wide_statistic differ in
what the gold answer *means* and in how much tolerance is defensible, not in
how a value is compared. Each family keeps its own module, its own registered
grader and its own defaults; the comparison itself lives here so there is one
place where "did the model get the number right" is decided.
"""

from __future__ import annotations

from typing import Any

from harness.extractor import extract_answer
from .base import (
    Grade,
    GradeContext,
    VERDICT_CORRECT,
    VERDICT_INCORRECT,
    VERDICT_NO_ANSWER,
)
from .numeric import compare_categorical, compare_list, compare_numeric, parse_quantity


def _gold_of(item: dict) -> dict:
    g = item.get("gold")
    if g is None:
        raise ValueError(f"item {item.get('item_id')!r} has no `gold` block")
    if not isinstance(g, dict):
        return {"value": g}
    return g


def grade_value_item(
    item: dict,
    response: str,
    ctx: GradeContext,
    grader: str,
    version: str,
    default_tol: dict,
) -> Grade:
    """Grade a single-value item (numeric, categorical or list)."""
    gold = _gold_of(item)
    answer_type = str(item.get("answer_type") or gold.get("type") or "numeric").lower()
    unit = str(gold.get("unit") or "none")
    vocabulary = list(gold.get("vocabulary") or (gold.get("accept") or {}).keys() or [])
    if answer_type in ("categorical", "verdict") and gold.get("value") is not None:
        if str(gold["value"]) not in vocabulary:
            vocabulary = list(vocabulary) + [str(gold["value"])]

    ex = extract_answer(response, answer_type=answer_type, vocabulary=vocabulary)
    flags = list(ex.flags)
    if ex.uncertain:
        flags.append("extraction_uncertain")

    base = dict(
        item_id=item.get("item_id", "<no-id>"),
        family=item.get("family", ""),
        arm=ctx.arm,
        grader=grader,
        grader_version=version,
        binary=True,
        extracted_answer=ex.value,
        gold_answer=gold.get("value"),
        extraction=ex.to_trace(),
        gold_trace=dict(gold.get("trace") or {}),
        raw_response=response or "",
        meta=dict(item.get("meta") or {}),
    )

    if not ex.found:
        return Grade(
            score=0.0,
            verdict=VERDICT_NO_ANSWER,
            comparison="no answer could be extracted from the response",
            flags=flags,
            detail={"answer_type": answer_type, "unit": unit,
                    "extraction_notes": ex.notes},
            **base,
        )

    # ---------------- numeric ----------------
    if answer_type in ("numeric", "number", "float", "int", "percent",
                       "multiple", "money", "count", "years", "months"):
        tol = ctx.tolerance(item, default_abs=default_tol.get("abs", 0.0),
                            default_rel=default_tol.get("rel", 0.0))
        # A count is a count. Never grant tolerance on an integer answer.
        if unit == "count" or answer_type == "count":
            item_tol = (item.get("grading") or {}).get("tolerance") or {}
            if not item_tol:
                tol = {"abs": 0.0, "rel": 0.0}
        q = parse_quantity(ex.value, expected_unit=unit)
        cmpres = compare_numeric(q, gold.get("value"), tol)
        flags.extend(f for f in cmpres.flags if f not in flags)
        return Grade(
            score=1.0 if cmpres.passed else 0.0,
            verdict=VERDICT_CORRECT if cmpres.passed else VERDICT_INCORRECT,
            comparison=cmpres.statement,
            flags=flags,
            detail={
                "answer_type": answer_type,
                "unit": unit,
                "parsed_quantity": q.to_trace(),
                "numeric_comparison": cmpres.to_trace(),
            },
            **base,
        )

    # ---------------- list ----------------
    if answer_type in ("list", "set"):
        strict = bool((item.get("grading") or {}).get("strict_set", True))
        passed, score, stmt, detail = compare_list(ex.value, gold.get("value") or [], strict=strict)
        return Grade(
            score=1.0 if passed else (0.0 if strict else score),
            verdict=VERDICT_CORRECT if passed else VERDICT_INCORRECT,
            comparison=stmt,
            flags=flags,
            detail={"answer_type": answer_type, "set_comparison": detail, "strict_set": strict},
            **base,
        )

    # ---------------- categorical ----------------
    accept = dict(gold.get("accept") or {})
    passed, stmt, cflags = compare_categorical(ex.value, str(gold.get("value")), accept)
    flags.extend(f for f in cflags if f not in flags)
    return Grade(
        score=1.0 if passed else 0.0,
        verdict=VERDICT_CORRECT if passed else VERDICT_INCORRECT,
        comparison=stmt,
        flags=flags,
        detail={"answer_type": answer_type, "accept_map": accept},
        **base,
    )
