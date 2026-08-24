"""Family 2 -- multi_doc_aggregation.

The answer requires more documents than one search returns. "Average EBITDA
margin across healthcare deals screened in 2023." Gold is computed from the
ledger; grading is numeric within tolerance.

Tolerance is looser than single_fact_recall on purpose. A model that finds all
the right deals and averages them can still land a few basis points away
through display rounding in the source documents (the corpus restates some
percentages as integers). Tolerance absorbs rounding, not missing documents:
at a 2% relative band, dropping one deal out of ten moves the mean far enough
to fail. The trace records how many rows fed the gold so you can see exactly
what a correct answer had to cover.
"""

from __future__ import annotations

from ._valuecore import grade_value_item
from .base import Grade, GradeContext, register

GRADER = "multi_doc_aggregation"
VERSION = "1.0.0"

DEFAULT_TOLERANCE = {"abs": 0.1, "rel": 0.02}


@register(GRADER)
def grade(item: dict, response: str, ctx: GradeContext) -> Grade:
    g = grade_value_item(item, response, ctx, GRADER, VERSION, DEFAULT_TOLERANCE)
    gt = g.gold_trace or {}
    g.detail["aggregation"] = {
        "op": gt.get("op"),
        "field": gt.get("field"),
        "n_rows_in_gold": gt.get("n"),
        "contributing_ids": (gt.get("contributing_ids") or [])[:60],
        "truncated": bool(gt.get("truncated")),
    }
    g.detail["family_note"] = (
        "gold computed over %s ledger rows; tolerance absorbs display rounding, "
        "not a missing document" % gt.get("n")
    )
    return g
