"""Family 3 -- corpus_wide_statistic.

Requires the whole corpus. "Median hold period across all exits." "How many
deals were passed on in specialty manufacturing?" Gold computed from the
ledger; grading numeric within tolerance.

TWO KINDS OF ANSWER, TWO TOLERANCES
-----------------------------------
A count is exact. If the question is "how many deals were passed on in
specialty manufacturing" the answer is an integer and 86 is not 87; the grader
forces zero tolerance whenever the gold unit is `count`, regardless of the
family default. Continuous statistics (medians, means, hold periods) keep a
small relative band for rounding.

This distinction is the whole reason counts are not graded like averages: a
tolerance on a count silently awards credit for miscounting.
"""

from __future__ import annotations

from ._valuecore import grade_value_item
from .base import Grade, GradeContext, register

GRADER = "corpus_wide_statistic"
VERSION = "1.0.0"

DEFAULT_TOLERANCE = {"abs": 0.1, "rel": 0.02}


@register(GRADER)
def grade(item: dict, response: str, ctx: GradeContext) -> Grade:
    g = grade_value_item(item, response, ctx, GRADER, VERSION, DEFAULT_TOLERANCE)
    gold = item.get("gold") or {}
    unit = str(gold.get("unit") or "none")
    gt = g.gold_trace or {}
    g.detail["statistic"] = {
        "op": gt.get("op"),
        "field": gt.get("field"),
        "population_n": gt.get("n"),
        "unit": unit,
        "exact_required": unit == "count",
    }
    g.detail["family_note"] = (
        "counts are graded exactly; continuous statistics carry a rounding band"
        if unit == "count" else
        "continuous corpus-wide statistic; tolerance covers rounding only"
    )
    return g
