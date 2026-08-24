"""Recency -- which of the values the file states is still true.

NOT ONE OF THE SIX FAMILIES. Graded here, reported on its own line, and never
folded into `single_fact_recall`: it measures currency rather than retrieval,
and the two fail differently. A model can be excellent at finding a stated
figure and still hand back one that stopped being true three years ago.

The grading itself is the ordinary numeric comparison -- the answer is a number
with a tolerance -- so this module is a thin wrapper over the shared value
grader rather than a new mechanism. What makes the slice hard is the ITEM, not
the grader: the wrong answer is a real figure, stated confidently, in a real
document, by a named author, and true when it was written.
"""

from __future__ import annotations

from ._valuecore import grade_value_item
from .base import Grade, GradeContext, register

GRADER = "recency"
VERSION = "1.0.0"

# Same band as single_fact_recall: a one-decimal restatement is the same answer.
# It is deliberately NOT wide enough to swallow the difference between the
# current value and the superseded one -- that difference is the whole item, and
# a tolerance that absorbed it would score the stale answer correct.
DEFAULT_TOLERANCE = {"abs": 0.05, "rel": 0.005}


@register(GRADER)
def grade(item: dict, response: str, ctx: GradeContext) -> Grade:
    g = grade_value_item(item, response, ctx, GRADER, VERSION, DEFAULT_TOLERANCE)
    g.detail["family_note"] = (
        "reported separately from the six families: the wrong answer here is a "
        "stale value that was true once, not a value that was never true"
    )
    meta = item.get("meta") or {}
    for k in ("source_row_id", "deal_codename"):
        if meta.get(k):
            g.detail[k] = meta[k]
    g.detail["recency"] = True
    return g
