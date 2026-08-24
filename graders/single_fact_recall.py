"""Family 1 -- single_fact_recall.

The answer sits in one document. "What multiple did Westbrook pay for Craygarth?"

Grading: numeric match with tolerance, or exact string for categoricals.

Expect roughly a tie between the arms. This family is here for credibility --
if a benchmark cannot show the two systems agreeing on facts that are plainly
retrievable, nobody should believe its harder families either. A large gap here
is more likely a harness bug than a capability difference, and the report says
so.

Tolerance is deliberately tight: the gold value is a literal ledger field, so
the only slack needed is for display rounding (one decimal is the house
convention), not for estimation.
"""

from __future__ import annotations

from ._valuecore import grade_value_item
from .base import Grade, GradeContext, register

GRADER = "single_fact_recall"
VERSION = "1.0.0"

# 0.05 absolute covers a one-decimal restatement; 0.5% relative covers larger
# magnitudes (a $150.0M revenue restated as $150M).
DEFAULT_TOLERANCE = {"abs": 0.05, "rel": 0.005}


@register(GRADER)
def grade(item: dict, response: str, ctx: GradeContext) -> Grade:
    g = grade_value_item(item, response, ctx, GRADER, VERSION, DEFAULT_TOLERANCE)
    g.detail["family_note"] = (
        "single retrievable fact; both arms can search the corpus, so a tie is the "
        "expected result and a large gap warrants a harness check before a claim"
    )
    src = (item.get("meta") or {}).get("source_deal")
    if src:
        g.detail["source_deal"] = src
    return g
