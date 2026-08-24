"""Graders: one module per question family, plus the fabrication scorer.

Every grader returns a `Grade` (graders/base.py) carrying a full trace: the raw
response, what was extracted, the gold answer, the comparison that was
performed, the verdict, and any flags. A wrong score must be explainable by
reading one JSON record -- never by attaching a debugger.

NO LLM JUDGE LIVES HERE, AND NONE MAY BE ADDED. Every check in this package is
a numeric comparison, a string comparison, or a regex operation.
"""

from .base import (  # noqa: F401
    EXTRA_SLICES,
    FAMILIES,
    GRADEABLE,
    Grade,
    GradeContext,
    get_grader,
    grade_item,
    register,
)
from . import (  # noqa: F401  (import for registration side effects)
    single_fact_recall,
    multi_doc_aggregation,
    corpus_wide_statistic,
    implicit_rule_application,
    convention_conformance,
    absence_and_abstention,
    recency,
)

__all__ = ["Grade", "GradeContext", "get_grader", "grade_item", "register",
           "FAMILIES", "EXTRA_SLICES", "GRADEABLE"]
