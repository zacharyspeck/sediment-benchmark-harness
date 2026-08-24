"""Grade record, grading context, and the family registry."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

#: The six SCORED families, of equal standing. Never averaged together.
FAMILIES = (
    "single_fact_recall",
    "multi_doc_aggregation",
    "corpus_wide_statistic",
    "implicit_rule_application",
    "convention_conformance",
    "absence_and_abstention",
)

#: Slices that are graded and reported but are NOT one of the six.
#:
#: `recency` asks which of two values a file states is still true. It is graded
#: with the numeric grader -- the answer is a number with a tolerance -- but it
#: is deliberately not a family: it would otherwise be folded into
#: single_fact_recall's score, and it measures something different (currency,
#: not retrieval). Keeping it out of FAMILIES is what guarantees that.
EXTRA_SLICES = ("recency",)

#: Everything the harness knows how to grade.
GRADEABLE = FAMILIES + EXTRA_SLICES

# Verdicts. `no_answer` and `error` are scored 0 but reported separately so a
# harness failure is never quietly counted as a model failure.
VERDICT_CORRECT = "correct"
VERDICT_INCORRECT = "incorrect"
VERDICT_PARTIAL = "partial"
VERDICT_NO_ANSWER = "no_answer"
VERDICT_ERROR = "error"


@dataclass
class Grade:
    """The full, inspectable result of grading one response."""

    item_id: str
    family: str
    arm: str
    grader: str
    grader_version: str

    score: float                    # 0..1
    binary: bool                    # True => a Bernoulli trial (Wilson applies)
    verdict: str

    extracted_answer: Any = None
    gold_answer: Any = None
    comparison: str = ""            # plain-English statement of what was compared
    flags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    extraction: dict = field(default_factory=dict)
    gold_trace: dict = field(default_factory=dict)
    raw_response: str = ""
    error: str | None = None

    # filled in by the runner
    fabrication: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def correct(self) -> bool | None:
        if self.verdict == VERDICT_ERROR:
            return None
        return self.score >= 1.0 if self.binary else None

    @property
    def extraction_uncertain(self) -> bool:
        return bool(self.extraction.get("extraction_uncertain"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["correct"] = self.correct
        d["extraction_uncertain"] = self.extraction_uncertain
        return d


@dataclass
class GradeContext:
    """Everything a grader may read. Deliberately small."""

    ledger: Any = None                     # harness.ledger.Ledger (or None)
    config: dict = field(default_factory=dict)
    arm: str = "unknown"

    def tolerance(self, item: dict, default_abs=0.0, default_rel=0.01) -> dict:
        """Item tolerance, falling back to family config then global default."""
        fam = item.get("family", "")
        cfg = self.config or {}
        fam_tol = ((cfg.get("families") or {}).get(fam) or {}).get("tolerance") or {}
        glob_tol = (cfg.get("defaults") or {}).get("tolerance") or {}
        item_tol = (item.get("grading") or {}).get("tolerance") or {}
        out = {"abs": default_abs, "rel": default_rel}
        for src in (glob_tol, fam_tol, item_tol):
            for k in ("abs", "rel"):
                if k in src and src[k] is not None:
                    out[k] = float(src[k])
        return out


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[[dict, str, GradeContext], Grade]] = {}


def register(family: str):
    def deco(fn):
        _REGISTRY[family] = fn
        return fn
    return deco


def get_grader(family: str):
    if family not in _REGISTRY:
        raise KeyError(
            f"no grader registered for family {family!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[family]


def grade_item(item: dict, response: str, ctx: GradeContext) -> Grade:
    """Grade one item, converting any grader crash into an `error` Grade.

    Per-item try/except is the only place exceptions are caught, and the
    traceback is preserved verbatim in the record. Nothing is swallowed.
    """
    family = item.get("family", "")
    item_id = item.get("item_id", "<no-id>")
    try:
        fn = get_grader(family)
    except KeyError as exc:
        return Grade(
            item_id=item_id, family=family, arm=ctx.arm, grader="<none>",
            grader_version="0", score=0.0, binary=True, verdict=VERDICT_ERROR,
            comparison="no grader for this family", error=str(exc),
            raw_response=response or "",
        )
    try:
        g = fn(item, response, ctx)
        g.arm = ctx.arm
        return g
    except Exception as exc:  # noqa: BLE001 - deliberate: one bad item must not end the run
        return Grade(
            item_id=item_id, family=family, arm=ctx.arm, grader=family,
            grader_version="0", score=0.0, binary=True, verdict=VERDICT_ERROR,
            comparison="grader raised",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            raw_response=response or "",
        )
