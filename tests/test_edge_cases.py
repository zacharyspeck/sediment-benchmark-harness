"""Adversarial regression cases from fixtures/edge_cases.yaml.

Separate from the canonical 30-item fixture set. Every case here is something
that broke a grader, nearly broke one, or is a way a real model output could
defeat a naive implementation.
"""

from __future__ import annotations

import os

import pytest

from graders.base import GradeContext, grade_item
from harness.runner import load_items

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE_FILE = os.path.join(REPO, "fixtures", "edge_cases.yaml")
ARM = "edge"


@pytest.fixture(scope="session")
def edge_items():
    return load_items(EDGE_FILE)


def _grade(item, ledger, grading_config, scorer):
    ctx = GradeContext(ledger=ledger, config=grading_config, arm=ARM)
    response = item["responses"][ARM]
    g = grade_item(item, response, ctx)
    fab = scorer.score(response, question=item.get("question", ""))
    g.fabrication = fab.to_dict()
    return g


def test_edge_file_is_populated(edge_items):
    assert len(edge_items) >= 20
    for it in edge_items:
        assert ARM in (it.get("responses") or {}), f"{it['item_id']} has no `edge` response"
        assert (it.get("expect") or {}).get(ARM), f"{it['item_id']} has no expectation"


def test_edge_cases(edge_items, ledger, grading_config, scorer):
    failures = []
    for item in edge_items:
        iid = item["item_id"]
        exp = item["expect"][ARM]
        try:
            g = _grade(item, ledger, grading_config, scorer)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{iid}: grader raised {type(exc).__name__}: {exc}")
            continue

        if g.error:
            failures.append(f"{iid}: errored: {g.error[:200]}")

        if "score" in exp and abs(g.score - float(exp["score"])) > 1e-9:
            failures.append(f"{iid}: score {g.score} != {exp['score']}  [{g.comparison}]")
        if "verdict" in exp and g.verdict != exp["verdict"]:
            failures.append(f"{iid}: verdict {g.verdict!r} != {exp['verdict']!r}")
        if "extraction_uncertain" in exp:
            got = bool(g.extraction.get("extraction_uncertain"))
            if got != bool(exp["extraction_uncertain"]):
                failures.append(
                    f"{iid}: extraction_uncertain {got} != {exp['extraction_uncertain']} "
                    f"(method={g.extraction.get('method')})")
        for flag in (exp.get("flags_include") or []):
            if flag not in g.flags:
                failures.append(f"{iid}: flag {flag!r} missing from {g.flags}")
        if "fabricated" in exp:
            got = bool(g.fabrication.get("fabricated"))
            if got != bool(exp["fabricated"]):
                surfaces = [f["surface"] for f in g.fabrication.get("flags", [])
                            if f.get("tier") == "A"]
                failures.append(
                    f"{iid}: fabricated {got} != {exp['fabricated']} (tier-A: {surfaces})")
        for surface in (exp.get("fabricated_surfaces_include") or []):
            surfaces = [f["surface"] for f in g.fabrication.get("flags", [])
                        if f.get("tier") == "A"]
            if not any(surface.casefold() in s.casefold() for s in surfaces):
                failures.append(f"{iid}: expected fabrication {surface!r}, got {surfaces}")
        if "outcome" in exp and (g.detail or {}).get("outcome") != exp["outcome"]:
            failures.append(
                f"{iid}: outcome {(g.detail or {}).get('outcome')!r} != {exp['outcome']!r}")
        if "perfect_conformance" in exp:
            got = bool((g.detail or {}).get("perfect_conformance"))
            if got != bool(exp["perfect_conformance"]):
                failures.append(
                    f"{iid}: perfect_conformance {got} != {exp['perfect_conformance']}; "
                    f"failed={(g.detail or {}).get('failed_check_ids')}")

    assert not failures, (
        f"{len(failures)} edge case(s) not met:\n  - " + "\n  - ".join(failures))


# --------------------------------------------------------------------------
# The negation family, called out because it is the one that could have
# invalidated a headline result.
# --------------------------------------------------------------------------
NEGATION_CASES = [
    ("we would not advance this", "advance", False),
    ("we would not advance this", "pass", True),
    ("this is not a pass", "pass", False),
    ("do not advance", "advance", False),
    ("do not advance", "pass", True),
    ("this is not a pass, we advance", "advance", True),
    ("advance, not pass", "advance", True),
    ("we would pass rather than advance", "pass", True),
    ("we would pass rather than advance", "advance", False),
    ("pass", "pass", True),
    ("advance", "advance", True),
    ("we pass", "pass", True),
    ("proceed", "advance", True),
    ("decline", "pass", True),
]


@pytest.mark.parametrize("answer,gold,expected", NEGATION_CASES)
def test_negation_is_not_scored_as_agreement(answer, gold, expected):
    from graders.implicit_rule_application import DEFAULT_ACCEPT
    from graders.numeric import compare_categorical

    got, stmt, flags = compare_categorical(answer, gold, DEFAULT_ACCEPT)
    assert got is expected, f"{answer!r} vs gold {gold!r}: {stmt}"


def test_longest_match_decides_the_label():
    """"would not advance" (a pass form) must beat "advance"."""
    from graders.implicit_rule_application import DEFAULT_ACCEPT
    from graders.numeric import compare_categorical

    ok, stmt, _ = compare_categorical("we would not advance", "pass", DEFAULT_ACCEPT)
    assert ok
    assert "not advance" in stmt


def test_negated_only_match_is_flagged_not_silently_wrong():
    from graders.numeric import compare_categorical

    ok, stmt, flags = compare_categorical("not a pass", "pass", {"pass": ["pass"]})
    assert ok is False
    assert "negated_match" in flags
    assert "negated" in stmt
