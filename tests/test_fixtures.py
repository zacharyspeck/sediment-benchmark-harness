"""Every grader, proved against a correct AND an incorrect fixture.

This is the test that matters. Each of the 30 fixture items carries two
hand-written responses and an `expect` block; this file asserts the graders
return exactly what the fixtures say they must. A grader that changes
behaviour fails here rather than quietly changing a benchmark result.
"""

from __future__ import annotations

import pytest

from graders.base import GradeContext, grade_item
from graders.base import FAMILIES

ARMS = ("fixture_strong", "fixture_weak")


def _cases(items):
    out = []
    for it in items:
        for arm in ARMS:
            if arm in (it.get("responses") or {}):
                out.append((it, arm))
    return out


def _grade(item, arm, ledger, grading_config, scorer):
    ctx = GradeContext(ledger=ledger, config=grading_config, arm=arm)
    response = item["responses"][arm]
    g = grade_item(item, response, ctx)
    fab = scorer.score(response, question=item.get("question", ""),
                       extra_allow=list((item.get("grading") or {}).get("allow_entities") or []))
    g.fabrication = fab.to_dict()
    return g


def _ids(items):
    return [f"{it['item_id']}::{arm}" for it, arm in _cases(items)]


# --------------------------------------------------------------------------
def test_every_family_has_fixtures(items):
    seen = {it["family"] for it in items}
    missing = set(FAMILIES) - seen
    assert not missing, f"families with no fixture coverage: {sorted(missing)}"


def test_thirty_items_two_responses_each(items):
    assert len(items) == 30, f"expected 30 fixture items, found {len(items)}"
    for it in items:
        resp = it.get("responses") or {}
        for arm in ARMS:
            assert arm in resp, f"{it['item_id']} has no response for {arm}"
        assert it.get("expect"), f"{it['item_id']} has no expect block"


def test_every_family_has_a_correct_and_an_incorrect_fixture(items):
    """The explicit requirement: each grader proved on both sides."""
    by_family: dict[str, dict[str, bool]] = {}
    for it in items:
        fam = it["family"]
        slot = by_family.setdefault(fam, {"scores_1": False, "scores_0": False})
        for arm in ARMS:
            exp = (it.get("expect") or {}).get(arm) or {}
            if exp.get("score") == 1.0 or exp.get("perfect_conformance") is True:
                slot["scores_1"] = True
            if exp.get("score") == 0.0 or exp.get("perfect_conformance") is False \
                    or exp.get("score_max") is not None:
                slot["scores_0"] = True
    for fam in FAMILIES:
        assert by_family.get(fam, {}).get("scores_1"), \
            f"{fam} has no fixture that must score as CORRECT"
        assert by_family.get(fam, {}).get("scores_0"), \
            f"{fam} has no fixture that must score as INCORRECT"


# --------------------------------------------------------------------------
def test_fixture_expectations(items, ledger, grading_config, scorer):
    """The whole matrix at once, reporting every mismatch rather than the first."""
    failures = []
    for item, arm in _cases(items):
        exp = (item.get("expect") or {}).get(arm)
        if not exp:
            continue
        iid = f"{item['item_id']}::{arm}"
        try:
            g = _grade(item, arm, ledger, grading_config, scorer)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{iid}: grader raised {type(exc).__name__}: {exc}")
            continue

        if g.error:
            failures.append(f"{iid}: grade carried an error: {g.error[:300]}")

        if "score" in exp and abs(g.score - float(exp["score"])) > 1e-9:
            failures.append(
                f"{iid}: score {g.score} != expected {exp['score']}  [{g.comparison}]")

        if "verdict" in exp and g.verdict != exp["verdict"]:
            failures.append(
                f"{iid}: verdict {g.verdict!r} != expected {exp['verdict']!r}  [{g.comparison}]")

        if "score_min" in exp and g.score < float(exp["score_min"]) - 1e-9:
            failures.append(
                f"{iid}: score {g.score:.3f} below score_min {exp['score_min']}  "
                f"failed={g.detail.get('failed_check_ids')}")

        if "score_max" in exp and g.score > float(exp["score_max"]) + 1e-9:
            failures.append(
                f"{iid}: score {g.score:.3f} above score_max {exp['score_max']}  "
                f"failed={g.detail.get('failed_check_ids')}")

        if "extraction_uncertain" in exp:
            got = bool(g.extraction.get("extraction_uncertain"))
            if got != bool(exp["extraction_uncertain"]):
                failures.append(
                    f"{iid}: extraction_uncertain {got} != expected "
                    f"{exp['extraction_uncertain']} (method={g.extraction.get('method')})")

        for flag in (exp.get("flags_include") or []):
            if flag not in g.flags:
                failures.append(f"{iid}: flag {flag!r} missing from {g.flags}")

        if "fabricated" in exp:
            got = bool(g.fabrication.get("fabricated"))
            if got != bool(exp["fabricated"]):
                surfaces = [f["surface"] for f in g.fabrication.get("flags", [])
                            if f.get("tier") == "A"]
                failures.append(
                    f"{iid}: fabricated {got} != expected {exp['fabricated']} "
                    f"(tier-A surfaces: {surfaces})")

        for surface in (exp.get("fabricated_surfaces_include") or []):
            surfaces = [f["surface"] for f in g.fabrication.get("flags", [])
                        if f.get("tier") == "A"]
            if not any(surface.casefold() in s.casefold() for s in surfaces):
                failures.append(
                    f"{iid}: expected fabrication flag containing {surface!r}, got {surfaces}")

        if "outcome" in exp:
            got = (g.detail or {}).get("outcome")
            if got != exp["outcome"]:
                failures.append(f"{iid}: outcome {got!r} != expected {exp['outcome']!r}")

        if "perfect_conformance" in exp:
            got = (g.detail or {}).get("perfect_conformance")
            if bool(got) != bool(exp["perfect_conformance"]):
                failures.append(
                    f"{iid}: perfect_conformance {got} != expected "
                    f"{exp['perfect_conformance']}; failed checks="
                    f"{(g.detail or {}).get('failed_check_ids')}")

        if "rule_cited" in exp:
            got = ((g.detail or {}).get("rule_citation") or {}).get("cited_any")
            if bool(got) != bool(exp["rule_cited"]):
                failures.append(f"{iid}: rule_cited {got} != expected {exp['rule_cited']}")

    assert not failures, (
        f"{len(failures)} fixture expectation(s) not met:\n  - " + "\n  - ".join(failures)
    )


# --------------------------------------------------------------------------
def test_no_harness_errors_anywhere(items, ledger, grading_config, scorer):
    for item, arm in _cases(items):
        g = _grade(item, arm, ledger, grading_config, scorer)
        assert g.verdict != "error", f"{item['item_id']}::{arm} errored: {g.error}"


def test_every_grade_carries_a_full_trace(items, ledger, grading_config, scorer):
    """A wrong score must be inspectable without a debugger."""
    for item, arm in _cases(items):
        g = _grade(item, arm, ledger, grading_config, scorer)
        iid = f"{item['item_id']}::{arm}"
        assert g.raw_response, f"{iid}: trace has no raw response"
        assert g.comparison, f"{iid}: trace has no comparison statement"
        assert g.extraction, f"{iid}: trace has no extraction record"
        assert g.grader and g.grader_version, f"{iid}: trace has no grader identity"
        assert isinstance(g.detail, dict), f"{iid}: trace has no detail block"
        assert g.gold_answer is not None, f"{iid}: trace has no gold answer"


def test_gold_values_still_match_the_ledger(items, ledger):
    """The fixtures' computed golds must agree with the mini ledger.

    Fixture gold values were computed from the ledger through the query layer.
    If someone edits mini_ledger.json without re-deriving them, this fails --
    which is the whole point of asserting it rather than trusting a comment.
    """
    from harness.query import aggregate, select

    checks = [
        ("mda-fix-01", {"sector": "healthcare services", "year": 2023, "stage": "screen"},
         {"op": "mean", "field": "numbers.adj_ebitda_margin_pct", "round": 1}),
        ("mda-fix-02", {"sector": "specialty manufacturing", "outcome": "pass"},
         {"op": "count"}),
        ("mda-fix-03", {"kill_code": "C1"},
         {"op": "mean", "field": "numbers.revenue_m", "round": 2}),
        ("mda-fix-04", {"kill_code": "C1"},
         {"op": "median", "field": "numbers.top_customer_pct", "round": 1}),
        ("mda-fix-05", {"sector": "healthcare services", "year": 2023},
         {"op": "sum", "field": "numbers.adj_ebitda_m", "round": 2}),
        ("cws-fix-01", {"_derived.has_exit": {"op": "truthy", "value": True}},
         {"op": "median", "field": "_derived.hold_period_years", "round": 2}),
        ("cws-fix-02", {"sector": "specialty manufacturing", "outcome": "pass"},
         {"op": "count"}),
        ("cws-fix-03", {"firm": "westbrook", "numbers.entry_multiple": {"op": "not_null"}},
         {"op": "mean", "field": "numbers.entry_multiple", "round": 2}),
        ("cws-fix-04", {"firm": "westbrook", "outcome": "pass",
                        "kill_code": {"op": "not_null"}},
         {"op": "mode", "field": "kill_code"}),
        ("cws-fix-05", {"firm": "westbrook", "stage": "screen", "outcome": "pass"},
         {"op": "count"}),
    ]
    by_id = {it["item_id"]: it for it in items}
    for item_id, where, aggspec in checks:
        item = by_id[item_id]
        rows = select(ledger.deals, where)
        res = aggregate(rows, aggspec)
        gold = item["gold"]["value"]
        assert res.value == gold, (
            f"{item_id}: fixture gold {gold!r} but the ledger now computes {res.value!r} "
            f"over n={res.n}. Re-derive the fixture gold or revert the ledger edit."
        )


def test_single_fact_golds_match_the_ledger(items, ledger):
    by_id = {it["item_id"]: it for it in items}
    pairs = [
        ("sfr-fix-01", "Kilnmouth", ("numbers", "entry_multiple")),
        ("sfr-fix-02", "Craygarth", ("numbers", "top_customer_pct")),
        ("sfr-fix-04", "Foxgworth", ("numbers", "adj_ebitda_m")),
        ("sfr-fix-05", "Juniper", ("exit", "multiple")),
    ]
    for item_id, codename, path in pairs:
        deal = ledger.deal_by_codename(codename)
        assert deal, f"{codename} missing from the mini ledger"
        val = deal
        for part in path:
            val = val[part]
        assert by_id[item_id]["gold"]["value"] == val, (
            f"{item_id}: gold {by_id[item_id]['gold']['value']!r} != ledger {val!r}")


def test_holdout_golds_match_the_ledger(items, ledger):
    by_id = {it["item_id"]: it for it in items}
    for item_id in ("ira-fix-01", "ira-fix-02", "ira-fix-03", "ira-fix-04", "ira-fix-05"):
        item = by_id[item_id]
        hid = item["meta"]["holdout_id"]
        target = ledger.holdout_by_id(hid)
        assert target, f"{item_id}: holdout {hid} not in the mini ledger"
        assert item["gold"]["value"] == target["ground_truth_outcome"], (
            f"{item_id}: gold {item['gold']['value']!r} != ledger ground truth "
            f"{target['ground_truth_outcome']!r}")
        assert item["meta"]["written_criteria_verdict"] == target["written_criteria_verdict"]
        assert bool(item["meta"]["discriminates"]) == bool(target["discriminates"])


def test_conformance_fixture_scores_are_pinned(ledger, grading_config):
    """Exact scores, so a checklist edit cannot drift them silently.

    The threshold assertions elsewhere in this file are one-sided (`score_max`),
    which means a check removal that RAISES a weak draft's score can slide under
    them unnoticed until it crosses the bound. These are the measured values
    after the 2026-08-19 family-5 revision; if a checklist changes, this test
    fails and the number has to be re-derived deliberately.
    """
    from graders.base import GradeContext, grade_item

    expected = {
        ("ccf-fix-01", "fixture_strong"): 1.0000,
        ("ccf-fix-01", "fixture_weak"): 0.3333,
        ("ccf-fix-02", "fixture_strong"): 1.0000,
        ("ccf-fix-02", "fixture_weak"): 0.5000,
        ("ccf-fix-03", "fixture_strong"): 1.0000,
        ("ccf-fix-03", "fixture_weak"): 0.6000,
        ("ccf-fix-04", "fixture_strong"): 1.0000,
        ("ccf-fix-04", "fixture_weak"): 0.2500,
        ("ccf-fix-05", "fixture_strong"): 1.0000,
        ("ccf-fix-05", "fixture_weak"): 0.5000,
    }
    import os

    from harness.runner import load_items

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items = {i["item_id"]: i
             for i in load_items(os.path.join(repo, "fixtures", "fixture_items.yaml"))}
    seen = {}
    for (iid, arm), want in expected.items():
        item = items[iid]
        ctx = GradeContext(ledger=ledger, config=grading_config, arm=arm)
        got = grade_item(item, item["responses"][arm], ctx).score
        seen[(iid, arm)] = round(got, 4)
        assert abs(got - want) < 5e-4, (
            f"{iid}/{arm} scored {got:.4f}, pinned at {want:.4f}. "
            "A checklist changed: re-derive this value deliberately rather than "
            "adjusting it to whatever the code now produces."
        )


def test_every_conformance_template_separates_a_draft_from_a_non_answer(
        ledger, grading_config):
    """No checklist may score 'No comment.' as highly as a real draft.

    Three of the five did before the section-order cut: their remaining
    unconditional checks were all NEGATIVE, so an empty document passed them
    all, and the conditional ones dropped out of the denominator. The template
    still emitted verdict=correct and perfect_conformance=true while carrying no
    information.
    """
    from graders.base import GradeContext, grade_item
    from harness.cli import load_grading_config

    cfg = load_grading_config()
    refs = sorted((cfg.get("checklists") or {}))
    assert refs, "no checklists loaded"

    degenerate = "ANSWER: Project Ingodale\n\nNo comment."
    competent = (
        "ANSWER: drafted\n\nPass. Criterion one, customer concentration.\n\n"
        "Ardent Industrial A is a business services provider with revenue of $22.6M. "
        "The company holds multi-year contracts under an MSA with its top accounts; "
        "relationships average 7 years of tenure with the same staff team delivering "
        "the work. Gross retention is 96.1% and is verifiable from the contract set; "
        "renewal is automatic under each service agreement. Adjusted EBITDA is $4.9M, "
        "a margin of 21.7%. Concentration risk is present but mitigated by contracted "
        "revenue and a diversified renewal schedule across the book.\n"
    )
    ctx = GradeContext(ledger=ledger, config=cfg, arm="probe")
    for ref in refs:
        item = {"item_id": f"probe-{ref}", "family": "convention_conformance",
                "gold": {"checklist_ref": ref}, "question": "draft"}
        d = grade_item(item, degenerate, ctx).score
        c = grade_item(item, competent, ctx).score
        assert d < 1.0, f"checklist {ref} scores a non-answer at {d:.3f}"
        assert c > d, f"checklist {ref} does not separate a draft ({c:.3f}) from a non-answer ({d:.3f})"
