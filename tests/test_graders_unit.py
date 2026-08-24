"""Grader edge cases beyond the fixture matrix."""

import pytest

from graders.base import GradeContext, grade_item
from graders.numeric import (
    compare_categorical,
    compare_list,
    compare_numeric,
    parse_quantity,
)


@pytest.fixture()
def ctx(ledger, grading_config):
    return GradeContext(ledger=ledger, config=grading_config, arm="test")


# ------------------------------------------------------- quantity parsing
@pytest.mark.parametrize("text,unit,expected", [
    ("6.4x", "multiple", 6.4),
    ("6.4", "multiple", 6.4),
    ("$28.4M", "money_m", 28.4),
    ("28.4", "money_m", 28.4),
    ("$28.4 million", "money_m", 28.4),
    ("28,400,000", "money_m", 28.4),
    ("$1.2bn", "money_m", 1200.0),
    ("22%", "percent", 22.0),
    ("22", "percent", 22.0),
    ("0.22", "percent", 22.0),
    ("5.5 years", "years", 5.5),
    ("66 months", "years", 5.5),
    ("86", "count", 86.0),
    ("approximately 3", "count", 3.0),
    ("-2.5", "multiple", -2.5),
])
def test_parse_quantity(text, unit, expected):
    q = parse_quantity(text, unit)
    assert q.value == pytest.approx(expected, rel=1e-6)


def test_parse_quantity_flags_a_rescaled_fraction():
    q = parse_quantity("0.415", "percent")
    assert q.value == pytest.approx(41.5)
    assert "scale_normalized" in q.flags


def test_parse_quantity_flags_whole_dollars():
    q = parse_quantity("$28,400,000", "money_m")
    assert q.value == pytest.approx(28.4)
    assert "scale_normalized" in q.flags


def test_parse_quantity_on_nonsense():
    q = parse_quantity("no idea", "multiple")
    assert q.value is None and q.notes


def test_million_is_not_shadowed_by_m():
    """Alternation order bug guard: a bare `m` must not eat the 'm' of million."""
    assert parse_quantity("12 million", "money_m").value == pytest.approx(12.0)
    assert parse_quantity("12m", "money_m").value == pytest.approx(12.0)


# ------------------------------------------------------------- comparison
def test_compare_numeric_respects_absolute_and_relative():
    tol = {"abs": 0.05, "rel": 0.0}
    assert compare_numeric(parse_quantity("6.44", "multiple"), 6.4, tol).passed
    assert not compare_numeric(parse_quantity("6.5", "multiple"), 6.4, tol).passed
    rel = {"abs": 0.0, "rel": 0.02}
    assert compare_numeric(parse_quantity("102", "count"), 100, rel).passed


def test_compare_numeric_statement_is_human_readable():
    c = compare_numeric(parse_quantity("6.5", "multiple"), 6.4, {"abs": 0.05, "rel": 0.0})
    assert c.statement.startswith("FAIL")
    assert "6.4" in c.statement


def test_tolerance_boundary_is_not_decided_by_float_noise():
    """7.15 - 7.1 is 0.05000000000000071 in binary float."""
    tol = {"abs": 0.05, "rel": 0.0}
    assert compare_numeric(parse_quantity("7.15", "money_m"), 7.1, tol).passed
    assert not compare_numeric(parse_quantity("7.16", "money_m"), 7.1, tol).passed


def test_compare_numeric_handles_a_zero_gold():
    c = compare_numeric(parse_quantity("0", "count"), 0, {"abs": 0.0, "rel": 0.0})
    assert c.passed


def test_compare_categorical_accepts_synonyms():
    accept = {"C1": ["customer concentration", "criterion one"]}
    ok, stmt, flags = compare_categorical("customer concentration", "C1", accept)
    assert ok


def test_compare_categorical_rejects_a_competing_label():
    accept = {"C1": ["concentration"], "C2": ["margin"]}
    ok, stmt, _ = compare_categorical("margin", "C1", accept)
    assert not ok
    assert "competing label" in stmt


def test_compare_categorical_tolerates_trailing_prose():
    accept = {"pass": ["we pass"]}
    ok, _, flags = compare_categorical("Pass -- concentration", "pass", accept)
    assert ok
    assert "categorical_substring_match" in flags


def test_compare_list_is_strict_by_default():
    ok, score, stmt, detail = compare_list("C1, C2", ["C1", "C2"])
    assert ok and score == 1.0
    ok2, score2, _, detail2 = compare_list("C1, C2, C3", ["C1", "C2"])
    assert not ok2 and score2 == 0.0
    assert detail2["extra"] == ["c3"]
    assert detail2["f1"] < 1.0


def test_compare_list_non_strict_reports_f1():
    ok, score, _, _ = compare_list("C1, C3", ["C1", "C2"], strict=False)
    assert not ok
    assert 0 < score < 1


# ------------------------------------------------------------- graders
def _item(**kw):
    base = {"item_id": "x", "family": "single_fact_recall", "question": "q",
            "answer_type": "numeric", "gold": {"value": 6.4, "unit": "multiple"}}
    base.update(kw)
    return base


def test_missing_gold_block_is_an_error_not_a_zero(ctx):
    g = grade_item({"item_id": "x", "family": "single_fact_recall", "question": "q"},
                   "ANSWER: 6.4x", ctx)
    assert g.verdict == "error"
    assert g.error and "gold" in g.error


def test_unknown_family_is_an_error(ctx):
    g = grade_item({"item_id": "x", "family": "astrology", "question": "q"}, "ANSWER: 1", ctx)
    assert g.verdict == "error"


def test_empty_response_scores_zero_as_no_answer(ctx):
    g = grade_item(_item(), "", ctx)
    assert g.score == 0.0 and g.verdict == "no_answer"
    assert g.extraction["extraction_uncertain"] is True


def test_counts_are_graded_exactly_even_if_config_is_loose(ctx):
    """The family default tolerance must not leak into a count."""
    item = _item(family="corpus_wide_statistic",
                 gold={"value": 100, "unit": "count"}, answer_type="numeric")
    assert grade_item(item, "ANSWER: 100", ctx).score == 1.0
    assert grade_item(item, "ANSWER: 101", ctx).score == 0.0


def test_item_tolerance_overrides_the_count_rule(ctx):
    item = _item(family="corpus_wide_statistic",
                 gold={"value": 100, "unit": "count"},
                 grading={"tolerance": {"abs": 2, "rel": 0}})
    assert grade_item(item, "ANSWER: 101", ctx).score == 1.0


def test_conformance_inapplicable_checks_leave_the_denominator(ctx):
    item = {
        "item_id": "c", "family": "convention_conformance", "question": "draft it",
        "gold": {"checks": [
            {"id": "always", "kind": "must_contain", "patterns": ["alpha"]},
            {"id": "gated", "kind": "must_contain", "patterns": ["beta"],
             "applies_when": {"pattern": "never-appears-anywhere"}},
        ]},
    }
    g = grade_item(item, "ANSWER: x\n\nalpha", ctx)
    assert g.detail["checks_applicable"] == 1
    assert g.detail["checks_passed"] == 1
    assert g.score == 1.0, "a gated-off check must not count as a pass or a fail"
    assert g.binary is False, "conformance is fractional, not a Bernoulli trial"


def test_conformance_malformed_check_is_a_harness_flag_not_a_model_failure(ctx):
    item = {
        "item_id": "c", "family": "convention_conformance", "question": "draft it",
        "gold": {"checks": [
            {"id": "ok", "kind": "must_contain", "patterns": ["alpha"]},
            {"id": "broken", "kind": "must_contain"},  # no patterns
        ]},
    }
    g = grade_item(item, "ANSWER: x\n\nalpha", ctx)
    assert "checklist_error" in g.flags
    assert g.detail["checks_applicable"] == 1
    assert g.score == 1.0


def test_conformance_section_order_detects_reordering(ctx):
    checks = [{"id": "order", "kind": "section_order", "sections": [
        {"name": "first", "any_of": ["alpha"]},
        {"name": "second", "any_of": ["beta"]},
    ]}]
    item = {"item_id": "c", "family": "convention_conformance", "question": "q",
            "gold": {"checks": checks}}
    good = grade_item(item, "ANSWER: t\n\nalpha then beta", ctx)
    bad = grade_item(item, "ANSWER: t\n\nbeta then alpha", ctx)
    assert good.score == 1.0 and bad.score == 0.0
    ev = bad.detail["checks"][0]["evidence"]
    assert ev["observed_order"] == ["second", "first"]


def test_conformance_number_format(ctx):
    checks = [{"id": "money", "kind": "number_format", "target": "money", "decimals": 1}]
    item = {"item_id": "c", "family": "convention_conformance", "question": "q",
            "gold": {"checks": checks}}
    assert grade_item(item, "ANSWER: t\n\nRevenue of $27.4M.", ctx).score == 1.0
    bad = grade_item(item, "ANSWER: t\n\nRevenue of $27M.", ctx)
    assert bad.score == 0.0
    assert bad.detail["checks"][0]["evidence"]["nonconforming"]


def test_conformance_unknown_check_kind_is_flagged(ctx):
    item = {"item_id": "c", "family": "convention_conformance", "question": "q",
            "gold": {"checks": [{"id": "weird", "kind": "vibe_check"}]}}
    g = grade_item(item, "ANSWER: t\n\nanything", ctx)
    assert "checklist_error" in g.flags


def test_conformance_ledger_drift_is_flagged(ctx):
    """A checklist that no longer agrees with the ledger must say so."""
    item = {"item_id": "c", "family": "convention_conformance", "question": "q",
            "gold": {"checks": [{
                "id": "order", "kind": "must_contain", "patterns": ["alpha"],
                "ledger_check": {"path": "canon.frozen_facts.westbrook_page_one_order",
                                 "expect_values": ["wrong", "order", "here"]},
            }]}}
    g = grade_item(item, "ANSWER: t\n\nalpha", ctx)
    assert "checklist_ledger_drift" in g.flags
    assert g.detail["ledger_drift"]


def test_abstention_requires_both_verdict_and_no_invention(ctx):
    item = {"item_id": "a", "family": "absence_and_abstention", "question": "thesis?",
            "gold": {"value": "no_thesis"}}
    # right verdict, but invents a thesis underneath it
    g = grade_item(item, "ANSWER: no thesis\n\nOur thesis was route density.", ctx)
    assert g.score == 0.0
    assert g.detail["outcome"] == "invented_thesis"
    assert g.detail["sub_scores"]["verdict_correct"] is True
    assert g.detail["sub_scores"]["no_invented_thesis"] is False


def test_abstention_distinguishes_refusal_from_invention(ctx):
    item = {"item_id": "a", "family": "absence_and_abstention", "question": "thesis?",
            "gold": {"value": "no_thesis"}}
    refusal = grade_item(item, "I don't have that information.", ctx)
    assert refusal.detail["outcome"] == "bare_refusal"
    assert refusal.score == 0.0


def test_implicit_rule_records_the_written_criteria_answer(ctx):
    item = {"item_id": "i", "family": "implicit_rule_application", "question": "advance?",
            "gold": {"value": "pass"},
            "meta": {"written_criteria_verdict": "advance", "discriminates": True,
                     "driving_rule_ids": ["RULE-F2"]}}
    wrong = grade_item(item, "ANSWER: advance\n\nIt clears every screen.", ctx)
    assert wrong.score == 0.0
    assert wrong.detail["matched_written_criteria"] is True
    assert "answered_the_written_criteria" in wrong.flags
    right = grade_item(item, "ANSWER: pass\n\nThe retention floor decides it.", ctx)
    assert right.score == 1.0
    assert right.detail["discriminates"] is True


def test_implicit_rule_citation_is_diagnostic_only(ctx):
    """Right answer for no stated reason still scores."""
    item = {"item_id": "i", "family": "implicit_rule_application", "question": "advance?",
            "gold": {"value": "pass"},
            "meta": {"written_criteria_verdict": "advance", "discriminates": True,
                     "driving_rule_ids": ["RULE-F2"]}}
    g = grade_item(item, "ANSWER: pass\n\nJust a feeling.", ctx)
    assert g.score == 1.0
    assert g.detail["rule_citation"]["cited_any"] is False
