"""The ledger adapter and the declarative query layer."""

import os

import pytest

from harness.ledger import Ledger, LedgerUnavailable, load_ledger
from harness.query import MISSING, QueryError, aggregate, get_path, select

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ ledger
def test_mini_ledger_loads_normalized(ledger):
    assert ledger.schema == "normalized"
    assert len(ledger.deals) == 16
    assert len(ledger.holdout_targets) == 4
    assert ledger.firms["westbrook"]["name"] == "Westbrook Capital Partners"


def test_summary_is_serialisable(ledger):
    s = ledger.summary()
    assert s["counts"]["deals"] == 16
    assert "warnings" in s


def test_derived_hold_period(ledger):
    kiln = ledger.deal_by_codename("Kilnmouth")
    assert kiln["_derived"]["hold_period_years"] == pytest.approx(6.319, abs=0.01)
    assert kiln["_derived"]["has_exit"] is True
    stalcrest = ledger.deal_by_codename("Stalcrest")
    assert stalcrest["_derived"]["has_exit"] is False
    assert "hold_period_years" not in stalcrest["_derived"]


def test_derived_holdout_discrimination(ledger):
    t = ledger.holdout_by_id("holdout-mini-000")
    assert t["_derived"]["computed_discriminates"] is True
    assert t["_derived"]["flag_mismatch"] is False
    clean = ledger.holdout_by_id("holdout-mini-002")
    assert clean["_derived"]["computed_discriminates"] is False


def test_discriminates_flag_disagreement_becomes_a_warning(tmp_path):
    """A ledger whose flag contradicts its own fields must warn, not pass quietly."""
    import json

    src = os.path.join(REPO, "fixtures", "mini_ledger.json")
    with open(src, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["holdout_targets"][0]["discriminates"] = False  # now contradicts the verdicts
    p = tmp_path / "bad_ledger.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    led = load_ledger(str(p), "normalized")
    assert any("discriminates" in w for w in led.warnings)


def test_missing_ledger_raises_with_a_useful_message():
    with pytest.raises(LedgerUnavailable) as exc:
        load_ledger(os.path.join(REPO, "no", "such", "ledger.json"))
    assert "mini_ledger" in str(exc.value)


def test_unknown_collection_raises(ledger):
    with pytest.raises(KeyError):
        ledger.collection("sandwiches")


def test_frozen_facts_accessors(ledger):
    assert ledger.written_screens()["C1"].startswith("no customer above 30%")
    assert ledger.mandate()["min_adj_ebitda_m"] == 4.0


def test_implicit_rule_lookup(ledger):
    r = ledger.implicit_rule("RULE-F2")
    assert r and "retention" in r["statement"].lower()
    assert ledger.implicit_rule("IR-NOPE") is None


# ------------------------------------------------------------------- query
def test_get_path_and_missing(ledger):
    d = ledger.deal_by_codename("Kilnmouth")
    assert get_path(d, "numbers.entry_multiple") == 6.4
    assert get_path(d, "numbers.nope") is MISSING
    assert get_path(d, "exit.multiple") == 9.2


def test_shorthand_equality_is_case_insensitive(ledger):
    upper = select(ledger.deals, {"firm": "WESTBROOK"})
    lower = select(ledger.deals, {"firm": "westbrook"})
    assert len(upper) == len(lower) == 13


def test_operators(ledger):
    deals = ledger.deals
    assert len(select(deals, {"outcome": "pass"})) == 8
    assert len(select(deals, {"year": {"op": "gte", "value": 2024}})) == 3
    assert len(select(deals, {"year": {"op": "between", "lo": 2023, "hi": 2023}})) == 6
    assert len(select(deals, {"kill_code": {"op": "in", "values": ["C1", "C2"]}})) == 6
    assert len(select(deals, {"codename": {"op": "regex", "value": "^C"}})) == 3
    assert len(select(deals, {"sector": {"op": "contains", "value": "health"}})) == 3
    assert len(select(deals, {"firm": {"op": "ne", "value": "westbrook"}})) == 3


def test_is_null_covers_absent_and_explicit_null(ledger):
    """Thin records omit the key; thick ones carry an explicit null."""
    absent_or_null = select(ledger.deals, {"numbers.entry_multiple": {"op": "is_null"}})
    assert len(absent_or_null) == 12
    explicit = select(ledger.deals, {"exit": {"op": "is_null"}})
    assert any(d["codename"] == "Stalcrest" for d in explicit)


def test_not_null_excludes_absent(ledger):
    rows = select(ledger.deals, {"numbers.entry_multiple": {"op": "not_null"}})
    assert len(rows) == 4


def test_truthy_on_derived(ledger):
    rows = select(ledger.deals, {"_derived.has_exit": {"op": "truthy", "value": True}})
    assert {d["codename"] for d in rows} == {"Kilnmouth", "Juniper", "Foxgworth"}


def test_list_contains(ledger):
    rows = select(ledger.holdout_targets,
                  {"driving_rule_ids": {"op": "list_contains", "value": "RULE-F2"}})
    assert len(rows) == 1


def test_unknown_operator_raises(ledger):
    with pytest.raises(QueryError):
        select(ledger.deals, {"year": {"op": "approximately", "value": 2023}})


# --------------------------------------------------------------- aggregate
def test_aggregations(ledger):
    hc23 = select(ledger.deals,
                  {"sector": "healthcare services", "year": 2023, "stage": "screen"})
    assert aggregate(hc23, {"op": "count"}).value == 3
    assert aggregate(hc23, {"op": "mean", "field": "numbers.adj_ebitda_margin_pct",
                            "round": 1}).value == 20.3
    assert aggregate(hc23, {"op": "sum", "field": "numbers.adj_ebitda_m",
                            "round": 2}).value == 17.5
    assert aggregate(hc23, {"op": "max", "field": "numbers.adj_ebitda_m"}).value == 7.6
    assert aggregate(hc23, {"op": "min", "field": "numbers.adj_ebitda_m"}).value == 3.8


def test_mode_and_group_counts(ledger):
    passes = select(ledger.deals, {"outcome": "pass", "kill_code": {"op": "not_null"}})
    assert aggregate(passes, {"op": "mode", "field": "kill_code"}).value == "C1"
    gc = aggregate(passes, {"op": "group_counts", "field": "kill_code"}).value
    assert gc["C1"] == 4 and gc["C2"] == 2


def test_aggregate_records_contributors(ledger):
    hc23 = select(ledger.deals, {"sector": "healthcare services", "year": 2023})
    res = aggregate(hc23, {"op": "mean", "field": "numbers.adj_ebitda_margin_pct"})
    # traces record the canonical ledger id, not the display codename
    assert set(res.contributing_ids) == {"wb-marlin", "wb-quarborne", "wb-pelican"}
    assert len(res.contributing_values) == 3
    assert "contributing_ids" in res.as_trace()


def test_aggregate_on_empty_rows_is_not_a_crash(ledger):
    res = aggregate([], {"op": "mean", "field": "numbers.revenue_m"})
    assert res.value is None and res.n == 0 and res.note


def test_value_op_requires_a_unique_row(ledger):
    rows = select(ledger.deals, {"codename": "Kilnmouth"})
    assert aggregate(rows, {"op": "value", "field": "numbers.entry_multiple"}).value == 6.4
    many = select(ledger.deals, {"firm": "westbrook"})
    res = aggregate(many, {"op": "value", "field": "numbers.entry_multiple"})
    assert res.value is None and "exactly 1" in res.note


def test_unknown_aggregate_op_raises(ledger):
    with pytest.raises(QueryError):
        aggregate(ledger.deals, {"op": "vibe", "field": "numbers.revenue_m"})
