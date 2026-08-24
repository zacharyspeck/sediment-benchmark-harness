"""CLI surfaces: item validation, ledger resolution, allowlist probe."""

from __future__ import annotations

import os
import types

import pytest

from harness import cli

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_ITEMS = os.path.join(REPO, "fixtures", "fixture_items.yaml")
EDGE_ITEMS = os.path.join(REPO, "fixtures", "edge_cases.yaml")
MINI = os.path.join(REPO, "fixtures", "mini_ledger.json")


def _args(**kw):
    return types.SimpleNamespace(**kw)


# ------------------------------------------------------------ validate-items
def test_validate_items_accepts_the_fixture_set(capsys):
    rc = cli.cmd_validate_items(_args(items=FIXTURE_ITEMS, strict=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "0 error(s)" in out
    assert out.strip().endswith("OK")


def test_validate_items_accepts_the_edge_set(capsys):
    rc = cli.cmd_validate_items(_args(items=EDGE_ITEMS, strict=True))
    assert rc == 0
    assert "error(s)" in capsys.readouterr().out


def test_validate_items_reports_power_per_family(capsys):
    cli.cmd_validate_items(_args(items=FIXTURE_ITEMS, strict=False))
    out = capsys.readouterr().out
    assert "95% CI half-width at this n" in out
    assert "below min_items_for_claim" in out, "an underpowered family must be called out"
    assert "pts at 50%" in out


def test_validate_items_reports_the_rule_application_bucket_split(capsys):
    cli.cmd_validate_items(_args(items=FIXTURE_ITEMS, strict=False))
    out = capsys.readouterr().out
    assert "implicit_rule_application buckets" in out
    assert "discriminating:" in out
    assert "oversampling that bucket" in out


def test_validate_items_catches_a_missing_gold(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "items:\n"
        "  - item_id: x1\n"
        "    family: single_fact_recall\n"
        "    question: q\n"
        "    answer_type: numeric\n",
        encoding="utf-8")
    rc = cli.cmd_validate_items(_args(items=str(p), strict=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "missing or malformed `gold`" in out


def test_validate_items_catches_a_bad_unit(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "items:\n"
        "  - item_id: x1\n"
        "    family: single_fact_recall\n"
        "    question: q\n"
        "    answer_type: numeric\n"
        "    gold: {value: 1.0, unit: furlongs}\n",
        encoding="utf-8")
    rc = cli.cmd_validate_items(_args(items=str(p), strict=True))
    assert rc == 1
    assert "furlongs" in capsys.readouterr().out


def test_validate_items_catches_an_unknown_checklist(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "items:\n"
        "  - item_id: x1\n"
        "    family: convention_conformance\n"
        "    question: draft it\n"
        "    answer_type: document\n"
        "    gold: {checklist_ref: no_such_checklist_v9}\n",
        encoding="utf-8")
    rc = cli.cmd_validate_items(_args(items=str(p), strict=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "no_such_checklist_v9" in out


def test_validate_items_warns_when_family4_meta_is_missing(tmp_path, capsys):
    p = tmp_path / "thin.yaml"
    p.write_text(
        "items:\n"
        "  - item_id: x1\n"
        "    family: implicit_rule_application\n"
        "    question: advance or pass?\n"
        "    answer_type: categorical\n"
        "    gold: {value: pass, unit: none}\n",
        encoding="utf-8")
    rc = cli.cmd_validate_items(_args(items=str(p), strict=True))
    out = capsys.readouterr().out
    assert rc == 0, "missing meta is a warning, not an error"
    assert "discriminating/clean breakout" in out


def test_validate_items_rejects_an_unknown_family(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "items:\n"
        "  - item_id: x1\n"
        "    family: astrology\n"
        "    question: q\n"
        "    gold: {value: 1}\n",
        encoding="utf-8")
    rc = cli.cmd_validate_items(_args(items=str(p), strict=True))
    assert rc == 1
    assert "unknown family" in capsys.readouterr().out


# ------------------------------------------------------------ ledger + misc
def test_resolve_ledger_honours_an_explicit_path(capsys):
    led = cli.resolve_ledger(MINI, quiet=True)
    assert led.schema == "normalized"
    assert len(led.deals) == 16


def test_grading_config_carries_the_checklists():
    cfg = cli.load_grading_config()
    assert cfg["checklists"], "checklists must be loaded from items/"
    assert "halloran_exec_summary_v1" in cfg["checklists"]


def test_allowlist_command_reports_a_populated_allowlist(capsys):
    rc = cli.cmd_allowlist(_args(ledger=MINI, probe=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert '"entities"' in out


def test_allowlist_probe_flags_an_invention(capsys):
    rc = cli.cmd_allowlist(_args(ledger=MINI, probe="We looked at Project Winnwold."))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Winnwold" in out


def test_main_dispatches(capsys):
    rc = cli.main(["validate-items", "--items", FIXTURE_ITEMS])
    assert rc == 0
    assert "items file" in capsys.readouterr().out
