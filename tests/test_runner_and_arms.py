"""Runner durability and resume, and the guarantee that no arm phones home."""

import io
import json
import os

import pytest

from harness.arms import Arm, ArmNotWired, FixtureArm, NullArm, build_arm, load_arms
from harness.runner import Runner, load_items, run_key

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS_YAML = os.path.join(REPO, "config", "arms.yaml")


# ------------------------------------------------------------------- arms
def test_arms_load_from_config():
    arms = load_arms(ARMS_YAML)
    assert "sediment" in arms and "claude" in arms
    assert "fixture_strong" in arms and "fixture_weak" in arms


def test_both_real_arms_are_disabled_and_unrunnable():
    """The zero-model-call guarantee, asserted rather than promised."""
    arms = load_arms(ARMS_YAML)
    for name in ("sediment", "claude", "sediment_with_cartridge", "sediment_no_search"):
        arm = arms[name]
        assert arm.enabled is False, f"{name} is enabled in committed config"
        assert arm.runnable() is False, f"{name} reports itself runnable"


def test_disabled_network_arm_refuses_to_generate():
    arms = load_arms(ARMS_YAML)
    with pytest.raises(ArmNotWired) as exc:
        arms["claude"].generate("what is 2+2", {"item_id": "x"})
    assert "BENCHMARK_ALLOW_NETWORK" in str(exc.value)


def test_enabling_config_alone_is_not_enough(monkeypatch):
    """Both switches must be thrown; config alone does not open the door."""
    arm = build_arm("probe", {"type": "anthropic_api", "enabled": True})
    monkeypatch.delenv("BENCHMARK_ALLOW_NETWORK", raising=False)
    assert arm.runnable() is False
    with pytest.raises(ArmNotWired):
        arm.generate("q", {"item_id": "x"})


def test_even_with_both_switches_there_is_no_transport(monkeypatch):
    arm = build_arm("probe", {"type": "anthropic_api", "enabled": True})
    monkeypatch.setenv("BENCHMARK_ALLOW_NETWORK", "1")
    assert arm.runnable() is True
    with pytest.raises(NotImplementedError):
        arm.generate("q", {"item_id": "x"})


def test_cartridge_is_config_not_code():
    """Adding a cartridge arm must not require touching Python."""
    plain = build_arm("a", {"type": "sediment", "cartridge": None})
    loaded = build_arm("b", {"type": "sediment", "cartridge": "westbrook-v1"})
    assert plain.cartridge is None
    assert loaded.cartridge == "westbrook-v1"
    assert type(plain) is type(loaded)


def test_unknown_arm_type_raises():
    with pytest.raises(ValueError):
        build_arm("x", {"type": "telepathy"})


def test_requesting_a_missing_arm_raises():
    with pytest.raises(ValueError):
        load_arms(ARMS_YAML, ["no_such_arm"])


def test_null_arm_returns_empty():
    arm = build_arm("n", {"type": "null"})
    assert isinstance(arm, NullArm)
    assert arm.generate("q", {"item_id": "x"}) == ""


def test_fixture_arm_reads_inline_responses():
    arm = build_arm("fixture_strong", {"type": "fixture"})
    assert isinstance(arm, FixtureArm)
    item = {"item_id": "i1", "responses": {"fixture_strong": "hello"}}
    assert arm.generate("q", item) == "hello"


def test_fixture_arm_missing_response_fails_loud():
    arm = build_arm("fixture_strong", {"type": "fixture"})
    with pytest.raises(KeyError):
        arm.generate("q", {"item_id": "i1", "responses": {"other": "x"}})


# ----------------------------------------------------------------- runner
def _mini_items():
    return [
        {"item_id": "t1", "family": "single_fact_recall", "question": "q1",
         "answer_type": "numeric", "gold": {"value": 6.4, "unit": "multiple"},
         "responses": {"a": "ANSWER: 6.4x"}},
        {"item_id": "t2", "family": "single_fact_recall", "question": "q2",
         "answer_type": "numeric", "gold": {"value": 7.0, "unit": "multiple"},
         "responses": {"a": "ANSWER: 9.9x"}},
    ]


def _runner(tmp_path, ledger, grading_config, items=None):
    arms = {"a": build_arm("a", {"type": "fixture"})}
    return Runner(items=items or _mini_items(), arms=arms, out_dir=str(tmp_path),
                  ledger=ledger, config=grading_config, repo_root=REPO)


def test_runner_writes_incrementally(tmp_path, ledger, grading_config):
    r = _runner(tmp_path, ledger, grading_config)
    counts = r.run()
    assert counts["graded"] == 2 and counts["errors"] == 0
    assert os.path.exists(os.path.join(str(tmp_path), "grades.jsonl"))
    assert os.path.exists(os.path.join(str(tmp_path), "run.json"))
    # one file per item as well as the jsonl
    files = os.listdir(os.path.join(str(tmp_path), "items"))
    assert len(files) == 2


def test_runner_resumes_and_does_not_regrade(tmp_path, ledger, grading_config):
    r1 = _runner(tmp_path, ledger, grading_config)
    assert r1.run()["graded"] == 2
    r2 = _runner(tmp_path, ledger, grading_config)
    counts = r2.run(resume=True)
    assert counts["graded"] == 0
    assert counts["skipped"] == 2


def test_runner_resumes_from_a_partial_file(tmp_path, ledger, grading_config):
    """The half-a-run case: one item on disk, the other still to do."""
    r1 = _runner(tmp_path, ledger, grading_config, items=_mini_items()[:1])
    r1.run()
    r2 = _runner(tmp_path, ledger, grading_config)
    counts = r2.run(resume=True)
    assert counts["skipped"] == 1
    assert counts["graded"] == 1


def test_runner_survives_a_torn_final_line(tmp_path, ledger, grading_config):
    """A run killed mid-write leaves a partial line; everything before it stands."""
    r1 = _runner(tmp_path, ledger, grading_config)
    r1.run()
    gp = os.path.join(str(tmp_path), "grades.jsonl")
    with io.open(gp, "a", encoding="utf-8") as fh:
        fh.write('{"key": "t9::a", "item_id": "t9"')  # torn
    r2 = _runner(tmp_path, ledger, grading_config)
    done = r2.completed_keys()
    assert run_key("t1", "a") in done and run_key("t2", "a") in done


def test_no_resume_regrades_everything(tmp_path, ledger, grading_config):
    r1 = _runner(tmp_path, ledger, grading_config)
    r1.run()
    r2 = _runner(tmp_path, ledger, grading_config)
    assert r2.run(resume=False)["graded"] == 2


def test_a_broken_item_does_not_stop_the_run(tmp_path, ledger, grading_config):
    items = _mini_items() + [
        {"item_id": "bad", "family": "single_fact_recall", "question": "q",
         "answer_type": "numeric", "responses": {"a": "ANSWER: 1"}},  # no gold block
    ]
    r = _runner(tmp_path, ledger, grading_config, items=items)
    counts = r.run()
    assert counts["graded"] == 3
    assert counts["errors"] == 1
    recs = [json.loads(l) for l in io.open(
        os.path.join(str(tmp_path), "grades.jsonl"), encoding="utf-8") if l.strip()]
    bad = next(r for r in recs if r["item_id"] == "bad")
    assert bad["verdict"] == "error"
    assert "no `gold` block" in bad["error"]
    assert "Traceback" in bad["error"], "the traceback must be preserved, not swallowed"


def test_run_json_records_the_zero_model_call_status(tmp_path, ledger, grading_config):
    r = _runner(tmp_path, ledger, grading_config)
    r.run()
    with io.open(os.path.join(str(tmp_path), "run.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["zero_model_calls"] is True
    assert meta["n_items"] == 2
    assert "fabrication_allowlist" in meta
    assert meta["ledger"]["schema"] == "normalized"


def test_load_items_rejects_duplicates(tmp_path):
    p = tmp_path / "dupe.yaml"
    p.write_text("items:\n  - item_id: a\n    family: f\n  - item_id: a\n    family: f\n",
                 encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_items(str(p))
    assert "duplicate" in str(exc.value)


def test_load_items_rejects_missing_family(tmp_path):
    p = tmp_path / "nofam.yaml"
    p.write_text("items:\n  - item_id: a\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_items(str(p))
    assert "family" in str(exc.value)
