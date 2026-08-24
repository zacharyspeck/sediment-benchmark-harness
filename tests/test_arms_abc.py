"""Arms A and C must differ at exactly one configuration path.

The benchmark's whole claim is A minus C. If anything other than the adapter
diverges between them -- a temperature, a max_tokens, one arm quietly losing its
search tool -- then the difference stops being attributable to studied knowledge
and no amount of care in the analysis recovers it.

These tests assert STRUCTURE, not a particular adapter value. `lora_adapter` is
null on both today because no adapter exists yet; a test that only checked "C's
adapter is null" would pass vacuously now and keep passing after A gets a real
one. What is asserted is that every other path is equal, which stays meaningful.
"""

from __future__ import annotations

import os

import pytest

from harness.arms import build_arm, load_arms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS_YAML = os.path.join(REPO, "config", "arms.yaml")

ADAPTER_PATH = ("params", "lora_adapter")
#: Fields that describe the arm to a reader rather than configuring it.
PROSE_FIELDS = {"notes"}


def _flatten(d, prefix=()):
    out = {}
    for k, v in (d or {}).items():
        path = prefix + (k,)
        if isinstance(v, dict):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


@pytest.fixture(scope="module")
def arms():
    return load_arms(ARMS_YAML)


def test_all_three_arms_exist(arms):
    for name in ("sediment", "claude", "base"):
        assert name in arms, f"arm {name} is missing from config/arms.yaml"


def test_a_and_c_are_structurally_identical_except_the_adapter(arms):
    a = arms["sediment"].describe()
    c = arms["base"].describe()
    fa = {k: v for k, v in _flatten(a).items() if k[0] not in PROSE_FIELDS and k[0] != "name"}
    fc = {k: v for k, v in _flatten(c).items() if k[0] not in PROSE_FIELDS and k[0] != "name"}

    assert set(fa) == set(fc), (
        "arm A and arm C declare different configuration keys: "
        f"only in A {sorted(set(fa) - set(fc))}, only in C {sorted(set(fc) - set(fa))}"
    )

    differing = sorted(k for k in fa if fa[k] != fc[k])
    assert differing in ([], [ADAPTER_PATH]), (
        "arm A and arm C differ at more than the adapter path. "
        f"Differing paths: {differing}. A minus C is only attributable to studied "
        "knowledge when everything else is held fixed."
    )


def test_arm_c_adapter_is_null(arms):
    assert arms["base"].params.get("lora_adapter") is None


def test_arm_c_declares_the_adapter_key_rather_than_omitting_it(arms):
    """Absent is not the same as null.

    An omitted key would make the A/C key-set comparison fail for the wrong
    reason, and would leave "does C have an adapter?" unanswerable from the
    recorded config.
    """
    assert "lora_adapter" in (arms["base"].params or {})


def test_a_and_c_share_the_model_and_the_search_tool(arms):
    a, c = arms["sediment"], arms["base"]
    assert a.model == c.model
    assert list(a.tools) == list(c.tools) == ["corpus_search"]
    assert a.params.get("corpus_dir") == c.params.get("corpus_dir")


def test_every_real_arm_is_disabled_and_unrunnable(arms):
    """Nothing that could make a network call may be runnable from committed config."""
    for name in ("sediment", "claude", "base", "sediment_with_cartridge", "sediment_no_search"):
        arm = arms[name]
        assert arm.enabled is False, f"{name} is enabled in committed config"
        assert arm.runnable() is False, f"{name} reports runnable"


def test_arm_b_is_a_subprocess_transport_with_a_corpus_directory(arms):
    """Arm B hands over a DIRECTORY, which an HTTP-shaped arm has nowhere to put."""
    b = arms["claude"]
    assert b.type == "local_cli_subprocess"
    assert b.params.get("corpus_dir"), "arm B has no corpus directory to search"
    assert "corpus_search" in b.tools


def test_every_searching_arm_points_at_the_same_corpus(arms):
    """Every arm searches the SAME corpus -- that is the premise of the design."""
    dirs = {
        n: a.params.get("corpus_dir")
        for n, a in arms.items()
        if "corpus_search" in (a.tools or [])
    }
    assert dirs, "no arm declares corpus_search"
    distinct = {d for d in dirs.values() if d}
    assert len(distinct) == 1, f"searching arms point at different corpora: {dirs}"


# ------------------------------------------------------- constant baseline
def test_constant_arm_answers_identically_regardless_of_the_question(arms):
    c = arms["constant_pass"]
    assert c.runnable() is True
    first = c.generate("Does Westbrook advance this or pass?", {"item_id": "x"})
    second = c.generate("What was the entry multiple on Project Ashrstead?", {"item_id": "y"})
    assert first == second
    assert first.startswith("ANSWER: pass")


def test_constant_arm_is_not_a_model_call(arms):
    """It must run under the zero-network constraint, unlike every real arm."""
    assert arms["constant_pass"].type == "constant"
    assert arms["constant_pass"].model is None
    assert os.environ.get("BENCHMARK_ALLOW_NETWORK") != "1"
    assert arms["constant_pass"].runnable() is True


def test_a_constant_arm_can_be_pointed_at_any_answer():
    arm = build_arm("constant_advance", {"type": "constant", "enabled": True,
                                         "params": {"answer": "advance"}})
    assert arm.generate("anything", {}).startswith("ANSWER: advance")


def test_null_arm_and_constant_arm_are_different_failures(arms):
    """null scores 0, constant can score 100. Neither substitutes for the other."""
    assert arms["null_arm"].generate("q", {}) == ""
    assert arms["constant_pass"].generate("q", {}) != ""
