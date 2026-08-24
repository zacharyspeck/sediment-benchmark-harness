"""The regenerated holdout set, and the policy applied to it.

The generator itself lives in the corpus and is imported read-only; these tests
cover the parts THIS repo owns -- the rule cap, the collision screen, and the
invariants the output file has to satisfy.
"""

from __future__ import annotations

import io
import json
import os

import pytest

from harness.holdout import apply_rule_cap, screen_target

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(REPO, "data", "holdout-targets-v2.json")


# ----------------------------------------------------------------- rule cap
def _units(counts):
    """Synthetic unit keys, `n` of each rule."""
    out = {}
    for rule, n in counts.items():
        for i in range(n):
            out[f"{rule}[lever={i}]"] = {"rule": rule}
    return out


def test_cap_is_respected_when_it_is_achievable():
    sel = apply_rule_cap(_units({"A": 14, "B": 9, "C": 6}), 0.35)
    assert sel["impossible"] is False
    n = len(sel["selected"])
    assert n > 0
    for rule, k in sel["per_rule"].items():
        assert k <= int(0.35 * n) + 1, f"{rule} takes {k} of {n}"


def test_cap_maximises_rather_than_stopping_at_the_scarcest_rule():
    """With 14/9/6 available the answer is 20, not 3 x 6."""
    sel = apply_rule_cap(_units({"A": 14, "B": 9, "C": 6}), 0.35)
    assert len(sel["selected"]) == 20


def test_an_impossible_cap_falls_back_to_the_smallest_achievable_one():
    """Two rules cannot both stay under 35% -- 2 x 0.35 < 1.

    The pre-registration says to use the smallest achievable cap and print it,
    not to silently drop the rule or silently exceed the cap.
    """
    sel = apply_rule_cap(_units({"A": 10, "B": 10}), 0.35)
    assert sel["impossible"] is True
    assert sel["cap_used"] == pytest.approx(0.5)
    assert sel["reason"]
    assert len(sel["selected"]) > 0


def test_cap_with_a_single_rule_is_impossible_and_says_so():
    sel = apply_rule_cap(_units({"A": 10}), 0.35)
    assert sel["impossible"] is True
    assert sel["cap_used"] == pytest.approx(1.0)


def test_cap_on_an_empty_pool_does_not_crash():
    sel = apply_rule_cap({}, 0.35)
    assert sel["selected"] == []
    assert sel["impossible"] is True


# --------------------------------------------------------------- collisions
class _FakeDoc:
    def __init__(self, rel, text):
        self.rel, self.text = rel, text


class _FakeIndex:
    def __init__(self, docs):
        self.docs = [_FakeDoc(r, t) for r, t in docs]

    def contains_anywhere(self, needle):
        return any(needle.lower() in d.text.lower() for d in self.docs)


def test_screen_rejects_a_target_whose_figures_co_occur_in_one_document():
    ix = _FakeIndex([("a.md", "Revenue of $28.1M and adjusted EBITDA of $6.0M.")])
    hits = screen_target({"revenue_m": 28.1, "adj_ebitda_m": 6.0, "id": "h1"}, ix)
    assert hits and "co-occur" in hits[0]


def test_screen_accepts_figures_that_appear_only_in_separate_documents():
    """A lone $28.1M is unremarkable in 10k financial documents.

    Rejecting on a single figure would reject nearly every candidate and shrink
    the set for no reason -- the pair in ONE document is what identifies a
    company.
    """
    ix = _FakeIndex([("a.md", "Revenue of $28.1M."), ("b.md", "EBITDA of $6.0M.")])
    assert screen_target({"revenue_m": 28.1, "adj_ebitda_m": 6.0, "id": "h1"}, ix) == []


def test_screen_rejects_a_target_whose_id_appears_anywhere():
    ix = _FakeIndex([("a.md", "see holdout-v2-disc-001 for detail")])
    hits = screen_target({"id": "holdout-v2-disc-001"}, ix)
    assert hits and "appears in the corpus" in hits[0]


# ------------------------------------------------------------- the artifact
@pytest.mark.skipif(not os.path.exists(V2), reason="holdout-targets-v2.json not generated")
class TestGeneratedFile:
    @pytest.fixture(scope="class")
    def payload(cls):
        return json.load(io.open(V2, encoding="utf-8"))


    def test_it_records_its_own_provenance(self, payload):
        gen = payload["generated_against"]
        assert len(gen["generator_sha256"]) == 64
        assert gen["provenance"]["ledger"]["shards"]["deals"]["sha256"]
        assert gen["provenance"]["corpus"]["documents"] > 0

    def test_every_target_carries_per_target_provenance(self, payload):
        for t in payload["holdout_targets"]:
            p = t["provenance"]
            assert p["generator"] == "v4_holdout.build"
            assert p["imported_read_only"] is True
            assert len(p["generator_sha256"]) == 64

    def test_target_ids_are_unique(self, payload):
        ids = [t["id"] for t in payload["holdout_targets"]]
        assert len(ids) == len(set(ids))

    def test_no_unit_key_is_used_twice(self, payload):
        keys = [t["unit_key"] for t in payload["holdout_targets"]]
        assert len(keys) == len(set(keys)), "a lever bin is represented twice"

    def test_the_saturation_actually_converged(self, payload):
        assert payload["saturation"]["saturated"] is True
        trace = payload["saturation"]["rounds"]
        assert trace[-1]["gained"] == 0, "the last round still found new units"

    def test_controls_are_thirty_percent_of_scored(self, payload):
        assert payload["n_control"] == max(1, round(0.30 * payload["n_scored"]))

    def test_no_rule_exceeds_the_cap_that_was_used(self, payload):
        n = payload["n_scored"]
        cap = payload["policy"]["rule_cap_used"]
        for rule, k in payload["rule_distribution"].items():
            assert k <= int(cap * n) + 1, f"{rule} is {k} of {n}, above the {cap} cap"

    def test_the_gold_distribution_is_recorded_even_when_degenerate(self, payload):
        """A single-valued gold is a finding, not an error -- but it must be visible."""
        assert payload["gold_distribution"]
        assert sum(payload["gold_distribution"].values()) == \
            payload["saturation"]["ceiling_distinct_units"]

    def test_nothing_was_written_into_the_corpus(self):
        """The output lives in this repo. The corpus path must be unwritable."""
        from harness.paths import is_writable

        corpus = os.path.join(os.path.dirname(REPO), "sediment-corpus")
        assert is_writable(os.path.join(corpus, "holdout-targets-v2.json")) is False
        assert is_writable(V2) is True
