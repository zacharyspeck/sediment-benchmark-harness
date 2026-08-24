"""The v6 pool ingest and the balanced family-4 sampler.

These tests exist because every one of them corresponds to a way this run could
have produced a plausible-looking item set that measured nothing:

  * the advance class silently measuring zero, because the rescue rule is
    recorded in a field the unit keyer does not read;
  * two controls with opposite gold collapsing onto one unit key, because
    target ids repeat across generator batches;
  * the gold split drifting off 50/50, which is the whole point of the change;
  * an item carrying no chance floor, so a report can print a score without it;
  * the 35% rule cap being applied across the family again, where the advance
    class's single rule would make it unsatisfiable and drag the pass class
    down with it;
  * the pool file being swapped for a different one without anything noticing.
"""

from __future__ import annotations

import io
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from harness import holdout_pool as HP                              # noqa: E402

POOL = os.path.join(REPO, "data", "holdout-pool-v6.json")
SELECTED = os.path.join(REPO, "data", "holdout-targets-v3.json")
ITEMS = os.path.join(REPO, "data", "items-v2.json")

#: The pool this build measured. Pinned so a swapped input cannot pass silently:
#: every number in CAPACITY.md, PRE-REGISTRATION.md and build-report-v2.md was
#: computed against exactly this file.
POOL_SHA256 = "d5994cf27e0ce5ceafe43a5cb331f3ab952f71e8a3f26eccc297afdab01b01b9"


def _load(path):
    return json.load(io.open(path, encoding="utf-8"))


requires_pool = pytest.mark.skipif(
    not os.path.exists(POOL), reason="data/holdout-pool-v6.json not present")
requires_selection = pytest.mark.skipif(
    not os.path.exists(SELECTED), reason="data/holdout-targets-v3.json not present")
requires_items = pytest.mark.skipif(
    not os.path.exists(ITEMS), reason="data/items-v2.json not present")


# --------------------------------------------------------------- the pool
@requires_pool
def test_pool_file_is_the_one_every_number_was_computed_against():
    assert HP.sha256_file(POOL) == POOL_SHA256, (
        "data/holdout-pool-v6.json is not the file this build measured. Every "
        "capacity number and the pre-registration amendment were computed "
        "against sha256 " + POOL_SHA256)


#: Fact-block labels for the fields the advance-class rule reads. Which fields
#: those are is a property of the corpus, so the list is supplied beside the
#: private pool rather than committed here. Absent -> the assertion skips.
def _lever_labels():
    import json as _json
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "lever_labels.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return list(_json.load(fh))


LEVERS_THE_ADVANCE_RULE_READS = _lever_labels()


@requires_pool
def test_normalise_folds_the_rescue_into_the_driving_rules():
    """The advance-class rule records itself under `rescued_by_rule`, never
    under `driving_rule_ids`. Without folding it in, every advance target
    rejects as "no driving rule recorded" and the advance class measures
    zero."""
    targets = _load(POOL)["targets"]
    adv = [t for t in targets
           if t.get("bucket") == "discriminating"
           and t.get("ground_truth_outcome") == "advance"]
    assert adv, "the pool must carry a discriminating advance quadrant"
    assert all(not t.get("driving_rule_ids") for t in adv), (
        "precondition: the raw advance targets carry no driving rule")
    for t in adv[:50]:
        rescue = list(t.get("rescued_by_rule") or [])
        assert rescue, "precondition: an advance target records a rescue rule"
        n = HP.normalise(t)
        assert set(rescue) <= set(n["driving_rule_ids"])
        assert n["_rescue_rules"] == rescue


@requires_pool
def test_normalise_makes_target_ids_unique():
    """Ids repeat across generator batches. A control's entire unit key is its
    id, so duplicates make two controls with opposite gold into one unit."""
    targets = _load(POOL)["targets"]
    raw = [t.get("id") for t in targets]
    assert len(set(raw)) < len(raw), (
        "precondition: raw pool ids collide, which is what normalise fixes")
    fixed = [HP.normalise(t)["id"] for t in targets]
    assert len(set(fixed)) == len(fixed), "normalised ids must be unique"


def test_boundary_gaps_exclude_only_unevidenced_dates():
    """Strictly interior dates are dropped; the observed endpoints survive.

    Asserted against whatever boundaries are configured rather than against
    literals. The endpoint values are a property of the corpus and are supplied
    privately, so hard-coding them here would republish them -- which is the
    thing withholding them is for. With none configured there is nothing to
    check and the test skips.
    """
    if not HP.BOUNDARY_GAPS:
        pytest.skip("no boundaries configured (private side-file absent)")
    for name, lo, hi in HP.BOUNDARY_GAPS:
        # The endpoints are evidenced observations and must NOT be dropped.
        assert HP.boundary_gap(lo) is None, f"{name}: lower endpoint was dropped"
        assert HP.boundary_gap(hi) is None, f"{name}: upper endpoint was dropped"
        # Anything strictly inside is unevidenced and must be dropped.
        mid = _midpoint(lo, hi)
        if mid not in (lo, hi):
            assert HP.boundary_gap(mid) == name, f"{name}: interior date survived"


def _midpoint(lo: str, hi: str) -> str:
    from datetime import date, timedelta
    a = date.fromisoformat(lo)
    b = date.fromisoformat(hi)
    return (a + timedelta(days=(b - a).days // 2)).isoformat()


# ------------------------------------------------------------- the cap
def test_rule_cap_is_honoured_at_the_requested_n():
    keys = [f"IR-A[x={i}]" for i in range(40)] + \
           [f"IR-B[x={i}]" for i in range(40)] + \
           [f"IR-C[x={i}]" for i in range(40)]
    res = HP.select_under_cap(keys, want=60, cap=0.35)
    assert len(res["selected"]) == 60
    assert not res["impossible"]
    allowed = int(0.35 * 60)
    for rule, n in res["per_rule"].items():
        assert n <= allowed, f"{rule} took {n} of a permitted {allowed}"


def test_rule_cap_reports_impossible_rather_than_exceeding_it():
    """Two rules cannot both stay under 35% of the same set. The pre-registration
    says use the smallest achievable cap and PRINT it, never exceed it quietly."""
    keys = [f"IR-A[x={i}]" for i in range(30)] + [f"IR-B[x={i}]" for i in range(30)]
    res = HP.select_under_cap(keys, want=40, cap=0.35)
    assert res["impossible"] is True
    assert res["cap_used"] > 0.35
    assert res["reason"], "an impossible cap must say why"


def test_rule_cap_never_pads_beyond_available_capacity():
    keys = [f"IR-A[x={i}]" for i in range(5)]
    res = HP.select_under_cap(keys, want=50, cap=0.35)
    assert len(res["selected"]) <= 5


# --------------------------------------------------------- the selection
@requires_selection
class TestSelection:
    @pytest.fixture(autouse=True)
    def _load_it(self):
        self.d = _load(SELECTED)
        self.targets = self.d["holdout_targets"]
        self.scored = [t for t in self.targets if t.get("scored")]
        self.controls = [t for t in self.targets if not t.get("scored")]

    def test_the_gold_split_is_exactly_balanced(self):
        from collections import Counter
        c = Counter(t["ground_truth_outcome"] for t in self.scored)
        assert len(c) == 2, f"expected two gold classes, got {dict(c)}"
        assert c["advance"] == c["pass"], (
            f"balanced sampling means equal classes; got {dict(c)}")

    def test_n_is_derived_from_availability_and_never_padded(self):
        avail = self.d["availability"]
        want = min(avail["discriminating_advance"], avail["discriminating_pass"],
                   self.d["policy"]["per_class_cap"])
        assert self.d["policy"]["n_per_class"] == want
        assert len(self.scored) == 2 * want

    def test_the_chance_floor_is_recorded_and_is_one_half(self):
        assert self.d["policy"]["chance_floor"] == 0.5

    def test_controls_are_balanced_and_unscored(self):
        from collections import Counter
        c = Counter(t["bucket"] for t in self.controls)
        assert c["clean_advance"] == c["clean_pass"] == 10
        assert all(not t.get("scored") for t in self.controls)

    def test_ids_are_unique_across_scored_and_controls(self):
        ids = [t["id"] for t in self.targets]
        assert len(set(ids)) == len(ids), (
            "a control id that collides with a scored id produces two items "
            "with one item_id")

    def test_unit_keys_are_unique(self):
        keys = [t["unit_key"] for t in self.targets]
        assert len(set(keys)) == len(keys), "one item per distinct unit"

    def test_the_cap_is_scoped_within_gold_class(self):
        assert self.d["policy"]["rule_cap_scope"] == "within gold class"
        assert self.d["policy"]["advance_rule_cap_applied"] is False
        assert self.d["policy"]["advance_rule_cap_note"], (
            "the advance class's single-rule concentration must be disclosed, "
            "not silently recorded as satisfied")

    def test_no_pass_class_rule_exceeds_the_cap(self):
        from collections import Counter
        n = len(self.scored) // 2
        allowed = int(self.d["policy"]["rule_cap_requested"] * n)
        c: Counter = Counter()
        for t in self.scored:
            if t["ground_truth_outcome"] != "pass":
                continue
            for r in dict.fromkeys(list(t.get("driving_rule_ids") or [])
                                   + list(t.get("rescued_by_rule") or [])):
                c[r] += 1
        assert c, "the pass class must record its rules"
        for rule, k in c.items():
            assert k <= allowed, f"{rule} at {k} exceeds the permitted {allowed}"

    def test_no_discriminating_target_sits_in_a_boundary_gap(self):
        for t in self.scored:
            assert HP.boundary_gap(t["as_of"]) is None, (
                f"{t['id']} is dated {t['as_of']}, inside an inferred window "
                f"boundary, so its gold is not evidenced anywhere in the corpus")

    def test_it_pins_the_pool_it_was_drawn_from(self):
        assert self.d["source_pool"]["sha256"] == POOL_SHA256


# ------------------------------------------------------------- the items
@requires_items
class TestGeneratedFamilyFour:
    @pytest.fixture(autouse=True)
    def _load_it(self):
        self.items = [i for i in _load(ITEMS)["items"]
                      if i["family"] == "implicit_rule_application"]
        self.scored = [i for i in self.items if (i["meta"] or {}).get("scored")]

    def test_every_scored_item_carries_the_chance_floor(self):
        """A report cannot print a family-4 score without the floor beside it if
        the floor rides on every item."""
        for i in self.scored:
            assert i["meta"].get("chance_floor") == 0.5, (
                f"{i['item_id']} carries no chance floor")

    def test_the_emitted_gold_split_is_balanced(self):
        from collections import Counter
        c = Counter(str(i["gold"]["value"]) for i in self.scored)
        assert c["advance"] == c["pass"], dict(c)

    def test_a_constant_responder_scores_exactly_the_floor(self):
        """The acceptance criterion, checked on the artifact rather than trusted
        from the report that measured it."""
        from collections import Counter
        c = Counter(str(i["gold"]["value"]) for i in self.scored)
        n = len(self.scored)
        for verdict in ("pass", "advance"):
            assert abs(c[verdict] / n - 0.5) < 1e-9, (
                f"a constant '{verdict}' responder scores {c[verdict] / n:.3f}, "
                f"not the 0.500 the pre-registration commits to")

    def test_halloran_targets_are_asked_the_halloran_question(self):
        """v1 asked seven Halloran targets 'Does Westbrook advance this or pass?'
        while their gold came from Halloran's own rules. Seven of twenty scored
        items were unanswerable in principle."""
        hl = [i for i in self.items if (i["meta"] or {}).get("firm") == "halloran"]
        if not hl:
            pytest.skip("no Halloran targets in this selection")
        for i in hl:
            assert "Halloran" in i["question"], (
                f"{i['item_id']} is a Halloran target asked about another firm")
            assert "Does Westbrook" not in i["question"]

    def test_westbrook_targets_are_asked_the_westbrook_question(self):
        wb = [i for i in self.items if (i["meta"] or {}).get("firm") == "westbrook"]
        assert wb, "the selection must contain Westbrook targets"
        for i in wb:
            assert "Westbrook" in i["question"]

    def test_the_fact_block_prints_every_lever_the_advance_rule_reads(self):
        """A lever the block does not print makes two targets with identical
        printed facts carry opposite gold -- undecidable, not hard. The fields
        the advance-class rule reads are configured privately; the block is
        asserted to carry them."""
        levers = LEVERS_THE_ADVANCE_RULE_READS
        if not levers:
            import pytest
            pytest.skip("lever labels not configured (private side-file absent)")
        for i in self.scored[:20]:
            q = i["question"]
            for label in levers:
                assert label in q, f"the fact block must print {label!r}"

    def test_no_source_row_is_used_twice(self):
        seen = set()
        for i in self.items:
            key = i["meta"]["source_row_id"]
            assert key not in seen, f"unit {key} used twice"
            seen.add(key)
