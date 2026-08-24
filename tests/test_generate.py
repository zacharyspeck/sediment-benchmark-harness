"""The generator: no reuse, no padding, and the same bytes every time.

The no-reuse rule is the one that matters most. A benchmark whose item count is
really the same facts asked twice is worse than a smaller honest one, and a
warning is something a tired person scrolls past -- so the generator raises.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import pytest

from harness.generate import (
    DuplicateSourceRow,
    _assert_no_reuse,
    _item_id,
    _stable_shuffle,
    largest_under_cap,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(REPO, "data", "items-v1.json")


# ------------------------------------------------------------------ no reuse
def _item(iid, fam, src):
    return {"item_id": iid, "family": fam, "meta": {"source_row_id": src}}


def test_reuse_within_a_family_raises():
    with pytest.raises(DuplicateSourceRow) as exc:
        _assert_no_reuse([
            _item("a", "single_fact_recall", "wb-ember"),
            _item("b", "single_fact_recall", "wb-ember"),
        ])
    assert "wb-ember" in str(exc.value)
    assert "no deal asked two ways" in str(exc.value)


def test_the_same_row_in_two_different_families_is_allowed():
    """The declared unit of independence is per family, not global."""
    _assert_no_reuse([
        _item("a", "single_fact_recall", "wb-ember"),
        _item("b", "absence_and_abstention", "wb-ember"),
    ])


def test_an_item_without_a_source_row_raises():
    """Otherwise the no-reuse rule is unauditable for that item."""
    with pytest.raises(DuplicateSourceRow):
        _assert_no_reuse([{"item_id": "a", "family": "f", "meta": {}}])


# ---------------------------------------------------------------- the caps
def test_a_reachable_cap_is_applied_as_requested():
    n, cap, impossible = largest_under_cap({"a": 100, "b": 100, "c": 100,
                                            "d": 100, "e": 100, "f": 100,
                                            "g": 100}, 0.20)
    assert impossible is False
    assert cap == 0.20
    assert n >= 150


def test_a_cap_too_small_for_the_number_of_buckets_falls_back_and_says_so():
    """Four buckets cannot each stay under 20% -- 4 x 0.20 = 0.80 < 1."""
    n, cap, impossible = largest_under_cap({"a": 50, "b": 50, "c": 50, "d": 50}, 0.20)
    assert impossible is True
    assert cap == pytest.approx(0.25)
    assert n > 0


def test_the_cap_is_bounded_by_the_scarcest_bucket_too():
    n, _cap, _imp = largest_under_cap({"a": 14, "b": 9, "c": 6}, 0.35)
    assert n == 20


# ------------------------------------------------------------- determinism
def test_stable_shuffle_is_seed_determined_and_order_independent():
    a = _stable_shuffle(["x", "y", "z", "w"], 1, "f")
    b = _stable_shuffle(["w", "z", "y", "x"], 1, "f")
    assert a == b, "input order leaked into the shuffle"
    assert _stable_shuffle(["x", "y", "z", "w"], 2, "f") != a or len(set(a)) == 1


def test_item_ids_are_a_function_of_template_and_row():
    assert _item_id("sfr-01", "wb-ember") == _item_id("sfr-01", "wb-ember")
    assert _item_id("sfr-01", "wb-ember") != _item_id("sfr-01", "wb-marlin")


# ------------------------------------------------------------ the artifact
@pytest.mark.skipif(not os.path.exists(ITEMS), reason="items-v1.json not generated")
class TestGeneratedItems:
    @pytest.fixture(scope="class")
    def payload(cls):
        return json.load(io.open(ITEMS, encoding="utf-8"))

    def test_zero_duplicate_source_rows_per_family(self, payload):
        _assert_no_reuse(payload["items"])

    def test_every_item_records_what_it_came_from(self, payload):
        for it in payload["items"]:
            m = it["meta"]
            assert m.get("template_id"), f"{it['item_id']} has no template_id"
            assert m.get("source_row_id"), f"{it['item_id']} has no source_row_id"
            assert m.get("difficulty"), f"{it['item_id']} has no difficulty"
            assert "bucket" in m
            assert "recency" in m
            assert it.get("family")
            assert it.get("question")
            assert it.get("gold")

    def test_every_item_records_the_ledger_it_was_built_from(self, payload):
        shards = payload["provenance"]["ledger"]["shards"]
        assert shards["deals"]["sha256"]
        assert payload["provenance"]["corpus"]["tree_digest"]
        assert payload["seed"] == 20260819

    def test_no_family_exceeds_the_pre_registered_cap(self, payload):
        from collections import Counter

        n = Counter(i["family"] for i in payload["items"] if i["meta"].get("scored", True))
        for fam, k in n.items():
            assert k <= 150, f"{fam} emitted {k} scored items, above the cap of 150"

    def test_family_one_honours_the_gold_field_cap(self, payload):
        from collections import Counter

        rows = [i for i in payload["items"] if i["family"] == "single_fact_recall"]
        per = Counter(i["meta"]["template_id"] for i in rows)
        rep = payload["selection_report"]["families"]["single_fact_recall"]
        cap = rep["field_cap_used"] or 0.20
        allowed = int(cap * len(rows)) + 1
        for tid, k in per.items():
            assert k <= allowed, f"{tid} supplies {k} of {len(rows)}, above the {cap} cap"

    def test_recency_items_are_tagged_and_in_their_own_family(self, payload):
        rec = [i for i in payload["items"] if i["family"] == "recency"]
        assert rec, "no recency items"
        for i in rec:
            assert i["meta"]["recency"] is True
        for i in payload["items"]:
            if i["family"] != "recency":
                assert i["meta"]["recency"] is False

    def test_holdout_controls_are_present_and_marked_unscored(self, payload):
        ira = [i for i in payload["items"] if i["family"] == "implicit_rule_application"]
        if not ira:
            pytest.skip("holdout-targets-v2.json was not present at generation time")
        scored = [i for i in ira if i["meta"]["scored"]]
        controls = [i for i in ira if not i["meta"]["scored"]]
        assert scored and controls
        for c in controls:
            assert c["meta"]["bucket"] in ("clean_advance", "clean_pass")
        for s in scored:
            assert s["meta"]["bucket"] == "discriminating"

    def test_the_item_file_loads_through_the_runner(self, payload):
        from harness.runner import load_items

        assert len(load_items(ITEMS)) == len(payload["items"])

    def test_regenerating_produces_identical_bytes(self, payload, tmp_path):
        """Same seed + same ledger -> the same file, byte for byte.

        Slow (it rebuilds the corpus index), so it is skipped unless the corpus
        is present. Without it, "deterministic" is an assertion rather than a
        property.
        """
        from harness.cli import resolve_corpus_dir

        if resolve_corpus_dir() is None:
            pytest.skip("corpus not present")
        pytest.importorskip("harness.generate")
        from harness.baseline import baseline
        from harness.cli import resolve_ledger
        from harness.corpusindex import load_corpus_index
        from harness.generate import build_items, write_items
        from harness.paths import permit_root
        from harness.templates import load_all_templates, load_template_file

        permit_root(str(tmp_path))
        led = resolve_ledger(None, quiet=True)
        corpus = resolve_corpus_dir()
        index = load_corpus_index(corpus, [d.get("codename") for d in led.deals])
        items_dir = os.path.join(REPO, "items")
        docs = {}
        for f in sorted(os.listdir(items_dir)):
            if f.endswith(".yaml"):
                fam, _t, doc = load_template_file(os.path.join(items_dir, f))
                docs[fam] = doc
        hv2 = json.load(io.open(os.path.join(REPO, "data", "holdout-targets-v2.json"),
                                encoding="utf-8"))
        prov = baseline(os.path.join(os.path.dirname(corpus), "ledger-v4"), corpus)

        hashes = []
        for i in range(2):
            built = build_items(led, load_all_templates(items_dir), index, docs, hv2)
            p = write_items(str(tmp_path / f"g{i}.json"), built, prov)
            hashes.append(hashlib.sha256(io.open(p, "rb").read()).hexdigest())
        assert hashes[0] == hashes[1], "two generations of the same inputs differ"
