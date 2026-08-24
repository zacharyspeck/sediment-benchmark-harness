"""Provenance pinning.

Every artifact this repo emits records the ledger and corpus hashes it was
computed from. The corpus has already been rebuilt once under the benchmark's
feet; nothing detected it. These tests keep the detector working.
"""

from __future__ import annotations

import io
import json
import os

from harness.baseline import baseline, corpus_fingerprint, provenance_lines, shard_hashes


def _write_shard(d, name, obj):
    p = os.path.join(d, name + ".json")
    with io.open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return p


def test_shard_hashes_flags_an_index_that_disagrees_with_disk(tmp_path):
    """The exact failure the live corpus exhibits: index.json is stale."""
    d = tmp_path / "ledger"
    d.mkdir()
    _write_shard(str(d), "deals", {"deals": [{"id": "a"}]})
    # An index claiming a hash the file does not have.
    _write_shard(str(d), "index", {
        "version": "9.9.9", "seed": 1,
        "shards": {"deals": {"bytes": 999, "sha256": "0" * 64}},
    })
    out = shard_hashes(str(d))
    assert out["index_version"] == "9.9.9"
    assert out["shards"]["deals"]["matches_index"] is False
    assert out["index_disagrees"] == ["deals"]


def test_shard_hashes_agrees_when_the_index_is_honest(tmp_path):
    import hashlib

    d = tmp_path / "ledger"
    d.mkdir()
    p = _write_shard(str(d), "deals", {"deals": []})
    raw = io.open(p, "rb").read()
    _write_shard(str(d), "index", {
        "version": "1.0.0", "seed": 7,
        "shards": {"deals": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}},
    })
    out = shard_hashes(str(d))
    assert out["shards"]["deals"]["matches_index"] is True
    assert out["index_disagrees"] == []


def test_shard_hashes_reports_shards_the_index_does_not_declare(tmp_path):
    d = tmp_path / "ledger"
    d.mkdir()
    _write_shard(str(d), "surprise", {"x": 1})
    _write_shard(str(d), "index", {"version": "1", "seed": 1, "shards": {}})
    out = shard_hashes(str(d))
    assert "surprise" in out["shards"]
    assert out["shards"]["surprise"]["present"] is True


def test_corpus_fingerprint_changes_when_a_document_changes_size(tmp_path):
    c = tmp_path / "corpus"
    (c / "westbrook").mkdir(parents=True)
    doc = c / "westbrook" / "a.md"
    doc.write_text("hello", encoding="utf-8")
    before = corpus_fingerprint(str(c))
    assert before["documents"] == 1
    assert before["documents_by_dir"] == {"westbrook": 1}

    doc.write_text("hello world, considerably longer", encoding="utf-8")
    after = corpus_fingerprint(str(c))
    assert after["tree_digest"] != before["tree_digest"]


def test_corpus_fingerprint_changes_when_a_document_is_added(tmp_path):
    c = tmp_path / "corpus"
    (c / "westbrook").mkdir(parents=True)
    (c / "westbrook" / "a.md").write_text("x", encoding="utf-8")
    before = corpus_fingerprint(str(c))
    (c / "westbrook" / "b.md").write_text("y", encoding="utf-8")
    after = corpus_fingerprint(str(c))
    assert after["documents"] == before["documents"] + 1
    assert after["tree_digest"] != before["tree_digest"]


def test_corpus_fingerprint_is_stable_across_calls(tmp_path):
    c = tmp_path / "corpus"
    (c / "halloran").mkdir(parents=True)
    (c / "halloran" / "a.md").write_text("x", encoding="utf-8")
    assert corpus_fingerprint(str(c))["tree_digest"] == corpus_fingerprint(str(c))["tree_digest"]


def test_corpus_fingerprint_handles_an_absent_tree():
    out = corpus_fingerprint("/no/such/place")
    assert out["present"] is False


def test_provenance_lines_say_so_when_the_corpus_is_unconfigured(tmp_path):
    d = tmp_path / "ledger"
    d.mkdir()
    _write_shard(str(d), "index", {"version": "1", "seed": 1, "shards": {}})
    lines = provenance_lines(baseline(str(d), None))
    assert any("NOT CONFIGURED" in ln for ln in lines)


def test_corpus_dir_resolves_against_the_live_config():
    """config/ledger.yaml must point at a real document tree.

    If this fails the grounding and contradiction gates cannot run, and every
    capacity number degrades to a ledger row count.
    """
    from harness.cli import resolve_corpus_dir

    p = resolve_corpus_dir()
    if p is None:
        import pytest

        pytest.skip("corpus not present in this checkout")
    assert os.path.isdir(p)
    assert corpus_fingerprint(p)["documents"] > 0
