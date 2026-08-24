"""Corpus and ledger provenance, pinned into every artifact this repo emits.

The corpus is a live working tree owned by another process. It has already
been rebuilt once under this benchmark's feet (v4 -> the 2026-08-19
remediation), which invalidated every grounding measurement taken before it.
Nothing in the harness detected that; `ledger-v4/index.json` carries a
`sha256` per shard and the harness never read one.

So: every artifact records the hashes it was computed from. If a number is
ever disputed, the first question is whether the inputs still hash the same.

`ledger-v4/index.json` is itself unreliable -- as of the remediation its
`portfolio` entry disagrees with the file on disk in both size and hash --
so `shard_hashes()` hashes the files rather than trusting the index, and
reports the disagreement instead of silently preferring one.
"""

from __future__ import annotations

import hashlib
import io
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_hashes(ledger_dir: str) -> dict:
    """Hash every ledger shard on disk and compare against index.json."""
    out: dict = {"ledger_dir": os.path.abspath(ledger_dir), "shards": {}, "index_disagrees": []}
    idx_path = os.path.join(ledger_dir, "index.json")
    idx = {}
    if os.path.exists(idx_path):
        idx = json.load(io.open(idx_path, encoding="utf-8"))
        out["index_version"] = idx.get("version")
        out["index_seed"] = idx.get("seed")
        out["index_sha256"] = _sha256_file(idx_path)

    declared = (idx.get("shards") or {})
    names = sorted(set(declared) | {
        os.path.splitext(f)[0] for f in os.listdir(ledger_dir)
        if f.endswith(".json") and f != "index.json"
    })
    for name in names:
        p = os.path.join(ledger_dir, name + ".json")
        if not os.path.exists(p):
            out["shards"][name] = {"present": False}
            continue
        h = _sha256_file(p)
        size = os.path.getsize(p)
        rec = {"present": True, "bytes": size, "sha256": h}
        d = declared.get(name)
        if d:
            rec["index_bytes"] = d.get("bytes")
            rec["index_sha256"] = d.get("sha256")
            rec["matches_index"] = (h == d.get("sha256"))
            if not rec["matches_index"]:
                out["index_disagrees"].append(name)
        out["shards"][name] = rec
    return out


def corpus_fingerprint(corpus_dir: str) -> dict:
    """Document count and a stable digest over the document tree.

    The digest is over (relative path, size) for every `.md` file, sorted --
    not over file contents, which would take minutes for 10k documents and
    ~43M tokens. It changes when documents are added, removed, renamed or
    resized, which is every remediation this corpus has actually had.
    """
    if not corpus_dir or not os.path.isdir(corpus_dir):
        return {"present": False, "corpus_dir": corpus_dir}
    entries = []
    total = 0
    for root, _dirs, files in os.walk(corpus_dir):
        for f in files:
            if not f.endswith(".md"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, corpus_dir).replace("\\", "/")
            size = os.path.getsize(full)
            entries.append((rel, size))
            total += size
    entries.sort()
    h = hashlib.sha256()
    for rel, size in entries:
        h.update(f"{rel}:{size}\n".encode("utf-8"))
    by_firm: dict = {}
    for rel, _s in entries:
        firm = rel.split("/")[0] if "/" in rel else "(root)"
        by_firm[firm] = by_firm.get(firm, 0) + 1
    return {
        "present": True,
        "corpus_dir": os.path.abspath(corpus_dir),
        "documents": len(entries),
        "documents_by_dir": dict(sorted(by_firm.items())),
        "total_bytes": total,
        "tree_digest": h.hexdigest(),
    }


def baseline(ledger_dir: str, corpus_dir: str | None = None) -> dict:
    """The provenance block every artifact embeds."""
    return {
        "ledger": shard_hashes(ledger_dir),
        "corpus": corpus_fingerprint(corpus_dir) if corpus_dir else {"present": False},
    }


def provenance_lines(b: dict) -> list[str]:
    """Human-readable provenance, for the top of a markdown artifact."""
    L = []
    led = b.get("ledger") or {}
    cor = b.get("corpus") or {}
    L.append(f"- ledger: `{led.get('ledger_dir')}`  index version `{led.get('index_version')}`, seed `{led.get('index_seed')}`")
    for name, rec in sorted((led.get("shards") or {}).items()):
        if not rec.get("present"):
            L.append(f"  - `{name}` MISSING")
            continue
        mark = "" if rec.get("matches_index", True) else "  **disagrees with index.json**"
        L.append(f"  - `{name}` {rec['bytes']:,} B  `{rec['sha256'][:16]}`{mark}")
    if cor.get("present"):
        L.append(f"- corpus: `{cor['corpus_dir']}`  {cor['documents']:,} documents, "
                 f"{cor['total_bytes']:,} B, tree digest `{cor['tree_digest'][:16]}`")
        for d, n in (cor.get("documents_by_dir") or {}).items():
            L.append(f"  - `{d}/` {n:,}")
    else:
        L.append("- corpus: NOT CONFIGURED (set `corpus_dir` in config/ledger.yaml)")
    return L
