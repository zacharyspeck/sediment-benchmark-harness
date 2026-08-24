"""Regenerate the holdout target set using the CORPUS's own generator.

WHY NOT WRITE A NEW GENERATOR
-----------------------------
The shipped 55 targets and any new ones have to be ONE population. A separate
implementation -- however careful -- would differ in its value distributions,
its rounding, its dead bands and its rule evaluation, and the old and new items
would then be testing subtly different things while being reported as one
number. So `v4_holdout.build` is imported from the corpus and driven directly,
along with the exact helpers `v4_main` passes it.

READ-ONLY IMPORT
----------------
`sys.dont_write_bytecode` is set for the duration so importing the corpus
modules cannot drop `__pycache__` directories into a repo this workstream must
not write to. Output goes to `data/` in THIS repo, never back to the corpus.

WHAT SATURATION MEANS HERE
--------------------------
Not "generate N". The generator is sampled until the set of distinct admissible
units stops growing, then the ceiling is whatever it converged to. Distinctness
is the same rule the audit uses: (firing rule-set, binned lever), admitted only
when every field the rule reads is printed in the item's fact block. A target
whose deciding field the item never shows is undecidable, not hard.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
from collections import Counter, defaultdict

from .capacity import holdout_unit_key, lever_bin, printed_fields


class GeneratorUnavailable(RuntimeError):
    pass


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus_generator(corpus_repo: str):
    """Import v4_holdout + its helpers from the corpus, writing nothing."""
    if not os.path.isdir(corpus_repo):
        raise GeneratorUnavailable(f"corpus repo not found: {corpus_repo}")
    prev_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    added = False
    try:
        if corpus_repo not in sys.path:
            sys.path.insert(0, corpus_repo)
            added = True
        HO = importlib.import_module("v4_holdout")
        B = importlib.import_module("build_ledger_v4")
        R = importlib.import_module("v4_rules")
        return HO, B, R
    except ImportError as exc:
        raise GeneratorUnavailable(
            f"could not import the corpus holdout generator from {corpus_repo}: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = prev_flag
        if added:
            try:
                sys.path.remove(corpus_repo)
            except ValueError:
                pass


def generate_pool(HO, B, wb_n: int, hl_n: int) -> list[dict]:
    """One `build()` call at the requested per-firm counts.

    WB_N/HL_N are module constants the generator reads at call time, so raising
    them is how the sampler is driven past its shipped 40/15 without touching
    any of its logic. They are restored afterwards.
    """
    old_wb, old_hl = HO.WB_N, HO.HL_N
    try:
        HO.WB_N, HO.HL_N = wb_n, hl_n
        return HO.build(B.rng_for, B.money, B.safe_margin, B.safe_conc,
                        B.SECTOR_PROFILE, B.SECTORS_WB, B.REGIONS, B.STATES)
    finally:
        HO.WB_N, HO.HL_N = old_wb, old_hl


# ------------------------------------------------------------------ screening
def screen_target(t: dict, index, extra_names=()) -> list[str]:
    """Every collision between this target and the corpus. Empty means clean.

    A holdout target's entire purpose is that it appears in NO document. If any
    of its identifying strings or its distinctive figures already occur in the
    corpus, the target is not a holdout -- a model could have read it.
    """
    hits = []
    for name in list(extra_names) + [t.get("id")]:
        if name and index.contains_anywhere(str(name)):
            hits.append(f"name {name!r} appears in the corpus")
    # Figures are screened as the TRIPLE, not individually: a lone "$41.9M"
    # occurs all over a corpus of 10,261 financial documents and rejecting on it
    # would reject everything. The triple is what identifies a company.
    rev, eb, top = t.get("revenue_m"), t.get("adj_ebitda_m"), t.get("top_customer_pct")
    if rev is not None and eb is not None:
        pair = f"${rev}M"
        if index.contains_anywhere(pair) and index.contains_anywhere(f"${eb}M"):
            # Both figures present somewhere is not yet a collision; require
            # them in the SAME document before rejecting.
            docs_rev = {d.rel for d in index.docs if pair in d.text}
            docs_eb = {d.rel for d in index.docs if f"${eb}M" in d.text}
            both = docs_rev & docs_eb
            if both:
                hits.append(
                    f"revenue {pair} and EBITDA ${eb}M co-occur in "
                    f"{len(both)} document(s), e.g. {sorted(both)[0]}"
                )
    return hits


# ------------------------------------------------------------------ saturation
def saturate(HO, B, index, lever_map: dict, printed: set[str],
             max_rounds: int = 24, start: int = 40, growth: float = 2.0,
             hard_cap: int = 60000) -> dict:
    """Sample until the distinct admissible unit set stops growing.

    Returns the accumulated units plus the trace of how it converged, so the
    ceiling is a measured plateau rather than a stopping point someone chose.
    """
    seen: dict[str, dict] = {}
    rejected: Counter = Counter()
    trace = []
    wb, hl = start, max(1, start // 3)
    dry_rounds = 0
    total_generated = 0

    for rnd in range(max_rounds):
        pool = generate_pool(HO, B, wb, hl)
        total_generated += len(pool)
        before = len(seen)
        for t in pool:
            if str(t.get("bucket")) != "discriminating":
                continue
            key, why = holdout_unit_key(t, lever_map, printed)
            if not key:
                rejected[why] += 1
                continue
            if key in seen:
                rejected["duplicate of an already-admitted unit"] += 1
                continue
            hits = screen_target(t, index)
            if hits:
                rejected["corpus collision: " + hits[0].split(" appears")[0]] += 1
                continue
            seen[key] = t
        gained = len(seen) - before
        trace.append({"round": rnd + 1, "wb_n": wb, "hl_n": hl,
                      "generated": len(pool), "distinct_units": len(seen),
                      "gained": gained})
        dry_rounds = dry_rounds + 1 if gained == 0 else 0
        if dry_rounds >= 3:
            break
        wb = min(int(wb * growth), hard_cap)
        hl = min(int(hl * growth), hard_cap)

    return {"units": seen, "rejected": dict(rejected.most_common()),
            "trace": trace, "total_generated": total_generated,
            "saturated": dry_rounds >= 3}


def saturate_controls(HO, B, index, want: int, max_rounds: int = 12,
                      start: int = 40, growth: float = 2.0) -> dict:
    """Controls, keyed by target id -- they fire no rule and have no lever."""
    seen: dict[str, dict] = {}
    rejected: Counter = Counter()
    wb, hl = start, max(1, start // 3)
    for _rnd in range(max_rounds):
        for t in generate_pool(HO, B, wb, hl):
            b = str(t.get("bucket"))
            if b not in ("clean_advance", "clean_pass"):
                continue
            key = f"{b}:{t.get('revenue_m')}:{t.get('adj_ebitda_m')}:{t.get('top_customer_pct')}:{t.get('gross_logo_retention_pct')}"
            if key in seen:
                continue
            hits = screen_target(t, index)
            if hits:
                rejected["corpus collision"] += 1
                continue
            seen[key] = t
        if len(seen) >= want:
            break
        wb, hl = int(wb * growth), int(hl * growth)
    return {"units": seen, "rejected": dict(rejected)}


def balanced_controls(units: dict, want: int) -> list[str]:
    """Take controls alternately from clean_advance and clean_pass.

    Taking the first `want` in sorted order gave six clean_advance and zero
    clean_pass, because the bucket name leads the key. That defeats the point:
    clean_advance catches an arm that says "pass" to everything, clean_pass
    catches one that says "advance" to everything, and a control set with only
    one of them only catches one of those failures.
    """
    by_bucket: dict[str, list[str]] = {}
    for k in sorted(units):
        by_bucket.setdefault(k.split(":", 1)[0], []).append(k)
    out: list[str] = []
    buckets = sorted(by_bucket)
    i = 0
    while len(out) < want and any(by_bucket[b] for b in buckets):
        b = buckets[i % len(buckets)]
        if by_bucket[b]:
            out.append(by_bucket[b].pop(0))
        i += 1
    return out


# ------------------------------------------------------------- rule-cap policy
def apply_rule_cap(units: dict, cap: float) -> dict:
    """Select the largest set in which no rule exceeds `cap` of the total.

    Returns the selection plus the cap actually used. When the requested cap is
    arithmetically impossible -- too few rules have capacity for any of them to
    stay under it -- the smallest achievable cap is used instead and reported,
    per the pre-registration.
    """
    by_rule: dict[str, list[str]] = defaultdict(list)
    for key in units:
        for part in key.split("|"):
            by_rule[part.split("[")[0]].append(key)
    caps = {r: len(ks) for r, ks in by_rule.items()}
    if not caps:
        return {"selected": [], "cap_used": cap, "impossible": True,
                "per_rule": {}, "reason": "no rule has capacity"}

    total = sum(caps.values())

    def best_n(c: float) -> int:
        out = 0
        for n in range(1, total + 1):
            allowed = int(c * n)
            if sum(min(v, allowed) for v in caps.values()) >= n:
                out = n
        return out

    used, impossible, reason = cap, False, ""
    n = best_n(cap)
    if n == 0:
        impossible = True
        used = 1.0 / len(caps)
        reason = (f"only {len(caps)} rules have capacity; {len(caps)} x {cap:.2f} = "
                  f"{len(caps) * cap:.2f} < 1.00, so no set can satisfy it")
        n = best_n(used)

    allowed = max(1, int(used * n))
    selected: list[str] = []
    taken: Counter = Counter()
    # Deterministic order: rule with least capacity first, then key, so the
    # scarce rules are not starved by the abundant ones.
    order = sorted(units, key=lambda k: (caps[k.split("|")[0].split("[")[0]], k))
    for key in order:
        rule = key.split("|")[0].split("[")[0]
        if taken[rule] >= allowed or len(selected) >= n:
            continue
        selected.append(key)
        taken[rule] += 1
    return {"selected": sorted(selected), "cap_used": round(used, 4),
            "impossible": impossible, "reason": reason,
            "per_rule": dict(taken), "rule_capacity": caps, "target_n": n}
