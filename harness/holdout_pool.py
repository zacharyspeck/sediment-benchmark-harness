r"""Family 4: ingest the saturated v6 pool and select a gold-balanced item set.

WHAT THIS MODULE IS NOT
-----------------------
It is not a holdout generator. `harness/holdout.py` drives the corpus's own
`v4_holdout.build` and is the right tool when new targets are needed. This
module never generates a target: it reads `data/holdout-pool-v6.json`, which was
produced upstream by `v4_holdout_saturate.py --n 4000` against the corpus's
implicit-rule engine, and selects from it. Re-deriving the rule engine here would give two
implementations of the same rules that drift apart while being reported as one
population.

WHY BALANCED SAMPLING
---------------------
Under the natural distribution the discriminating slice is 87.3% `pass`
(537 of 615), so a responder that reads nothing and always answers "pass"
scores 87.3%. That is not a measurement of anything. Sampling equal numbers of
each gold class puts the chance floor at 50%, and the floor is printed beside
the score rather than left for the reader to compute.

The balance is derived, not imposed: `n = min(available_advance, available_pass,
75)` where "available" is the count of admissible distinct units measured after
every gate. If the advance class measures 3, the family runs at 3 and 3 and says
so. There is no floor and nothing is padded.

THE PIPELINE, IN ORDER
----------------------
  1. normalise      rescued_by_rule folded into driving_rule_ids
  2. boundary gap   discriminating targets falling inside an inferred
                    boundary region are dropped -- their gold is not evidenced
  3. unit key       (firing rule-set, binned lever), lever must be PRINTED
  4. collision      the target must appear in no corpus document
  5. dedupe         one target per unit key
  6. fact dedupe    one target per distinct rendered fact block, advance class
  7. partition      (bucket, gold) -> four disjoint pools
  8. select         balanced across gold class; 35% per-rule cap WITHIN the
                    pass class; controls at 10 + 10

Every stage records what it dropped and why. Deterministic throughout: the only
randomness is a seeded shuffle over a sorted list.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
from collections import Counter, defaultdict

from .capacity import holdout_unit_key
from .templates import substitute

#: Per-class ceiling, pre-registered. `n_advance = n_pass = min(avail, avail, 75)`.
BALANCED_CAP = 75

#: No single rule may exceed this share of the PASS class. Unchanged from v1;
#: what changed is that it is now scoped within gold class rather than across
#: the family. See PRE-REGISTRATION.md, amendment of 2026-08-20.
RULE_CAP = 0.35

#: Controls, excluded from the score and reported on their own line.
N_CONTROL_PER_CLASS = 10

#: Applicability boundaries for `RULE-A1`, the single rule that produces the
#: advance class. They are INFERRED from observed behaviour rather than stated
#: in any document, so a discriminating target falling between the last
#: observation on one side of a boundary and the first on the other has gold
#: that no document evidences, and is dropped.
#:
#: THE BOUNDARY VALUES ARE NOT PUBLISHED. They are a property of the corpus,
#: and printing them here would let a reader resolve the advance class without
#: consulting the corpus at all -- which is the entire thing family 4 measures.
#: They are supplied at runtime alongside the private pool, via
#: `config/boundary_gaps.json` (a list of [label, start, end] triples). With no
#: such file present, no target is dropped and `select()` refuses to run,
#: because silently skipping this gate would change the population while
#: reporting it as unchanged.
def _load_boundary_gaps():
    import json as _json
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "boundary_gaps.json")
    if not os.path.exists(p):
        return ()
    with io.open(p, encoding="utf-8") as fh:
        return tuple(tuple(g) for g in _json.load(fh))


BOUNDARY_GAPS = _load_boundary_gaps()


class PoolUnavailable(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pool(path: str) -> tuple[list[dict], str]:
    if not os.path.exists(path):
        raise PoolUnavailable(
            f"{path} does not exist. It is copied from the corpus repo by "
            f"tools/phase1_check_inputs.py; this repo does not regenerate it.")
    doc = json.load(io.open(path, encoding="utf-8"))
    targets = doc.get("targets") or doc.get("holdout_targets") or []
    if not targets:
        raise PoolUnavailable(f"{path} carries no targets")
    return targets, sha256_file(path)


_FIG = __import__("re").compile(r"\$\d+(?:\.\d+)?M")


def make_screen(index):
    """A collision screen with `harness.holdout.screen_target`'s semantics, fast.

    The shipped screen answers "do this target's revenue and EBITDA figures
    co-occur in one document" by scanning every document twice, per target. Over
    897 targets against 10,285 documents that is ~1,800 passes of a 155 MB tree.
    The set of `$N.NM` tokens per document is computed once instead and the same
    question becomes a set intersection.

    Identical semantics, not merely similar: `"$9.6M" in text` and the token
    regex agree because the `$` anchors the left edge and `M` the right, so
    neither `$19.6M` nor `$9.65M` can satisfy one and not the other.
    """
    fig: dict[str, set] = {}
    for d in index.docs:
        for tok in set(_FIG.findall(d.text)):
            fig.setdefault(tok, set()).add(d.rel)
    # Every synthetic target id begins with this. One substring test over the
    # cached blob decides whether ANY id could collide.
    ids_possible = index.contains_anywhere("holdout-")

    def screen(t: dict, _index=None) -> list[str]:
        hits = []
        tid = t.get("id")
        if ids_possible and tid and index.contains_anywhere(str(tid)):
            hits.append(f"name {tid!r} appears in the corpus")
        rev, eb = t.get("revenue_m"), t.get("adj_ebitda_m")
        if rev is not None and eb is not None:
            a, b = f"${rev}M", f"${eb}M"
            both = fig.get(a, set()) & fig.get(b, set())
            if both:
                hits.append(f"revenue {a} and EBITDA {b} co-occur in "
                            f"{len(both)} document(s), e.g. {sorted(both)[0]}")
        return hits

    screen.figure_tokens = len(fig)                                # for reporting
    return screen


def _stable_shuffle(items: list, seed: int, salt: str) -> list:
    out = sorted(items, key=str)
    random.Random(f"{seed}:{salt}").shuffle(out)
    return out


def normalise(t: dict) -> dict:
    """Fold `rescued_by_rule` into `driving_rule_ids`.

    Advance-class targets record their deciding rule under `rescued_by_rule`
    rather than `driving_rule_ids`. `holdout_unit_key` reads only the latter,
    so without this fold every advance target rejects as "no driving rule
    recorded" and the class measures zero. A rescue is what decided the
    verdict; it is a firing rule.
    """
    out = dict(t)
    rescue = list(t.get("rescued_by_rule") or [])
    out["driving_rule_ids"] = list(t.get("driving_rule_ids") or []) + rescue
    out["_rescue_rules"] = rescue

    # IDS IN THE POOL ARE NOT UNIQUE. `v4_holdout.build` numbers each batch from
    # zero, and the saturating driver calls it once per seed, so `holdout-we-000`
    # names 90-odd different companies. Everything downstream keys on the id:
    # `_row_id` for the "targets seen" count, and `holdout_unit_key` for a
    # control's entire unit key. Left alone, controls with opposite gold collapse
    # onto one key and the audit reports 55 targets seen out of 897 with a
    # NEGATIVE near-duplicate collapse. `_seed` disambiguates: ids are unique
    # within a batch, and the pool was deduplicated on a figure signature, so
    # (id, seed) is unique across the pool.
    seed = t.get("_seed")
    ident = str(t.get("id") or "")
    if seed is not None and not ident.endswith(f"-s{seed}"):
        out["id"] = f"{ident}-s{seed}"
        out["pool_id"] = ident
    return out


def boundary_gap(as_of: str) -> str | None:
    for name, lo, hi in BOUNDARY_GAPS:
        if lo < str(as_of) < hi:
            return name
    return None


def render_fact_block(fact_block: str, target: dict, slots: dict) -> str:
    """The fact block exactly as an arm would see it, for the duplicate test."""
    binding = {name: target.get(spec if isinstance(spec, str) else name)
               for name, spec in (slots or {}).items()}
    return substitute(fact_block or "", binding)


# --------------------------------------------------------------------- admit
def admit(targets: list[dict], lever_map: dict, printed: set, index,
          screen, fact_block: str = "", slots: dict | None = None) -> dict:
    """Run every gate. Returns admitted units per class plus a full drop trace."""
    drops: Counter = Counter()
    gap_drops: Counter = Counter()
    gap_detail: list[dict] = []
    inadmissible: Counter = Counter()
    collisions: list[str] = []

    by_key: dict[str, dict] = {}
    key_collapse = 0

    for raw in targets:
        t = normalise(raw)
        bucket = str(t.get("bucket") or "")

        # 2. boundary gap -- discriminating only; a control fires no rule and
        #    its gold does not depend on any inferred window.
        if bucket == "discriminating":
            g = boundary_gap(t.get("as_of"))
            if g:
                gap_drops[g] += 1
                gap_detail.append({"gap": g, "as_of": t.get("as_of"),
                                   "id": t.get("id"),
                                   "is_add_on": bool(t.get("is_add_on")),
                                   "gold": t.get("ground_truth_outcome")})
                drops["boundary gap (%s): gold not evidenced in the corpus" % g] += 1
                continue

        # 3. unit key
        key, why = holdout_unit_key(t, lever_map, printed)
        if not key:
            inadmissible[why] += 1
            drops["inadmissible: " + why] += 1
            continue

        # 4. collision screen -- a holdout target that appears in a document is
        #    not held out. The corpus's saturate script ran its own screen; this
        #    one is additional and can only subtract.
        hits = screen(t, index) if screen else []
        if hits:
            collisions.append(f"{t.get('id')}: {hits[0]}")
            drops["corpus collision"] += 1
            continue

        # 5. one target per unit key
        if key in by_key:
            key_collapse += 1
            drops["duplicate of an already-admitted unit"] += 1
            continue
        by_key[key] = t

    # 6. identical printed fact block. Distinct deals stay distinct; duplicate
    #    QUESTIONS collapse. Applied to the advance class per instruction, and
    #    measured on every class so the number is comparable.
    fact_collapse: Counter = Counter()
    seen_block: dict[tuple[str, str], str] = {}
    kept: dict[str, dict] = {}
    for key in sorted(by_key):
        t = by_key[key]
        cls = _class_of(t)
        block = render_fact_block(fact_block, t, slots or {}) if fact_block else key
        sig = (cls, block)
        if sig in seen_block:
            fact_collapse[cls] += 1
            drops["identical printed fact block"] += 1
            continue
        seen_block[sig] = key
        kept[key] = t

    classes: dict[str, dict] = defaultdict(dict)
    for key, t in kept.items():
        classes[_class_of(t)][key] = t

    return {
        "units": kept,
        "classes": {k: dict(v) for k, v in classes.items()},
        "drops": dict(drops.most_common()),
        "inadmissible_reasons": dict(inadmissible.most_common()),
        "boundary_gap_drops": dict(gap_drops),
        "boundary_gap_detail": sorted(gap_detail, key=lambda d: (d["gap"], d["as_of"])),
        "unit_key_collapse": key_collapse,
        "fact_block_collapse": dict(fact_collapse),
        "collisions": sorted(collisions),
        "targets_seen": len(targets),
    }


def _class_of(t: dict) -> str:
    """The four disjoint pools the sampler draws from."""
    bucket = str(t.get("bucket") or "")
    gold = str(t.get("ground_truth_outcome") or "")
    if bucket == "discriminating":
        return "discriminating_" + gold
    return bucket


# -------------------------------------------------------------------- cap
def rules_of(key: str) -> list[str]:
    return [part.split("[")[0] for part in key.split("|")]


def select_under_cap(keys: list[str], want: int, cap: float) -> dict:
    """Pick exactly `want` keys with no rule above `cap` of them.

    Differs from `harness.holdout.apply_rule_cap`, which computes its own n:
    here `want` is set by the balance requirement and the cap has to be honoured
    at that n or reported as impossible at it. When the requested cap cannot be
    met, the smallest achievable cap is used and recorded, per the
    pre-registration -- the cap is never quietly exceeded and no rule is
    quietly dropped.
    """
    by_rule: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        for r in rules_of(k):
            by_rule[r].append(k)
    caps = {r: len(v) for r, v in by_rule.items()}
    if not caps or want <= 0:
        return {"selected": [], "cap_used": cap, "impossible": not caps,
                "reason": "no rule has capacity" if not caps else "",
                "per_rule": {}, "rule_capacity": caps, "want": want}

    def feasible(c: float, n: int) -> bool:
        allowed = int(c * n)
        return allowed >= 1 and sum(min(v, allowed) for v in caps.values()) >= n

    used, impossible, reason = cap, False, ""
    if not feasible(cap, want):
        impossible = True
        used = 1.0 / len(caps)
        reason = (f"{len(caps)} rules have capacity {caps}; at n={want} a "
                  f"{cap:.2f} cap allows {int(cap * want)} per rule, covering "
                  f"{sum(min(v, int(cap * want)) for v in caps.values())} < {want}")
        while used < 1.0 and not feasible(used, want):
            used += 0.01
    allowed = max(1, int(used * want))

    # Scarce rules first so an abundant one cannot starve them, then key order.
    order = sorted(keys, key=lambda k: (min(caps[r] for r in rules_of(k)), k))
    taken: Counter = Counter()
    selected: list[str] = []
    for k in order:
        if len(selected) >= want:
            break
        if any(taken[r] >= allowed for r in rules_of(k)):
            continue
        selected.append(k)
        for r in rules_of(k):
            taken[r] += 1
    return {"selected": sorted(selected), "cap_used": round(used, 4),
            "impossible": impossible, "reason": reason, "allowed_per_rule": allowed,
            "per_rule": dict(sorted(taken.items())), "rule_capacity": caps,
            "want": want}


# ------------------------------------------------------------------ select
def select(admitted: dict, seed: int, balanced_cap: int = BALANCED_CAP,
           rule_cap: float = RULE_CAP,
           n_control: int = N_CONTROL_PER_CLASS) -> dict:
    """Balanced across gold class. Availability decides n; nothing is padded."""
    if not BOUNDARY_GAPS:
        raise PoolUnavailable(
            "no applicability boundaries are configured, so the boundary-gap "
            "gate cannot have run and this selection would report a population "
            "it did not actually filter. Supply config/boundary_gaps.json "
            "alongside the private pool. See the note beside BOUNDARY_GAPS.")
    cls = admitted["classes"]
    adv = sorted(cls.get("discriminating_advance") or {})
    pas = sorted(cls.get("discriminating_pass") or {})

    n_want = min(len(adv), len(pas), balanced_cap)

    # The pass class must be able to supply n_want under the per-rule cap. If it
    # cannot at the requested cap, the achievable cap is used and recorded --
    # the cap is never relaxed silently and n is never padded to compensate.
    capped = select_under_cap(pas, n_want, rule_cap)
    n = min(n_want, len(capped["selected"]))
    if n < n_want:
        capped = select_under_cap(pas, n, rule_cap)

    adv_sel = sorted(_stable_shuffle(adv, seed, "ira-advance")[:n])
    pass_sel = sorted(capped["selected"][:n]) if n else []
    if len(pass_sel) > n:
        pass_sel = pass_sel[:n]

    # The advance class carries exactly one rule, so no cap applies. That is
    # structural rather than a sampling choice. Recorded, not hidden.
    adv_rules = Counter()
    for k in adv_sel:
        for r in rules_of(k):
            adv_rules[r] += 1

    controls: dict[str, list[str]] = {}
    for cname in ("clean_advance", "clean_pass"):
        avail = sorted(cls.get(cname) or {})
        controls[cname] = sorted(_stable_shuffle(avail, seed, "ira-" + cname)[:n_control])

    return {
        "n_per_class": n,
        "n_scored": 2 * n,
        "available": {"discriminating_advance": len(adv),
                      "discriminating_pass": len(pas),
                      "clean_advance": len(cls.get("clean_advance") or {}),
                      "clean_pass": len(cls.get("clean_pass") or {})},
        "balanced_cap": balanced_cap,
        "n_want_before_cap": n_want,
        "advance_keys": adv_sel,
        "pass_keys": pass_sel,
        "control_keys": controls,
        "pass_rule_cap": {k: v for k, v in capped.items() if k != "selected"},
        "advance_rule_distribution": dict(adv_rules),
        "advance_rule_cap_applied": False,
        "advance_rule_cap_note": (
            "One rule constitutes 100% of the advance class. It is the only rule "
            "in this corpus that produces that class, so the concentration is "
            "structural rather than a sampling choice. The 35% cap is not applied "
            "here and is not silently satisfied; it is disclosed."),
        "chance_floor": 0.5 if n else None,
    }
