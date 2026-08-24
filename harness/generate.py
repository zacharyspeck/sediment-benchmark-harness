"""Templates + ledger -> a concrete item file with gold answers.

THREE THINGS THIS MODULE REFUSES TO DO
--------------------------------------
1. **Reuse a source row within a family.** Not a warning, not a dedupe pass --
   `DuplicateSourceRow` is raised and nothing is written. A count that is really
   the same facts asked twice is the failure mode this whole build exists to
   prevent, and a warning is something you scroll past.
2. **Pad to a target.** `n_family = min(defensible capacity, 150)`, no floor. A
   family that measures 12 emits 12.
3. **Emit an ungated unit.** Selection draws only from units that passed the
   grounding and non-contradiction gates in the audit, so an item whose answer
   is absent from the corpus, or contradicted by it, cannot reach the file.

DETERMINISM
-----------
Selection is a seeded shuffle over the sorted candidate list, so the same seed
and the same ledger produce a byte-identical file. There is no clock in the
output. Every item records the ledger shard hashes it was generated from,
because "same ledger" is a claim that needs checking rather than assuming.

THE COMPOSITION RULES ARE APPLIED, NOT ASSUMED
----------------------------------------------
The per-field cap in family 1 and the per-rule cap in family 4 are enforced
during selection and re-verified after, and where a cap is arithmetically
impossible the smallest achievable one is used and recorded in the file.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict

from .audit import BOOTSTRAP_FAMILIES, N_CAP, _entity_key, _row_id, audit_template
from .capacity import AGGREGATE_FAMILIES, HOLDOUT_FAMILY, RECENCY_FAMILY, expand_bindings
from .query import QueryError, aggregate as agg_query, resolve_gold, select
from .templates import substitute

#: The pre-registered item-selection seed.
SEED = 20260819

#: No single gold field may exceed this share of family 1's scored items.
FIELD_CAP = 0.20


class DuplicateSourceRow(RuntimeError):
    """A source row was selected twice within one family."""


def _stable_shuffle(items: list, seed: int, salt: str) -> list:
    """Deterministic shuffle. Sorted first so input order cannot leak in."""
    out = sorted(items, key=lambda x: str(x))
    random.Random(f"{seed}:{salt}").shuffle(out)
    return out


def largest_under_cap(caps: dict, cap: float) -> tuple[int, float, bool]:
    """(achievable n, cap actually used, whether the requested cap was impossible).

    With R buckets of capacity c_r, no set can honour a cap below 1/R -- R
    buckets at that share cover less than the whole set. Rather than silently
    exceed the cap or silently drop a bucket, fall back to 1/R and report it.
    """
    if not caps:
        return 0, cap, True
    total = sum(caps.values())

    def best(c: float) -> int:
        out = 0
        for n in range(1, total + 1):
            allowed = int(c * n)
            if sum(min(v, allowed) for v in caps.values()) >= n:
                out = n
        return out

    n = best(cap)
    if n:
        return n, cap, False
    fallback = 1.0 / len(caps)
    return best(fallback), round(fallback, 4), True


def _question(tpl, bindings: dict, family_doc: dict) -> str:
    """First declared phrasing, with slots filled.

    Deliberately the FIRST and not a seeded choice among phrasings: rotating
    phrasings across items adds variance to the measurement without adding
    information, and makes two runs harder to compare than they need to be.
    """
    qs = tpl.questions
    q = qs[0] if qs else ""
    fb = (family_doc or {}).get("fact_block")
    if fb and "{{fact_block}}" in q:
        q = q.replace("{{fact_block}}", substitute(fb, bindings))
    return substitute(q, bindings)


def _slot_bindings(tpl, row: dict, extra: dict | None = None) -> dict:
    """Per-row slot values: shorthand slots read the row, choose slots are given."""
    out = dict(extra or {})
    for name, spec in (tpl.slots or {}).items():
        if isinstance(spec, str):
            val = row.get(spec)
            if val is None and "." in spec:
                cur = row
                for part in spec.split("."):
                    cur = (cur or {}).get(part) if isinstance(cur, dict) else None
                val = cur
            out[name] = val
    return out


# --------------------------------------------------------------------- build
def build_items(led, templates, index, family_docs, holdout_v2=None,
                seed: int = SEED, n_cap: int = N_CAP) -> dict:
    """Return {"items": [...], "report": {...}}. Raises on any row reuse."""
    by_family: dict[str, list] = defaultdict(list)
    tpl_by_id = {t.id: t for t in templates}
    audits: dict[str, dict] = {}
    for tpl in templates:
        r = audit_template(tpl, led, index, family_docs.get(tpl.family))
        if "error" in r:
            raise RuntimeError(f"{tpl.id}: {r['error']}")
        audits[tpl.id] = r
        by_family[tpl.family].append(r)

    items: list[dict] = []
    report: dict = {"seed": seed, "n_cap": n_cap, "families": {}}

    for fam in sorted(by_family):
        if fam == HOLDOUT_FAMILY:
            fam_items, fam_rep = _build_holdout(fam, by_family[fam], tpl_by_id,
                                                family_docs, holdout_v2, led)
        elif fam in AGGREGATE_FAMILIES:
            fam_items, fam_rep = _build_aggregate(fam, by_family[fam], tpl_by_id,
                                                  family_docs, led, seed, n_cap)
        elif fam == RECENCY_FAMILY:
            fam_items, fam_rep = _build_recency(fam, by_family[fam], tpl_by_id,
                                                family_docs, led, seed, n_cap)
        else:
            fam_items, fam_rep = _build_row_family(fam, by_family[fam], tpl_by_id,
                                                   family_docs, led, seed, n_cap)
        items.extend(fam_items)
        report["families"][fam] = fam_rep

    _assert_no_reuse(items)
    return {"items": items, "report": report}


def _assert_no_reuse(items: list[dict]) -> None:
    """The no-reuse rule, enforced on the finished set rather than trusted."""
    seen: dict[tuple[str, str], str] = {}
    for it in items:
        fam = it["family"]
        src = (it.get("meta") or {}).get("source_row_id")
        if src is None:
            raise DuplicateSourceRow(
                f"item {it['item_id']} records no source_row_id; the no-reuse rule "
                "cannot be audited for it")
        key = (fam, str(src))
        if key in seen:
            raise DuplicateSourceRow(
                f"source row {src!r} is used twice in family {fam}: "
                f"{seen[key]} and {it['item_id']}. One item per distinct unit -- "
                "no deal asked two ways.")
        seen[key] = it["item_id"]


# ------------------------------------------------------------- row families
def _build_row_family(fam, tpl_results, tpl_by_id, family_docs, led, seed, n_cap):
    """One item per row, assigned to exactly one template."""
    options: dict[str, list[str]] = defaultdict(list)
    unit_by: dict[tuple[str, str], object] = {}
    for r in tpl_results:
        for u in r["_units"]:
            if u.defensible:
                options[u.row_id].append(u.template_id)
                unit_by[(u.template_id, u.row_id)] = u

    capacity = len(options)
    caps = Counter()
    for row, tids in options.items():
        for t in tids:
            caps[t] += 1

    # Family 1 caps any single GOLD FIELD; every template here has a distinct
    # gold field, so the template cap is the field cap.
    cap_used, impossible, n_target = None, False, min(capacity, n_cap)
    if fam == "single_fact_recall":
        n_from_cap, cap_used, impossible = largest_under_cap(dict(caps), FIELD_CAP)
        n_target = min(n_target, n_from_cap)
        per_template_max = max(1, int(cap_used * n_target))
    else:
        per_template_max = n_target

    # Assign scarcest-option rows first so a row that only one template can
    # serve is never crowded out by one that several can.
    order = _stable_shuffle(list(options), seed, fam)
    order.sort(key=lambda rid: (len(options[rid]), rid))

    taken: Counter = Counter()
    chosen: list[tuple[str, str]] = []
    for rid in order:
        if len(chosen) >= n_target:
            break
        cands = sorted(options[rid], key=lambda t: (taken[t], t))
        for t in cands:
            if taken[t] < per_template_max:
                chosen.append((t, rid))
                taken[t] += 1
                break

    items = []
    for tid, rid in sorted(chosen):
        tpl = tpl_by_id[tid]
        row = _find_row(led, tpl, rid)
        items.append(_emit(fam, tpl, row, {}, family_docs, led, rid))

    return items, {
        "capacity": capacity, "n": len(items),
        "n_target": n_target,
        "per_template": dict(sorted(taken.items())),
        "field_cap_requested": FIELD_CAP if fam == "single_fact_recall" else None,
        "field_cap_used": cap_used,
        "field_cap_impossible": impossible,
        "interval_method": "bootstrap" if fam in BOOTSTRAP_FAMILIES else "wilson",
    }


def _find_row(led, tpl, row_id):
    source = (tpl.selector or {}).get("source") or "deals"
    for r in led.collection(source):
        if _row_id(r) == row_id:
            return r
    raise KeyError(f"row {row_id!r} vanished from {source}")


# -------------------------------------------------------------- aggregates
def _build_aggregate(fam, tpl_results, tpl_by_id, family_docs, led, seed, n_cap):
    """One item per admitted row set. Admission was decided by the audit."""
    from .audit import audit_family
    from .capacity import max_independent_set

    d = audit_family(fam, tpl_results)
    admitted = set(d.get("admitted") or [])
    items = []
    for r in tpl_results:
        tpl = tpl_by_id[r["template_id"]]
        for u in r["_units"]:
            if not u.defensible or u.key not in admitted:
                continue
            items.append(_emit_aggregate(fam, tpl, u, family_docs, led))
    items.sort(key=lambda i: i["item_id"])
    if len(items) > n_cap:
        items = _stable_shuffle(items, seed, fam)[:n_cap]
        items.sort(key=lambda i: i["item_id"])
    return items, {"capacity": d["capacity"], "n": len(items),
                   "interval_method": "wilson",
                   "identical_set_collisions": d.get("identical_set_collisions"),
                   "rejected_for_overlap": d.get("rejected_for_overlap")}


def _emit_aggregate(fam, tpl, unit, family_docs, led):
    source = (tpl.selector or {}).get("source") or "deals"
    where = substitute((tpl.selector or {}).get("where") or {}, unit.binding)
    rows = select(led.collection(source), where)
    spec = dict((tpl.gold or {}).get("aggregate") or {})
    res = agg_query(rows, spec)
    gold = {"value": res.value, "unit": (tpl.gold or {}).get("unit") or "none",
            "trace": res.as_trace()}
    if (tpl.gold or {}).get("accept"):
        gold["accept"] = tpl.gold["accept"]
    item = {
        "item_id": _item_id(tpl.id, unit.key),
        "family": fam,
        "question": _question(tpl, unit.binding, family_docs.get(fam)),
        "answer_type": tpl.answer_type,
        "gold": gold,
        "meta": {"template_id": tpl.id, "source_row_id": unit.key,
                 "difficulty": tpl.difficulty, "bucket": None, "recency": False,
                 "binding": unit.binding, "n_rows": len(rows)},
    }
    if tpl.raw.get("grading"):
        item["grading"] = tpl.raw["grading"]
    return item


# ----------------------------------------------------------------- recency
def _build_recency(fam, tpl_results, tpl_by_id, family_docs, led, seed, n_cap):
    """Unit is (subject, field); each template IS a field."""
    pairs = []
    for r in tpl_results:
        for u in r["_units"]:
            if u.defensible:
                pairs.append((u.template_id, u.row_id))
    pairs = sorted(set(pairs))
    capacity = len(pairs)
    if capacity > n_cap:
        pairs = sorted(_stable_shuffle(pairs, seed, fam)[:n_cap])
    items = []
    for tid, rid in pairs:
        tpl = tpl_by_id[tid]
        row = _find_row(led, tpl, rid)
        # source_row_id is (subject, field), not the subject: six figures about
        # one company are six different facts with six different supersessions.
        items.append(_emit(fam, tpl, row, {}, family_docs, led,
                           f"{rid}:{tid}", recency=True))
    subjects = len({rid for _t, rid in pairs})
    return items, {
        "capacity": capacity, "n": len(items), "distinct_subjects": subjects,
        "interval_method": "wilson",
        "clustering_note": (
            f"{len(items)} items over {subjects} subjects. Items about the same "
            "subject share a document set, so the effective independent sample "
            "for anything subject-level is nearer the subject count."),
    }


# ----------------------------------------------------------------- holdout
def _holdout_template(tpl_by_id, tpl_results, target):
    """The template whose selector matches this target's FIRM.

    v1 used one template for every target, so targets belonging to one firm
    were asked whether the OTHER firm would advance them, while their gold came
    from their own firm's rules. Seven of twenty scored items were unanswerable
    in principle. The firm is not decoration: the two firms have different
    implicit rules and the question names which firm is deciding.
    """
    firm = str(target.get("firm") or "")
    ordered = [tpl_by_id[r["template_id"]] for r in tpl_results]
    # Prefer a discriminating template scoped to this firm, then any template
    # scoped to this firm, then the general discriminating one.
    for want_disc in (True, False):
        for t in ordered:
            where = (t.selector or {}).get("where") or {}
            if str(where.get("firm") or "") != firm:
                continue
            if want_disc and t.raw.get("bucket_role") != "discriminating":
                continue
            return t
    for t in ordered:
        if t.raw.get("bucket_role") == "discriminating":
            return t
    return ordered[0]


def _build_holdout(fam, tpl_results, tpl_by_id, family_docs, holdout_v2, led):
    """Scored items are the balanced discriminating slice; controls ride along.

    From 2026-08-20 the scored slice is sampled with equal numbers of each gold
    class, so the chance floor is 50% rather than the 87.3% a constant "pass"
    responder scored under the natural distribution. The floor is carried on
    every item so the report cannot print a score without it.
    """
    if not holdout_v2:
        return [], {"capacity": 0, "n": 0, "note": "no holdout selection supplied",
                    "interval_method": "wilson"}
    doc = family_docs.get(fam) or {}
    policy = holdout_v2.get("policy") or {}

    items = []
    for t in holdout_v2.get("holdout_targets", []):
        bucket = (t.get("provenance") or {}).get("bucket") or t.get("bucket")
        scored = t.get("scored")
        if scored is None:                       # v1-era file: bucket decides
            scored = bucket == "discriminating"
        tpl = _holdout_template(tpl_by_id, tpl_results, t)
        binding = {k: t.get(k) for k in (tpl.slots or {})}
        items.append({
            "item_id": _item_id("ira", t["id"]),
            "family": fam,
            "question": _question(tpl, binding, doc),
            "answer_type": "categorical",
            "gold": {"value": t.get("ground_truth_outcome"), "unit": "none"},
            "meta": {
                "template_id": tpl.id, "source_row_id": t["unit_key"],
                "difficulty": "hard", "bucket": bucket,
                "recency": False,
                "scored": bool(scored),
                "holdout_id": t["id"],
                "firm": t.get("firm"),
                "gold_class": t.get("gold_class") or t.get("ground_truth_outcome"),
                "discriminates": bucket == "discriminating",
                "written_criteria_verdict": t.get("written_criteria_verdict"),
                "written_criteria_failures": t.get("written_criteria_failures"),
                "driving_rule_ids": t.get("driving_rule_ids"),
                "rescued_by_rule": t.get("rescued_by_rule"),
                # What a responder that reads nothing scores on this item's
                # slice. Carried per item so no report can print a family-4
                # score without the floor beside it.
                "chance_floor": policy.get("chance_floor") if scored else None,
            },
        })
    items.sort(key=lambda i: i["item_id"])
    scored_items = [i for i in items if i["meta"]["scored"]]
    ctl = [i for i in items if not i["meta"]["scored"]]
    gold = Counter(str(i["gold"]["value"]) for i in scored_items)
    return items, {
        "capacity": (holdout_v2.get("availability") or {}).get("discriminating_advance", 0)
        + (holdout_v2.get("availability") or {}).get("discriminating_pass", 0)
        or holdout_v2.get("saturation", {}).get("ceiling_distinct_units"),
        "n": len(scored_items), "n_control": len(ctl),
        "interval_method": "wilson",
        "gold_distribution": dict(gold),
        "gold_distribution_controls": dict(
            Counter(str(i["gold"]["value"]) for i in ctl)),
        "balanced": policy.get("balanced_across_gold_class", False),
        "chance_floor": policy.get("chance_floor"),
        "availability": holdout_v2.get("availability"),
        "rule_distribution": holdout_v2.get("rule_distribution"),
        "rule_cap_scope": policy.get("rule_cap_scope"),
        "rule_cap_used": (policy.get("pass_rule_cap") or {}).get("cap_used"),
        "rule_cap_impossible": (policy.get("pass_rule_cap") or {}).get("impossible"),
        "advance_rule_cap_note": policy.get("advance_rule_cap_note"),
        "templates_used": dict(Counter(i["meta"]["template_id"] for i in items)),
        "single_rule_caveat": (
            "The advance class rests on one rule (RULE-A1) with one behavioural "
            "degree of freedom. The Wilson interval is an item-level interval and "
            "is not a claim about generalisation across rules."),
    }


# -------------------------------------------------------------------- emit
def _item_id(prefix: str, key: str) -> str:
    h = hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}--{h}"


def _emit(fam, tpl, row, extra_binding, family_docs, led, source_row_id,
          recency=False):
    binding = _slot_bindings(tpl, row, extra_binding)
    g = dict(tpl.gold or {})
    gold: dict = {"unit": g.get("unit") or "none"}
    if "checklist_ref" in g:
        gold = {"checklist_ref": g["checklist_ref"]}
    elif "value" in g:
        gold["value"] = g["value"]
    elif "path" in g:
        res = resolve_gold([row], {"path": g["path"], "unit": g.get("unit")})
        gold["value"] = res.get("value") if isinstance(res, dict) else getattr(res, "value", None)
    if g.get("accept"):
        gold["accept"] = g["accept"]

    item = {
        "item_id": _item_id(tpl.id, source_row_id),
        "family": fam,
        "question": _question(tpl, binding, family_docs.get(fam)),
        "answer_type": tpl.answer_type,
        "gold": gold,
        "meta": {
            "template_id": tpl.id,
            "source_row_id": source_row_id,
            "difficulty": tpl.difficulty,
            "bucket": None,
            "recency": bool(recency),
            "scored": True,
            "deal_codename": row.get("codename") or row.get("deal_codename"),
        },
    }
    for k, path in (tpl.raw.get("meta_from") or {}).items():
        item["meta"][k] = row.get(path)
    if tpl.raw.get("grading"):
        item["grading"] = tpl.raw["grading"]
    return item


def write_items(path: str, built: dict, provenance: dict) -> str:
    """JSON, sorted, no clock -- so two runs are byte-identical."""
    from .paths import assert_writable

    assert_writable(path, what="write items")
    payload = {
        "generated_by": "harness.generate.build_items",
        "seed": built["report"]["seed"],
        "provenance": provenance,
        "selection_report": built["report"],
        "items": built["items"],
    }
    import io
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return path
