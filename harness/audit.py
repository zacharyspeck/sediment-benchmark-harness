"""The capacity audit: templates + ledger + corpus -> CAPACITY.md, capacity.json.

Deterministic and clock-free by construction, so the artifact is byte-identical
across runs and a disputed number can be re-derived rather than argued about.
Provenance (ledger shard hashes, corpus tree digest) is embedded in both files.

Read `harness/capacity.py` first -- it holds the definitions. This module is the
driver: it walks the templates, applies the gates, resolves each family's unit
of independence, and renders.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from .capacity import (
    AGGREGATE_FAMILIES,
    HOLDOUT_FAMILY,
    RECENCY_FAMILY,
    JACCARD_MAX,
    OVERLAP_MAX,
    UnitRecord,
    expand_bindings,
    halfwidth_at,
    jaccard,
    max_independent_set,
    overlap_coefficient,
    overlap_histogram,
    holdout_unit_key,
    printed_fields,
    slot_cardinality,
    top_overlapping,
    unit_passes_gates,
    load_lever_fields,
)
from .query import QueryError, resolve_gold, select
from .templates import substitute

# Families scored as a checklist fraction rather than a binary outcome.
BOOTSTRAP_FAMILIES = {"convention_conformance"}

N_CAP = 150  # pre-registered ceiling: n_family = min(capacity, 150)


def _largest_under_cap(caps: list[int], cap: float) -> dict:
    """Largest item count where no rule exceeds `cap` of the total.

    Returns the achievable n at the requested cap, and -- when the requested cap
    is arithmetically impossible because too few rules have capacity -- the
    smallest cap that IS achievable, which the pre-registration says to use and
    print instead.
    """
    if not caps:
        return {"requested_cap": cap, "achievable_n": 0, "cap_used": cap,
                "impossible": True, "reason": "no rule has capacity"}
    total = sum(caps)
    best = 0
    for n in range(1, total + 1):
        allowed = int(cap * n)          # floor: a rule may supply at most this
        if sum(min(c, allowed) for c in caps) >= n:
            best = n
    if best > 0:
        return {"requested_cap": cap, "achievable_n": best, "cap_used": cap,
                "impossible": False}
    # Too few rules: R rules at cap each can only cover R*cap of the total, so
    # the cap must be at least 1/R.
    smallest = 1.0 / len(caps)
    best2 = 0
    for n in range(1, total + 1):
        allowed = int(smallest * n) or 1
        if sum(min(c, allowed) for c in caps) >= n:
            best2 = n
    return {"requested_cap": cap, "achievable_n": best2, "cap_used": round(smallest, 4),
            "impossible": True,
            "reason": f"only {len(caps)} rules have capacity; {len(caps)} x {cap} = "
                      f"{len(caps) * cap:.2f} < 1.00, so no set can satisfy it"}


def _row_id(row: dict) -> str:
    for k in ("id", "codename", "name"):
        v = row.get(k)
        if v:
            return str(v)
    return "<no-id>"


def _entity_key(row: dict) -> str:
    """Cross-collection identity, for overlap only. Falls back to the row id."""
    ek = (row.get("_derived") or {}).get("entity_key")
    return str(ek) if ek else _row_id(row)


def _gold_for(tpl, row, ledger):
    """Best-effort gold for a single row, for the grounding check."""
    g = tpl.gold or {}
    if "path" in g:
        # resolve_gold returns an AggResult for aggregate specs and a plain
        # dict for path specs; normalise both rather than assuming one.
        try:
            r = resolve_gold([row], g)
        except QueryError:
            return None, "", None
        val = r.get("value") if isinstance(r, dict) else getattr(r, "value", None)
        return val, str(g.get("unit") or ""), g.get("accept")
    if "value" in g:
        return g.get("value"), str(g.get("unit") or ""), g.get("accept")
    return None, "", None


# ------------------------------------------------------------------ per-file
def audit_template(tpl, ledger, index, family_doc: dict | None = None,
                   screen=None) -> dict:
    """Measure one template through all four stages."""
    fam = tpl.family
    fdoc = family_doc or {}
    lever_map = load_lever_fields(fdoc)
    printed = printed_fields(fdoc.get("fact_block") or "")
    source = (tpl.selector or {}).get("source") or "deals"
    try:
        rows_all = ledger.collection(source)
    except Exception as exc:                                  # unknown source
        return {"template_id": tpl.id, "family": fam, "error": str(exc)}

    card, card_expr = slot_cardinality(tpl)
    bindings = expand_bindings(tpl, ledger)

    units: list[UnitRecord] = []
    raw_rows = 0
    empty_bindings = 0

    for b in bindings:
        where = substitute((tpl.selector or {}).get("where") or {}, b)
        try:
            rows = select(rows_all, where)
        except QueryError as exc:
            return {"template_id": tpl.id, "family": fam, "error": f"selector: {exc}"}
        raw_rows += len(rows)
        if not rows:
            empty_bindings += 1
            continue

        if fam in AGGREGATE_FAMILIES:
            ids = frozenset(_entity_key(r) for r in rows)
            # An aggregate is answerable only if EVERY contributing row is
            # retrievable. One missing row makes the correct total unobtainable
            # in principle, not merely hard.
            # Honour the template's codename_field: portfolio companies name
            # their documents through `deal_codename`, not `codename`, and
            # defaulting silently reported all eleven as ungrounded.
            subj = ((tpl.raw.get("gates") or {}).get("grounding") or {}).get(
                "codename_field") or "codename"
            missing = [r for r in rows if not index.has_doc(r.get(subj))]
            label = tpl.id + ("[" + ", ".join(f"{k}={v}" for k, v in sorted(b.items())) + "]" if b else "")
            units.append(UnitRecord(
                key=label, template_id=tpl.id, family=fam, binding=dict(b),
                row_set=ids, grounded=not missing,
                reason="" if not missing else f"{len(missing)} of {len(rows)} rows have no document",
            ))
        elif fam == HOLDOUT_FAMILY:
            for r in rows:
                key, why = holdout_unit_key(r, lever_map, printed)
                # Holdout targets appear in ZERO documents by design -- that is
                # the point of a holdout. So "grounded" here does not mean a
                # document answers it; it means the item is DECIDABLE from what
                # it prints. The corpus check runs in the other direction: a
                # target that leaked into a document is disqualified.
                #
                # The full collision screen runs here, not just the id test.
                # Until 2026-08-20 the audit checked only whether the target's
                # id appeared in a document while the SELECTOR additionally
                # rejected targets whose revenue and EBITDA figures co-occur in
                # one document. Capacity therefore reported 78 admissible
                # advance units where the selector could find 43, which is the
                # "two numbers in two artifacts" failure this module's own
                # comments warn about. One gate, applied in both places.
                hits = screen(r) if screen else []
                leaked = bool(hits) or (
                    index.contains_anywhere(str(r.get("id"))) if r.get("id") else False)
                units.append(UnitRecord(
                    key=key or f"<inadmissible:{_row_id(r)}>",
                    template_id=tpl.id, family=fam, row_id=_row_id(r),
                    binding=dict(b), grounded=bool(key) and not leaked,
                    contradicted=leaked,
                    reason=why or (hits[0] if hits else
                                   "target id appears in the corpus" if leaked else ""),
                    gold=r.get("ground_truth_outcome"),
                ))
        else:
            for r in rows:
                gv, unit, accept = _gold_for(tpl, r, ledger)
                grounded, contra, reason = unit_passes_gates(
                    tpl, r, index, gold_value=gv, unit=unit, accept=accept)
                units.append(UnitRecord(
                    key=_row_id(r), template_id=tpl.id, family=fam,
                    row_id=_row_id(r), codename=r.get("codename"), binding=dict(b),
                    grounded=grounded, contradicted=contra, reason=reason, gold=gv,
                ))

    if fam in AGGREGATE_FAMILIES:
        distinct = len({u.row_set for u in units})
    elif fam == HOLDOUT_FAMILY:
        distinct = len({u.key for u in units if u.grounded})
    else:
        distinct = len({u.row_id for u in units})

    grounded = [u for u in units if u.grounded]
    defensible = [u for u in units if u.defensible]
    reasons = Counter(u.reason for u in units if u.reason)

    ceiling = "slot cardinality" if card < raw_rows and fam in AGGREGATE_FAMILIES else "row count"
    if fam in AGGREGATE_FAMILIES:
        ceiling = "slot cardinality" if card <= len(units) else "non-empty bindings"

    return {
        "template_id": tpl.id,
        "family": fam,
        "source": source,
        "slot_cardinality": card,
        "slot_cardinality_expr": card_expr,
        "bindings_declared": len(bindings),
        "bindings_empty": empty_bindings,
        "raw_rows": raw_rows,
        "units_before_dedup": len(units),
        "units_distinct": distinct,
        "units_grounded": (len({u.row_set for u in grounded}) if fam in AGGREGATE_FAMILIES
                           else len({u.key for u in grounded}) if fam == HOLDOUT_FAMILY
                           else len({u.row_id for u in grounded})),
        "units_defensible": (len({u.row_set for u in defensible}) if fam in AGGREGATE_FAMILIES
                             else len({u.key for u in defensible}) if fam == HOLDOUT_FAMILY
                             else len({u.row_id for u in defensible})),
        "real_ceiling": ceiling,
        "gate_failures": dict(reasons.most_common()),
        "_units": units,
    }


# ---------------------------------------------------------------- per-family
def audit_family(fam: str, tpl_results: list[dict]) -> dict:
    """Collapse a family's templates onto its unit of independence."""
    all_units: list[UnitRecord] = []
    for t in tpl_results:
        all_units.extend(t.get("_units") or [])

    raw_sum = sum(t.get("raw_rows", 0) for t in tpl_results)
    out: dict = {
        "family": fam,
        "templates": [t["template_id"] for t in tpl_results],
        "raw_sum_before_dedup": raw_sum,
        "interval_method": "bootstrap" if fam in BOOTSTRAP_FAMILIES else "wilson",
    }

    if fam in AGGREGATE_FAMILIES:
        # Unit = the set of rows aggregated. Dedupe identical sets first, then
        # admit under BOTH thresholds via an exact maximum independent set.
        seen: dict = {}
        for u in all_units:
            seen.setdefault(u.row_set, u)          # first label wins, stable
        distinct = list(seen.values())
        out["units_distinct"] = len(distinct)
        out["identical_set_collisions"] = len(all_units) - len(distinct)

        defensible = [u for u in distinct if u.defensible]
        out["units_grounded"] = len([u for u in distinct if u.grounded])
        out["units_defensible_before_overlap"] = len(defensible)

        labelled = [(u.key, u.row_set) for u in defensible]
        sets = [s for _l, s in labelled]
        # Per-family admission is computed here for reporting, but the BINDING
        # number is the joint one computed across both aggregate families in
        # run_audit(). A family-2 slice sitting entirely inside a family-3
        # population is not independent of it just because they live in
        # different files, and admitting both would let one item's answer hand
        # you most of the other's.
        keep = max_independent_set(sets)
        out["capacity_within_family"] = len(keep)
        out["capacity"] = len(keep)
        out["admitted"] = [labelled[i][0] for i in keep]
        out["rejected_for_overlap"] = len(defensible) - len(keep)
        out["_candidates"] = labelled
        out["jaccard_histogram"] = overlap_histogram(sets, jaccard)
        out["overlap_histogram"] = overlap_histogram(sets, overlap_coefficient)
        out["top_overlapping_pairs"] = top_overlapping(labelled, 10)
        out["strict_subset_pairs"] = sum(
            1 for i in range(len(sets)) for j in range(len(sets))
            if i != j and sets[i] < sets[j])
    elif fam == RECENCY_FAMILY:
        keyed = {(u.template_id, u.row_id): u for u in all_units}
        out["units_distinct"] = len(keyed)
        defensible = {k: u for k, u in keyed.items() if u.defensible}
        out["units_grounded"] = len([1 for u in keyed.values() if u.grounded])
        out["units_defensible"] = len(defensible)
        out["capacity"] = len(defensible)
        out["distinct_subjects"] = len({u.row_id for u in defensible.values()})
        out["by_field"] = dict(sorted(Counter(
            t for t, _r in defensible).items()))
        out["clustering_note"] = (
            f"{len(defensible)} units over {len({u.row_id for u in defensible.values()})} "
            "subjects -- units about the same subject share a document set, so the "
            "effective independent sample is closer to the subject count than to the "
            "unit count for anything subject-level."
        )
        out["gate_failures"] = dict(Counter(
            u.reason for u in keyed.values() if u.reason).most_common())
    elif fam == HOLDOUT_FAMILY:
        # Unit = (firing rule-set, lever bin). Several shipped targets collapse
        # onto one bin -- they differ only by a fraction of a point on the
        # deciding field, which no reader and no model can act on differently.
        admissible = [u for u in all_units if u.grounded]
        by_key: dict[str, list[UnitRecord]] = defaultdict(list)
        for u in admissible:
            by_key[u.key].append(u)
        scored_keys = {k for k in by_key if not k.startswith("control:")}
        control_keys = {k for k in by_key if k.startswith("control:")}
        out["units_distinct"] = len(by_key)
        out["targets_seen"] = len({u.row_id for u in all_units})
        out["targets_inadmissible"] = len({u.row_id for u in all_units if not u.grounded})
        out["near_duplicate_collapse"] = len({u.row_id for u in admissible}) - len(by_key)
        out["units_grounded"] = len(by_key)
        out["units_defensible"] = len(by_key)
        # Only the discriminating slice is SCORED. Controls are generated at 30%
        # of the scored count and reported on their own line (pre-registration).
        out["scored_capacity"] = len(scored_keys)
        out["control_capacity"] = len(control_keys)
        out["capacity"] = len(scored_keys)
        out["inadmissible_reasons"] = dict(Counter(
            u.reason for u in all_units if not u.grounded and u.reason).most_common())
        # Count DISTINCT UNITS, not (target, template) pairs -- a target
        # selected by two templates is one unit and must not be counted twice.
        first_gold = {}
        for u in admissible:
            first_gold.setdefault(u.key, str(u.gold))
        out["gold_distribution"] = dict(Counter(
            g for k, g in first_gold.items() if not k.startswith("control:")).most_common())
        out["gold_distribution_controls"] = dict(Counter(
            g for k, g in first_gold.items() if k.startswith("control:")).most_common())
        # Controls are generated at 30% of scored, per the pre-registration.
        out["controls_planned"] = min(len(control_keys), round(0.30 * len(scored_keys)))
        # Which rule drives each admitted unit -- the 35% cap needs this.
        rule_counts: Counter = Counter()
        for k in scored_keys:
            for part in k.split("|"):
                rule_counts[part.split("[")[0]] += 1
        out["rule_distribution"] = dict(rule_counts.most_common())
        # The pre-registered cap is 35% of scored items. With R rules of
        # capacity c_r, the largest set honouring a cap of `cap` is
        # min over feasible n of sum_r min(c_r, floor(cap*n)) >= n.
        out["rule_cap"] = _largest_under_cap(list(rule_counts.values()), 0.35)

        # --------------------------------------------- gold-class breakdown
        # Family 4 is sampled BALANCED across gold class from 2026-08-20, so
        # "capacity" is no longer one number: n is set by the SMALLER of the two
        # scored classes, and a table that prints only their sum hides the
        # binding constraint. The 35% rule cap is also scoped within the pass
        # class now, so its arithmetic has to be reported per class.
        by_class: dict[str, set] = defaultdict(set)
        for u in admissible:
            gold = str(u.gold)
            if u.key.startswith("control:"):
                by_class["control_" + gold].add(u.key)
            else:
                by_class["scored_" + gold].add(u.key)
        cls_out: dict = {}
        for cname, keys in sorted(by_class.items()):
            rc: Counter = Counter()
            for k in keys:
                for part in k.split("|"):
                    rc[part.split("[")[0]] += 1
            entry = {"units": len(keys),
                     "rule_distribution": dict(rc.most_common())}
            if cname.startswith("scored_"):
                entry["rule_cap"] = _largest_under_cap(list(rc.values()), 0.35)
            cls_out[cname] = entry
        out["by_gold_class"] = cls_out

        adv = len(by_class.get("scored_advance") or ())
        pas = len(by_class.get("scored_pass") or ())
        n_bal = min(adv, pas, 75)
        if not (adv or pas):
            binding = None
        elif 75 <= min(adv, pas):
            # Naming the smaller class here would be wrong: when both classes
            # exceed the ceiling, the ceiling is what sets n, not either class.
            binding = "the pre-registered per-class ceiling of 75"
        else:
            binding = "advance" if adv <= pas else "pass"
        out["balanced"] = {
            "scored_advance_units": adv,
            "scored_pass_units": pas,
            "binding_class": binding,
            "per_class_cap": 75,
            "n_per_class": n_bal,
            "n_scored": 2 * n_bal,
            "chance_floor": 0.5 if n_bal else None,
            "wilson_halfwidth_at_n_scored": halfwidth_at(2 * n_bal),
            "note": (
                "n_advance = n_pass = min(available_advance, available_pass, 75). "
                "Derived from measured availability; nothing is padded. The chance "
                "floor is 50% and is printed beside the score."),
        }
    else:
        # Unit = one row, used once in the family. A row qualifying under two
        # templates is ONE unit, not two.
        by_row: dict[str, list[UnitRecord]] = defaultdict(list)
        for u in all_units:
            by_row[u.row_id].append(u)
        out["units_distinct"] = len(by_row)
        out["cross_template_duplicates"] = len(all_units) - len(by_row)

        grounded_rows = {r for r, us in by_row.items() if any(u.grounded for u in us)}
        defensible_rows = {r for r, us in by_row.items() if any(u.defensible for u in us)}
        out["units_grounded"] = len(grounded_rows)
        out["units_defensible"] = len(defensible_rows)
        out["capacity"] = len(defensible_rows)

        # Which template each defensible row can serve -- the generator needs
        # this to spread items and to honour the family-1 field cap.
        avail: dict[str, int] = Counter()
        for r in defensible_rows:
            for u in by_row[r]:
                if u.defensible:
                    avail[u.template_id] += 1
        out["defensible_by_template"] = dict(sorted(avail.items()))
        out["rows_with_one_option"] = sum(
            1 for r in defensible_rows
            if len({u.template_id for u in by_row[r] if u.defensible}) == 1)

    cap = out["capacity"]
    out["n_planned"] = min(cap, N_CAP)
    out["wilson_halfwidth_at_capacity"] = halfwidth_at(cap)
    out["wilson_halfwidth_at_n_planned"] = halfwidth_at(out["n_planned"])
    return out


# ------------------------------------------------------------------- drivers
def run_audit(ledger, templates, index, provenance: dict,
              family_docs: dict | None = None, screen=None) -> dict:
    # One collision screen for the whole audit, built once. The holdout family
    # is the only one that uses it; building it lazily here rather than letting
    # each caller decide is what stops capacity and selection drifting apart.
    if screen is None and any(t.family == HOLDOUT_FAMILY for t in templates):
        from .holdout_pool import make_screen
        screen = make_screen(index)
    by_family: dict[str, list[dict]] = defaultdict(list)
    tpl_rows = []
    for tpl in templates:
        r = audit_template(tpl, ledger, index, (family_docs or {}).get(tpl.family),
                           screen=screen)
        tpl_rows.append(r)
        if "error" not in r:
            by_family[r["family"]].append(r)

    families = {}
    for fam, rows in sorted(by_family.items()):
        families[fam] = audit_family(fam, rows)

    # ------------------------------------------------ joint aggregate admission
    # The unit of independence for BOTH aggregate families is "the set of rows
    # aggregated", so independence has to be judged over the pooled candidate
    # set. Judged per family, a 13-row sector slice and the 68-row population
    # containing it both get admitted, and answering the slice hands you a fifth
    # of the population answer.
    agg_fams = [f for f in families if f in AGGREGATE_FAMILIES]
    if agg_fams:
        pooled: list[tuple[str, str, frozenset]] = []
        for f in agg_fams:
            for label, rs in families[f].pop("_candidates", []):
                pooled.append((f, label, rs))
        keep_idx = set(max_independent_set([rs for _f, _l, rs in pooled]))
        admitted_by_fam: dict[str, list[str]] = {f: [] for f in agg_fams}
        for i, (f, label, _rs) in enumerate(pooled):
            if i in keep_idx:
                admitted_by_fam[f].append(label)
        for f in agg_fams:
            d = families[f]
            d["admitted"] = sorted(admitted_by_fam[f])
            d["capacity"] = len(admitted_by_fam[f])
            d["rejected_for_overlap"] = d["units_defensible_before_overlap"] - d["capacity"]
            d["n_planned"] = min(d["capacity"], N_CAP)
            d["wilson_halfwidth_at_capacity"] = halfwidth_at(d["capacity"])
            d["wilson_halfwidth_at_n_planned"] = halfwidth_at(d["n_planned"])
            d["joint_admission_note"] = (
                "admitted jointly with " + ", ".join(sorted(x for x in agg_fams if x != f))
                + ": both families share the unit 'set of rows aggregated', so a slice "
                  "contained in another family's population is not independent of it."
            ) if len(agg_fams) > 1 else ""
        joint_summary = {
            "candidates": len(pooled),
            "admitted": len(keep_idx),
            "rejected": len(pooled) - len(keep_idx),
            "families": sorted(agg_fams),
        }
    else:
        joint_summary = {}

    return {
        "joint_aggregate_admission": joint_summary,
        "provenance": provenance,
        "policy": {
            "n_family": "min(measured defensible capacity, 150)",
            "floor": None,
            "jaccard_max": JACCARD_MAX,
            "overlap_coefficient_max": OVERLAP_MAX,
        },
        "corpus": index.summary(),
        "templates": [{k: v for k, v in r.items() if not k.startswith("_")} for r in tpl_rows],
        "families": {f: {k: v for k, v in d.items() if not k.startswith("_")}
                     for f, d in families.items()},
        "_family_objects": families,
        "_template_objects": tpl_rows,
    }


def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def render_capacity_md(audit: dict) -> str:
    from .baseline import provenance_lines

    L: list[str] = []
    A = L.append
    A("# Capacity audit")
    A("")
    A("Measured, not estimated. Every number here was computed by the harness's own")
    A("selector and aggregation code against the ledger and the document tree named")
    A("below. Reproduce with:")
    A("")
    A("```")
    A("python -m harness.cli capacity --md-out CAPACITY.md --json-out capacity.json")
    A("```")
    A("")
    A("The artifact carries no clock and no RNG, so a re-run is byte-identical unless")
    A("the inputs changed -- which is what the hashes are for.")
    A("")
    A("## Provenance")
    A("")
    for ln in provenance_lines(audit["provenance"]):
        A(ln)
    A("")

    c = audit["corpus"]
    A(f"Corpus index: **{_fmt(c['documents'])} documents**, {_fmt(c['titled'])} with a parseable "
      f"subject, {_fmt(c['distinct_subjects'])} distinct subjects, {_fmt(c['distinct_doctypes'])} "
      f"document types, {_fmt(c['distinct_dates'])} distinct dates.")
    A("")

    A("## How to read this")
    A("")
    A("Four stages, each strictly smaller than the last:")
    A("")
    A("| stage | meaning |")
    A("|---|---|")
    A("| **raw** | rows the selector matches, summed over every declared slot binding |")
    A("| **distinct** | after collapsing to the family's unit of independence |")
    A("| **grounded** | after dropping units no corpus document can answer |")
    A("| **defensible** | after dropping units a corpus document contradicts |")
    A("")
    A("Only **defensible** sets n. Per the pre-registration, "
      "`n_family = min(defensible capacity, 150)`, with no floor.")
    A("")

    # ------------------------------------------------------------- families
    A("## Family capacity")
    A("")
    A("| family | raw sum | distinct | grounded | defensible | n planned | 95% half-width at n |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for fam, d in audit["families"].items():
        gd = d.get("units_grounded", "-")
        df = d.get("units_defensible", d.get("units_defensible_before_overlap", "-"))
        if fam == HOLDOUT_FAMILY:
            # `defensible` here would count controls, which are never scored.
            df = f"{d['scored_capacity']} scored + {d['control_capacity']} control"
        method = "" if d["interval_method"] == "wilson" else " (bootstrap)"
        A(f"| `{fam}` | {_fmt(d['raw_sum_before_dedup'])} | {_fmt(d['units_distinct'])} | "
          f"{_fmt(gd)} | {_fmt(df)} | **{_fmt(d['n_planned'])}** | "
          f"+/-{d['wilson_halfwidth_at_n_planned']}{method} |")
    A("")
    A("`convention_conformance` scores a fraction of a checklist rather than a binary")
    A("outcome, so its interval is a seeded percentile bootstrap, not Wilson. Wilson is")
    A("an interval for a binomial proportion and a fractional score is not a Bernoulli")
    A("trial; applying it would pool correlated checks within one document as if they")
    A("were independent trials and print an interval that is too narrow. The half-width")
    A("shown for that family is the Wilson figure for scale only and is marked.")
    A("")

    # ------------------------------------------------------------- templates
    A("## Per template")
    A("")
    A("| template | source | slot cardinality | real ceiling | raw rows | distinct | grounded | defensible |")
    A("|---|---|---|---|---:|---:|---:|---:|")
    for t in audit["templates"]:
        if "error" in t:
            A(f"| `{t['template_id']}` | -- | -- | **ERROR** | -- | -- | -- | {t['error']} |")
            continue
        A(f"| `{t['template_id']}` | {t['source']} | {t['slot_cardinality_expr']} | "
          f"{t['real_ceiling']} | {_fmt(t['raw_rows'])} | {_fmt(t['units_distinct'])} | "
          f"{_fmt(t['units_grounded'])} | {_fmt(t['units_defensible'])} |")
    A("")

    A("### Why units were dropped")
    A("")
    any_gate = False
    for t in audit["templates"]:
        if t.get("gate_failures"):
            any_gate = True
            A(f"**`{t['template_id']}`**")
            for reason, n in t["gate_failures"].items():
                A(f"- {_fmt(n)} -- {reason}")
            A("")
    if not any_gate:
        A("No unit failed a gate. That is unusual enough to be worth checking rather")
        A("than celebrating: confirm the gates are actually configured on the templates.")
        A("")

    # ------------------------------------------------------------- aggregates
    for fam, d in audit["families"].items():
        if fam not in AGGREGATE_FAMILIES:
            continue
        A(f"## `{fam}` -- overlap")
        A("")
        A(f"Unit is the SET of rows aggregated. {_fmt(d['units_distinct'])} distinct sets "
          f"after collapsing {_fmt(d['identical_set_collisions'])} identical ones "
          f"(different questions over the same rows are ONE unit, not two).")
        A("")
        A(f"Admission requires **both** Jaccard <= {JACCARD_MAX} and overlap coefficient "
          f"<= {OVERLAP_MAX}. Capacity is the exact maximum independent set: "
          f"**{_fmt(d['capacity'])}** admitted, {_fmt(d['rejected_for_overlap'])} rejected "
          f"for overlap.")
        A("")
        A(f"Strict-subset pairs among the candidates: **{_fmt(d['strict_subset_pairs'])}**. "
          "Jaccard alone does not see these; the overlap coefficient does, which is why "
          "both thresholds are applied.")
        A("")
        for title, key in (("Jaccard", "jaccard_histogram"), ("Overlap coefficient", "overlap_histogram")):
            h = d[key]
            A(f"**{title}** over {_fmt(h['pairs'])} pairs")
            A("")
            A("| bucket | pairs |")
            A("|---|---:|")
            for b, n in h["buckets"].items():
                A(f"| {b} | {_fmt(n)} |")
            A("")
        A("**Ten most-overlapping pairs**")
        A("")
        A("| a | b | n(a) | n(b) | Jaccard | overlap coef | strict subset |")
        A("|---|---|---:|---:|---:|---:|---|")
        for r in d["top_overlapping_pairs"]:
            A(f"| `{r['a']}` | `{r['b']}` | {_fmt(r['n_a'])} | {_fmt(r['n_b'])} | "
              f"{r['jaccard']} | {r['overlap_coef']} | {'yes' if r['subset'] else 'no'} |")
        A("")

    # ------------------------------------------------------------ row families
    # ------------------------------------------------------------- holdout
    hd = audit["families"].get(HOLDOUT_FAMILY)
    if hd:
        A(f"## `{HOLDOUT_FAMILY}` -- admissibility and gold")
        A("")
        A(f"Unit is (firing rule-set, lever bin), admitted only when every firing")
        A(f"rule's deciding field is PRINTED in the fact block. "
          f"{_fmt(hd.get('targets_seen', 0))} targets seen, "
          f"**{_fmt(hd.get('targets_inadmissible', 0))} inadmissible**, "
          f"{_fmt(hd.get('near_duplicate_collapse', 0))} more collapsed onto an "
          f"already-occupied bin, leaving **{_fmt(hd['capacity'])}** distinct units.")
        A("")
        if hd.get("inadmissible_reasons"):
            A("**Why targets were inadmissible**")
            A("")
            A("| reason | targets |")
            A("|---|---:|")
            for reason, n in hd["inadmissible_reasons"].items():
                A(f"| {reason} | {_fmt(n)} |")
            A("")
        gold = hd.get("gold_distribution") or {}
        A("**Gold distribution over admitted units**")
        A("")
        A("| gold | units |")
        A("|---|---:|")
        for g, n in gold.items():
            A(f"| `{g}` | {_fmt(n)} |")
        A("")
        if len(gold) <= 1 and gold:
            only = next(iter(gold))
            A(f"> **EVERY ADMITTED UNIT HAS GOLD `{only}`.** A responder that ignores")
            A(f"> the facts entirely and always answers `{only}` scores 100% on this")
            A("> slice. Until the rule engine can return the other verdict on a")
            A("> discriminating case, this family measures nothing about learned")
            A("> behaviour and its score must not be read as if it does. The")
            A("> constant-answer baseline arm exists to make that visible in every")
            A("> report rather than only here.")
            A("")
        bal = hd.get("balanced") or {}
        if bal.get("n_per_class") is not None:
            A("### Balanced across gold class")
            A("")
            A("Family 4 is sampled with equal numbers of each gold class from")
            A("2026-08-20. Capacity is therefore not one number: **n is set by the")
            A("smaller of the two scored classes**, and a single total would hide")
            A("which one binds.")
            A("")
            A("| scored class | admissible units |")
            A("|---|---:|")
            A(f"| gold `advance` | {_fmt(bal['scored_advance_units'])} |")
            A(f"| gold `pass` | {_fmt(bal['scored_pass_units'])} |")
            A("")
            A(f"`n_advance = n_pass = min({bal['scored_advance_units']}, "
              f"{bal['scored_pass_units']}, {bal['per_class_cap']}) = "
              f"**{bal['n_per_class']}**`, so **n = {bal['n_scored']}** scored items, "
              f"binding on the **{bal['binding_class']}** class.")
            A("")
            A(f"Chance floor **50%**. Wilson half-width at n={bal['n_scored']}: "
              f"+/-{bal['wilson_halfwidth_at_n_scored']} points.")
            A("")
            A("> **The advance class rests on a single rule with one behavioural")
            A("> degree of freedom.** `RULE-A1` is the only rule in this corpus")
            A("> that produces the advance class. Every advance item is that one")
            A("> rule applied to a different deal. The interval below is an")
            A("> ITEM-LEVEL interval: it")
            A("> describes how precisely this rule's application was measured. It is")
            A("> **not** a claim about generalisation across rules, and a score here")
            A("> does not license the sentence \"the model learned the firm's")
            A("> unwritten behaviour\" -- only \"the model applied this one unwritten")
            A("> behaviour\". This is the same caveat the recency slice carries, for")
            A("> the same reason.")
            A("")

        for cname, entry in sorted((hd.get("by_gold_class") or {}).items()):
            total = max(1, entry["units"])
            cap = entry.get("rule_cap") or {}
            A(f"**Per-rule concentration -- `{cname}`** ({_fmt(entry['units'])} units)")
            A("")
            A("| rule | units | share |")
            A("|---|---:|---:|")
            for r, n in entry["rule_distribution"].items():
                A(f"| `{r}` | {_fmt(n)} | {100.0 * n / total:.1f}% |")
            A("")
            if cap:
                if len(entry["rule_distribution"]) == 1:
                    only = next(iter(entry["rule_distribution"]))
                    A(f"One rule (`{only}`) supplies 100% of this class. The 35% cap is")
                    A("not applied here and is not silently satisfied: the concentration")
                    A("is structural, not a sampling choice, and is disclosed.")
                else:
                    A(f"35% cap: largest honouring set is {_fmt(cap['achievable_n'])} "
                      f"units at cap {cap['cap_used']}"
                      + (f" -- **requested 0.35 impossible**: {cap.get('reason')}"
                         if cap.get("impossible") else ""))
                A("")

        if hd.get("rule_distribution"):
            total = max(1, hd["capacity"])
            A("**Per-rule concentration -- all scored units**")
            A("")
            A("| rule | units | share |")
            A("|---|---:|---:|")
            for r, n in hd["rule_distribution"].items():
                A(f"| `{r}` | {_fmt(n)} | {100.0 * n / total:.1f}% |")
            A("")

    # ------------------------------------------------------------- recency
    rd = audit["families"].get(RECENCY_FAMILY)
    if rd:
        A(f"## `{RECENCY_FAMILY}` -- reported separately, never folded into a family score")
        A("")
        if rd["capacity"] == 0:
            A("**Measured zero.** No fact in this corpus is stated with two different")
            A("values at two different dates where the later one supersedes the earlier.")
            A("The gate failures below are the proof; nothing was forced.")
            A("")
        else:
            A(f"**{_fmt(rd['capacity'])} units** over {_fmt(rd.get('distinct_subjects', 0))} subjects.")
            A("")
            A("A unit is admitted only when the figure is stated on two or more distinct")
            A("document dates, with two or more distinct values, the latest date states")
            A("exactly one value, and that value equals the ledger's current value. The")
            A("wrong answer is the stale figure -- stated confidently, in a real document,")
            A("by a named author, and true once.")
            A("")
            A(f"> {rd.get('clustering_note', '')}")
            A("")
            A("| field (template) | units |")
            A("|---|---:|")
            for t, n in (rd.get("by_field") or {}).items():
                A(f"| `{t}` | {_fmt(n)} |")
            A("")
        if rd.get("gate_failures"):
            A("**Why candidates were rejected**")
            A("")
            A("| reason | units |")
            A("|---|---:|")
            for reason, n in rd["gate_failures"].items():
                A(f"| {reason} | {_fmt(n)} |")
            A("")

    A("## Row-unit families -- template availability")
    A("")
    A("A row that qualifies under two templates is ONE unit. `rows with one option` is")
    A("the count that can only be served by a single template, which is what constrains")
    A("spreading items across templates and honouring the family-1 field cap.")
    A("")
    for fam, d in audit["families"].items():
        if fam in AGGREGATE_FAMILIES or fam in (HOLDOUT_FAMILY, RECENCY_FAMILY):
            continue
        A(f"**`{fam}`** -- {_fmt(d['units_distinct'])} distinct rows, "
          f"{_fmt(d['cross_template_duplicates'])} cross-template duplicates removed once, "
          f"{_fmt(d['rows_with_one_option'])} rows with a single template option")
        A("")
        A("| template | defensible rows it can serve |")
        A("|---|---:|")
        for t, n in (d.get("defensible_by_template") or {}).items():
            A(f"| `{t}` | {_fmt(n)} |")
        A("")

    return "\n".join(L) + "\n"
