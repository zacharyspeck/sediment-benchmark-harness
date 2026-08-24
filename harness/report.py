"""Turn a run's grades.jsonl into a report that is hard to misread.

DESIGN RULES
------------
1. No score appears without its interval.
2. When two arms' intervals overlap, the line says "no significant difference".
   It never says which arm is higher as though that meant something.
3. The width of the interval is stated in the report itself, next to a table of
   what widths to expect at various n, so nobody has to trust an assertion
   about noise.
4. Harness errors are separated from model failures. An item the grader crashed
   on is not evidence about a model, and is excluded from the denominator and
   listed by name. An item the model failed to answer IS a model failure and
   scores zero.
5. Every extraction_uncertain and every fabrication flag is printed with its
   trace. If a number is wrong, the reader can find out why without a debugger.
"""

from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from typing import Any, Iterable

from .paths import assert_writable
from .stats import (
    ArmComparison,
    MeanStat,
    ProportionStat,
    compare_arms,
    detectable_difference,
    halfwidth_table,
    n_to_separate,
    sample_size_for_difference,
)

__all__ = ["load_grades", "aggregate", "render_markdown", "write_report",
           "summary_payload", "write_summary_json"]

FAMILY_ORDER = [
    "single_fact_recall",
    "multi_doc_aggregation",
    "corpus_wide_statistic",
    "implicit_rule_application",
    "convention_conformance",
    "absence_and_abstention",
]

#: Slices reported on their own line and never folded into a family score.
NON_FAMILY_SLICES = {"recency"}

FAMILY_BLURB = {
    "single_fact_recall": "one document holds the answer; a tie is the expected result",
    "multi_doc_aggregation": "needs more documents than one search returns",
    "corpus_wide_statistic": "needs the whole corpus",
    "implicit_rule_application": "behaviour that appears in no document",
    "convention_conformance": "deterministic checklist; fractional score, bootstrap CI",
    "absence_and_abstention": "thin record; the wrong answer is a confident invention",
    "recency": "REPORTED SEPARATELY, not one of the six: which stated value is still true",
}


def load_grades(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with io.open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"_unparseable_line": n, "verdict": "error",
                            "error": "unparseable jsonl line"})
    return out


def _is_error(g: dict) -> bool:
    return g.get("verdict") == "error"


def aggregate(grades: list[dict], conf: float = 0.95, bootstrap_seed: int = 20260818,
              baseline_arm: str | None = None) -> dict:
    """Group grades by family and arm and attach the right kind of interval."""
    arms = sorted({g.get("arm", "?") for g in grades if not g.get("_unparseable_line")})
    fams = [f for f in FAMILY_ORDER if any(g.get("family") == f for g in grades)]
    fams += sorted({g.get("family") for g in grades
                    if g.get("family") and g.get("family") not in FAMILY_ORDER})

    cells: dict[tuple[str, str], dict] = {}
    for fam in fams:
        for arm in arms:
            # Items the item file marks unscored are EXCLUDED from the family
            # score. Family 4's controls are the only ones today: they are
            # generated, graded and reported on their own line, and the
            # pre-registration has always said they do not enter the score. Until
            # 2026-08-20 they did -- v1's family-4 number was 26 items where the
            # policy said 20. The default is True, so no other family moves.
            rows = [g for g in grades
                    if g.get("family") == fam and g.get("arm") == arm
                    and g.get("scored", True) is not False]
            errs = [g for g in rows if _is_error(g)]
            scored = [g for g in rows if not _is_error(g)]
            binary = all(bool(g.get("binary", True)) for g in scored) if scored else True
            scores = [float(g.get("score") or 0.0) for g in scored]

            if not scored:
                stat = None
            elif binary:
                stat = ProportionStat(k=sum(scores), n=len(scores), conf=conf, label=f"{fam}/{arm}")
            else:
                stat = MeanStat(values=scores, conf=conf, seed=bootstrap_seed,
                                label=f"{fam}/{arm}")

            uncertain = [g for g in scored if g.get("extraction_uncertain")]
            fab_a = [g for g in scored
                     if ((g.get("fabrication") or {}).get("tier_a_count") or 0) > 0]
            fab_b_total = sum((g.get("fabrication") or {}).get("tier_b_count") or 0
                              for g in scored)

            cells[(fam, arm)] = {
                "family": fam, "arm": arm,
                "n_total": len(rows), "n_scored": len(scored), "n_errors": len(errs),
                "binary": binary,
                "stat": stat,
                "point": stat.point if stat else None,
                "scores": scores,
                "errors": errs,
                "uncertain": uncertain,
                "uncertain_rate": (len(uncertain) / len(scored)) if scored else 0.0,
                "fab_a_items": fab_a,
                "fab_a_rate": (len(fab_a) / len(scored)) if scored else 0.0,
                "fab_b_total": fab_b_total,
                "rows": rows,
            }

    # Comparisons are always made against a baseline arm, so an ablation run
    # (same model, cartridge on/off, search on/off) reads the same way a
    # two-arm run does. With exactly two arms this is the ordinary pairing.
    baseline = baseline_arm if baseline_arm in arms else (arms[0] if arms else None)
    comparisons: dict[str, ArmComparison] = {}
    pairwise: dict[str, list[ArmComparison]] = {}
    if baseline and len(arms) >= 2:
        for fam in fams:
            sb = cells[(fam, baseline)]["stat"]
            if sb is None:
                continue
            row = []
            for other in arms:
                if other == baseline:
                    continue
                so = cells[(fam, other)]["stat"]
                if so is None:
                    continue
                row.append(compare_arms(so, sb, other, baseline))
            if row:
                pairwise[fam] = row
                comparisons[fam] = row[0]  # the headline pairing

    return {"arms": arms, "baseline": baseline, "families": fams, "cells": cells,
            "comparisons": comparisons, "pairwise": pairwise, "conf": conf}


# --------------------------------------------------------------------------
def _pct(x) -> str:
    return "n/a" if x is None else f"{100*float(x):.1f}%"


def _fence(text: str, limit: int = 700) -> str:
    t = (text or "").strip()
    if len(t) > limit:
        t = t[:limit] + f"\n... [truncated, {len(text)} chars total]"
    return t.replace("`", "'")


def _rule_application_breakout(grades: list[dict], arms: list[str]) -> list[str]:
    """implicit_rule_application, split by whether the item discriminates."""
    out: list[str] = []
    rows = [g for g in grades
            if g.get("family") == "implicit_rule_application" and not _is_error(g)]
    if not rows:
        return out

    out.append("### `implicit_rule_application`, broken out by slice\n")
    # PRE-REGISTRATION amendment 2026-08-20, section 2a: this appears ABOVE the
    # table wherever a family-4 score is printed, and never in a footnote.
    out.append(
        "> **The advance class rests on a single rule with one behavioural degree of "
        "freedom.** `RULE-A1` is the only rule in this corpus that produces the "
        "advance class; every advance item is that one rule applied to a different "
        "deal. The Wilson interval below is an "
        "**item-level** interval -- it describes how precisely this one rule's "
        "application was measured. It is **not** a claim about generalisation across "
        "rules. A score here licenses \"the model applied this one unwritten "
        "behaviour\", not \"the model learned the firm's unwritten behaviour.\"\n"
    )
    _floors = {g.get("chance_floor") for g in rows if g.get("chance_floor") is not None}
    if len(_floors) == 1:
        _f = next(iter(_floors))
        out.append(
            f"> **Chance floor {100 * _f:.0f}%.** The scored slice is sampled balanced "
            f"across gold class, so a responder that reads nothing scores "
            f"{100 * _f:.0f}% -- measured, not asserted: `constant_pass` and "
            f"`constant_advance` both land exactly there. Read every score in this "
            f"table against {100 * _f:.0f}%, not against zero.\n"
        )
    out.append(
        "Roughly half the holdout set is *clean* -- the written criteria and the firm's "
        "actual behaviour agree, so applying the published screens gets them right. Only "
        "the *discriminating* items separate a system that absorbed the behaviour from one "
        "that read the criteria. A blended score hides exactly that.\n"
    )
    out.append("| arm | slice | n | score | 95% CI | answered as written criteria |")
    out.append("|---|---|---:|---:|---|---:|")
    for arm in arms:
        arows = [g for g in rows if g.get("arm") == arm]
        for label, sel in (
            ("discriminating", lambda g: (g.get("detail") or {}).get("discriminates") is True),
            ("clean", lambda g: (g.get("detail") or {}).get("discriminates") is not True),
            ("all", lambda g: True),
        ):
            sub = [g for g in arows if sel(g)]
            if not sub:
                continue
            k = sum(float(g.get("score") or 0) for g in sub)
            st = ProportionStat(k=k, n=len(sub))
            matched_written = sum(
                1 for g in sub if (g.get("detail") or {}).get("matched_written_criteria") is True
            )
            out.append(
                f"| {arm} | {label} | {len(sub)} | {_pct(st.point)} | {st.ci.fmt()} | "
                f"{matched_written}/{len(sub)} |"
            )
    out.append("")
    out.append(
        "> **How to read the last column.** An arm whose answers track the written criteria "
        "on discriminating items has learned the documents, not the behaviour. That is the "
        "control this family exists to provide: the written-criteria baseline scores 100% on "
        "clean items and 0% on discriminating ones.\n"
    )
    return out


def _abstention_breakout(grades: list[dict], arms: list[str]) -> list[str]:
    out: list[str] = []
    rows = [g for g in grades
            if g.get("family") == "absence_and_abstention" and not _is_error(g)]
    if not rows:
        return out
    out.append("### Abstention family, by outcome\n")
    out.append(
        "A refusal and an invention are both wrong, and a buyer reads them very "
        "differently. Only `correct_abstention` scores.\n"
    )
    out.append("| arm | correct_abstention | bare_refusal | invented_thesis | other |")
    out.append("|---|---:|---:|---:|---:|")
    for arm in arms:
        arows = [g for g in rows if g.get("arm") == arm]
        c = defaultdict(int)
        for g in arows:
            c[(g.get("detail") or {}).get("outcome", "other")] += 1
        out.append(
            f"| {arm} | {c['correct_abstention']} | {c['bare_refusal']} | "
            f"{c['invented_thesis']} | {c['incorrect_other'] + c['other']} |"
        )
    out.append("")
    return out


# --------------------------------------------------------------------------
def render_markdown(agg: dict, grades: list[dict], run_meta: dict | None = None,
                    config: dict | None = None, title: str = "Benchmark run report") -> str:
    cfg = dict(config or {})
    meta = dict(run_meta or {})
    arms = agg["arms"]
    fams = agg["families"]
    cells = agg["cells"]
    comps = agg["comparisons"]
    unc_threshold = float((cfg.get("reporting") or {}).get("extraction_uncertain_threshold", 0.05))

    L: list[str] = []
    L.append(f"# {title}\n")

    # ---------------- executive summary ----------------
    # Deliberately first and deliberately blunt: the single most likely way to
    # misuse this report is to read a family table and quote the higher number.
    n_grades = len([g for g in grades if not g.get("_unparseable_line")])
    n_err = len([g for g in grades if _is_error(g)])
    sig = [f for f, c in comps.items() if c.verdict == "significant"]
    nosig = [f for f, c in comps.items() if c.verdict != "significant"]
    L.append("## Summary\n")
    L.append(
        f"{len(fams)} families, {len(arms)} arms, {n_grades} gradings, "
        f"**{n_err} harness error(s)**. Baseline arm: `{agg.get('baseline')}`.\n"
    )
    if comps:
        if sig:
            L.append(f"- **Separated at this sample size:** "
                     + ", ".join(f"`{f}`" for f in sig))
        if nosig:
            L.append(f"- **No significant difference:** "
                     + ", ".join(f"`{f}`" for f in nosig))
        L.append("")
        L.append(
            "> A family in the second list is **not** a tie. It means this run cannot "
            "tell the arms apart at this n. The table below says how many items each "
            "one would need before its observed gap could be called.\n"
        )
        rows = []
        for fam in fams:
            c = comps.get(fam)
            if not c or c.verdict == "significant":
                continue
            pa, pb = c.point_a, c.point_b
            n_now = max((cells[(fam, a)]["n_scored"] for a in arms), default=0)
            rows.append((fam, pa, pb, n_now,
                         n_to_separate(pa, pb, agg.get("conf", 0.95)),
                         sample_size_for_difference(pa, pb) if pa != pb else None))
        if rows:
            L.append("| family | observed gap | n now | n to separate "
                     "(this report's rule) | n at 80% power (classical) |")
            L.append("|---|---:|---:|---:|---:|")
            for fam, pa, pb, n_now, need_overlap, need_power in rows:
                gap = abs(100 * (pa - pb))
                a_s = "not at any n" if need_overlap is None else f"{need_overlap:,}"
                p_s = "n/a" if need_power is None else f"{need_power:,}"
                L.append(f"| `{fam}` | {gap:.1f} pts ({100*pa:.0f}% vs {100*pb:.0f}%) "
                         f"| {n_now} | {a_s} | {p_s} |")
            L.append("")
            L.append(
                "> The two columns answer different questions and the first is the one "
                "that binds. **n to separate** is the smallest n at which the Wilson "
                "intervals stop overlapping -- the rule this report actually applies. "
                "**n at 80% power** is the textbook two-proportion calculation, shown "
                "because an analyst will expect it; it rests on a normal approximation "
                "that is unreliable near 0% and 100% and it is a less conservative test, "
                "so it can read lower than the n this report would need.\n")
    L.append("")

    # ---------------- provenance ----------------
    L.append("## Run provenance\n")
    L.append("| field | value |")
    L.append("|---|---|")
    for k in ("run_started", "harness_git_sha", "python", "platform", "yaml_parser",
              "n_items", "n_arms", "n_planned", "n_resumed", "zero_model_calls"):
        if k in meta:
            L.append(f"| {k} | `{meta[k]}` |")
    led = meta.get("ledger") or {}
    if led:
        L.append(f"| ledger schema | `{led.get('schema')}` |")
        L.append(f"| ledger source | `{led.get('source')}` |")
        counts = led.get("counts") or {}
        L.append(f"| ledger deals / holdout targets | `{counts.get('deals')} / "
                 f"{counts.get('holdout_targets')}` |")
    fa = meta.get("fabrication_allowlist") or {}
    if fa:
        L.append(f"| fabrication allowlist | `{fa.get('entities')} entities, "
                 f"{fa.get('entity_tokens')} tokens, {fa.get('stoplist_terms')} stoplist terms` |")
    L.append("")
    if led.get("warnings"):
        L.append("**Ledger warnings**\n")
        for w in led["warnings"]:
            L.append(f"- {w}")
        L.append("")

    L.append("**Arms**\n")
    L.append("| arm | type | model | cartridge | tools | runnable | notes |")
    L.append("|---|---|---|---|---|---|---|")
    for name, a in (meta.get("arms") or {}).items():
        L.append(
            f"| `{name}` | {a.get('type')} | {a.get('model') or '-'} | "
            f"{a.get('cartridge')} | {', '.join(a.get('tools') or []) or '-'} | "
            f"{a.get('runnable')} | {a.get('notes','')} |"
        )
    L.append("")

    # ---------------- how to read ----------------
    L.append("## Read this before reading any number\n")
    hw = halfwidth_table()
    row25 = next((r for r in hw if r["n"] == 25), {})
    L.append(
        f"At **n=25 per family** a 95% Wilson interval is about "
        f"**+/-{row25.get('p=50%', '?')} points** near 50% and "
        f"**+/-{row25.get('p=90%', '?')} points** near 90%. "
        f"To detect a 10-point difference at 80% power you need about "
        f"**{sample_size_for_difference(0.70, 0.60)} items per arm**; a 5-point difference "
        f"needs about **{sample_size_for_difference(0.65, 0.60)}**. "
        f"At n=25 the smallest gap that is detectable at all is roughly "
        f"**{detectable_difference(25)} points**.\n"
    )
    L.append(
        "So: a family-level gap smaller than roughly 30 points at n=25 is not a result. "
        "This report prints `no significant difference` whenever the two intervals overlap "
        "and does not name a winner.\n"
    )
    L.append("**Wilson half-width (percentage points), by n and true rate**\n")
    L.append("| n | p=50% | p=70% | p=90% |")
    L.append("|---:|---:|---:|---:|")
    for r in hw:
        L.append(f"| {r['n']} | +/-{r['p=50%']} | +/-{r['p=70%']} | +/-{r['p=90%']} |")
    L.append("")
    L.append(
        "> **Two conventions worth knowing.** (1) The overlap rule is *conservative*: "
        "non-overlapping intervals always mean a real difference, but overlapping intervals "
        "do not prove there is none. Where the paired-difference interval (Newcombe) excludes "
        "zero while the intervals still overlap, the report says `suggestive, underpowered` "
        "rather than picking a side. (2) Wilson applies to a proportion. "
        "`convention_conformance` scores a *fraction of a checklist* per item, which is not a "
        "Bernoulli trial, so it gets a seeded percentile bootstrap interval instead. Every "
        "table below names the method it used.\n"
    )

    # ---------------- family scores ----------------
    L.append("## Family scores\n")
    header = ("| family | n scored | chance floor | "
              + " | ".join(f"{a}" for a in arms) + " | verdict |")
    sep = "|---|---:|---:|" + "|".join(["---"] * len(arms)) + "|---|"
    L.append(header)
    L.append(sep)
    min_n = int(((cfg or {}).get("reporting") or {}).get("min_items_for_claim", 25))
    for fam in fams:
        ns = max((cells[(fam, a)]["n_scored"] for a in arms), default=0)
        vals = []
        for a in arms:
            c = cells[(fam, a)]
            st = c["stat"]
            vals.append(f"{_pct(c['point'])} {st.ci.fmt()}" if st else "n/a")
        comp = comps.get(fam)
        verdict = comp.headline if comp else ("-" if len(arms) != 2 else "n/a")
        # The chance floor: what a responder that reads nothing scores on this
        # family. Recorded per item by the generator, so it is a property of the
        # sample rather than a constant asserted here.
        floors = {g.get("chance_floor") for g in grades
                  if g.get("family") == fam and g.get("chance_floor") is not None}
        floor = f"{100 * next(iter(floors)):.0f}%" if len(floors) == 1 else "--"
        note = ""
        if fam in NON_FAMILY_SLICES:
            note = " *(separate slice)*"
        elif 0 < ns < min_n:
            note = f" *(n below min_items_for_claim = {min_n})*"
        L.append(f"| `{fam}`{note} | {ns} | {floor} | " + " | ".join(vals)
                 + f" | **{verdict}** |")
    L.append("")
    thin = [f for f in fams if f not in NON_FAMILY_SLICES
            and 0 < max((cells[(f, a)]["n_scored"] for a in arms), default=0) < min_n]
    if thin:
        L.append(
            f"> **{len(thin)} famil{'y' if len(thin) == 1 else 'ies'} below the "
            f"configured minimum of {min_n} items for a claim: "
            + ", ".join(f"`{f}`" for f in thin) + ".** The interval is printed and is "
            "wide; that is the honest report of a small measurement, not a reason to "
            "pad it. Read those rows as directional at best."
        )
        L.append("")
    L.append("Six families of equal standing, reported side by side and **never "
             "averaged**. Any row marked *(separate slice)* is not one of the six.")
    L.append("")
    for fam in fams:
        L.append(f"- `{fam}` -- {FAMILY_BLURB.get(fam, '')}")
    L.append("")

    degenerate = [
        (fam, arm) for fam in fams for arm in arms
        if cells[(fam, arm)]["stat"] is not None
        and "DEGENERATE" in cells[(fam, arm)]["stat"].ci.method
    ]
    if degenerate:
        L.append("> :warning: **Zero-width intervals below are not certainty.** "
                 + ", ".join(f"`{f}`/{a}" for f, a in degenerate)
                 + " scored identically on every item, so the percentile bootstrap "
                   "collapses to a point. That says the sample showed no variation, not "
                   "that the true rate is known exactly -- with this many items the real "
                   "uncertainty is close to the Wilson width in the table above.\n")

    if comps:
        L.append("**Comparison detail**\n")
        for fam, c in comps.items():
            L.append(f"- **`{fam}`** -- {c.caveat}")
            if c.diff_ci:
                L.append(f"  - paired difference ({c.arm_a} - {c.arm_b}): "
                         f"{c.diff_ci.fmt()} ({c.diff_ci.method})")
        L.append("")

    # ---------------- every pairing, when there are ablation arms ----------
    pairwise = agg.get("pairwise") or {}
    if len(arms) > 2 and pairwise:
        L.append("### Every arm against the baseline\n")
        L.append(
            f"Baseline is `{agg.get('baseline')}`. Ablation arms (cartridge on/off, "
            f"search on/off) read exactly like a second system, because they are one "
            f"-- an arm is a config entry.\n"
        )
        L.append("| family | arm | score | 95% CI | vs baseline |")
        L.append("|---|---|---:|---|---|")
        for fam in fams:
            for c in pairwise.get(fam, []):
                L.append(f"| `{fam}` | {c.arm_a} | {_pct(c.point_a)} | {c.ci_a.fmt()} "
                         f"| {c.headline} |")
        L.append("")

    # ---------------- difficulty ----------------
    diffs = sorted({g.get("difficulty") for g in grades
                    if g.get("difficulty") and not _is_error(g)},
                   key=lambda d: {"easy": 0, "medium": 1, "hard": 2}.get(str(d), 9))
    if len(diffs) > 1:
        L.append("### By difficulty tier\n")
        L.append(
            "Tiers come from the template that produced each item. A gap that lives "
            "entirely in the hard tier is a different claim from one spread evenly.\n"
        )
        L.append(
            "Broken out per family. These are NOT pooled across families: the six "
            "have different graders, different item counts and heavily overlapping "
            "source rows, so a pooled proportion has no valid interval and a pooled "
            "point estimate is dominated by whichever family happens to be largest.\n"
        )
        L.append("| family | tier | " + " | ".join(arms) + " |")
        L.append("|---|---|" + "|".join(["---:"] * len(arms)) + "|")
        for fam in fams:
            for tier in diffs:
                vals = []
                any_rows = False
                for arm in arms:
                    rows = [g for g in grades if g.get("difficulty") == tier
                            and g.get("arm") == arm and g.get("family") == fam
                            and not _is_error(g)]
                    if not rows:
                        vals.append("-")
                        continue
                    any_rows = True
                    k = sum(float(g.get("score") or 0) for g in rows)
                    binary = all(bool(g.get("binary", True)) for g in rows)
                    if binary:
                        st = ProportionStat(k=k, n=len(rows))
                        vals.append(f"{_pct(st.point)} {st.ci.fmt()} (n={len(rows)})")
                    else:
                        vals.append(f"{100*k/len(rows):.1f}% (n={len(rows)}, mean)")
                if any_rows:
                    L.append(f"| `{fam}` | {tier} | " + " | ".join(vals) + " |")
        L.append("")

    # ---------------- breakouts ----------------
    L.extend(_rule_application_breakout(grades, arms))
    L.extend(_abstention_breakout(grades, arms))

    # ---------------- conformance detail ----------------
    conf_rows = [g for g in grades
                 if g.get("family") == "convention_conformance" and not _is_error(g)]
    if conf_rows:
        L.append("### Convention conformance, by check\n")
        L.append(
            "Fractional scores, so the interval above is a bootstrap. This table shows "
            "which checklist items actually fail -- a 0.80 conformance score means very "
            "different things depending on which 20% is missing.\n"
        )
        by_check: dict[tuple[str, str], list[int]] = defaultdict(list)
        for g in conf_rows:
            for chk in (g.get("detail") or {}).get("checks") or []:
                if chk.get("passed") is None:
                    continue
                by_check[(g.get("arm"), chk.get("id"))].append(1 if chk.get("passed") else 0)
        L.append("| check | " + " | ".join(f"{a} pass rate" for a in arms) + " |")
        L.append("|---|" + "|".join(["---:"] * len(arms)) + "|")
        check_ids = sorted({cid for (_, cid) in by_check})
        for cid in check_ids:
            cellvals = []
            for a in arms:
                v = by_check.get((a, cid))
                cellvals.append(f"{sum(v)}/{len(v)}" if v else "-")
            L.append(f"| `{cid}` | " + " | ".join(cellvals) + " |")
        L.append("")

    # ---------------- extraction health ----------------
    L.append("## Extraction health\n")
    L.append(
        f"`extraction_uncertain` means the ANSWER line was missing or unusable and the "
        f"harness fell back to reading the prose. Threshold: **{100*unc_threshold:.0f}%**. "
        f"Above it, the fix is a stronger format instruction in the prompt, not a cleverer "
        f"fallback -- a smarter fallback converts a formatting failure into a correctness "
        f"score.\n"
    )
    L.append("| family | " + " | ".join(f"{a}" for a in arms) + " |")
    L.append("|---|" + "|".join(["---:"] * len(arms)) + "|")
    worst = 0.0
    for fam in fams:
        vals = []
        for a in arms:
            c = cells[(fam, a)]
            r = c["uncertain_rate"]
            worst = max(worst, r)
            mark = " :warning:" if r > unc_threshold else ""
            vals.append(f"{len(c['uncertain'])}/{c['n_scored']} ({_pct(r)}){mark}")
        L.append(f"| `{fam}` | " + " | ".join(vals) + " |")
    L.append("")
    # Reported per (family, arm) and never pooled. A pooled rate mixes six
    # graders and dilutes a formatting failure in one arm with the other arm's
    # clean output, which is how a real problem gets averaged into silence.
    breaches = []
    for fam in fams:
        for a in arms:
            c = cells[(fam, a)]
            if c["n_scored"] and c["uncertain_rate"] > unc_threshold:
                breaches.append((fam, a, c["uncertain_rate"], len(c["uncertain"]), c["n_scored"]))
    if breaches:
        L.append(
            f"**Uncertain-extraction rate is ABOVE the {100*unc_threshold:.0f}% threshold "
            f"in {len(breaches)} family/arm cell(s)** -- strengthen the answer-format "
            f"instruction before trusting these scores:"
        )
        L.append("")
        for fam, a, rate, k, n in breaches:
            L.append(f"- `{fam}` / `{a}`: {_pct(rate)} ({k}/{n})")
    else:
        L.append(
            f"Every family/arm cell is within the {100*unc_threshold:.0f}% "
            f"uncertain-extraction threshold."
        )
    L.append("")

    # ---------------- fabrication ----------------
    L.append("## Fabrication\n")
    L.append(
        "Independent of correctness. **Tier A** is the headline: a deal codename, person, "
        "firm or portfolio company that does not exist in the ledger. **Tier B** is advisory: "
        "a dollar or multiple figure matching no ledger value. Tier B is deliberately not in "
        "the headline, because a correctly *computed* statistic -- an average margin, a median "
        "hold period -- is by construction a number that appears nowhere in the ledger. "
        "Folding the two together would be the easiest way to produce an impressive and "
        "dishonest chart.\n"
    )
    L.append("| family | " + " | ".join(f"{a} Tier A" for a in arms)
             + " | " + " | ".join(f"{a} Tier B" for a in arms) + " |")
    L.append("|---|" + "|".join(["---:"] * (2 * len(arms))) + "|")
    for fam in fams:
        a_vals, b_vals = [], []
        for a in arms:
            c = cells[(fam, a)]
            a_vals.append(f"{len(c['fab_a_items'])}/{c['n_scored']} ({_pct(c['fab_a_rate'])})")
            b_vals.append(f"{c['fab_b_total']} flags")
        L.append(f"| `{fam}` | " + " | ".join(a_vals) + " | " + " | ".join(b_vals) + " |")
    L.append("")
    # Per family, with no cross-family total. A pooled Tier A rate carried a
    # Wilson interval over items that share most of their source rows -- Wilson
    # assumes independent trials, and these are not independent. The per-family
    # rates above are the reportable numbers.
    L.append("Tier A rate per family and arm. There is deliberately no benchmark-wide "
             "total: items across families are drawn from overlapping source rows, so "
             "a pooled proportion would carry an interval it has not earned.")
    L.append("")
    for a in arms:
        worst = [(f, len(cells[(f, a)]["fab_a_items"]), cells[(f, a)]["n_scored"])
                 for f in fams if cells[(f, a)]["n_scored"]]
        worst = [w for w in worst if w[1]]
        if worst:
            worst.sort(key=lambda w: -(w[1] / w[2]))
            top = ", ".join(f"`{f}` {k}/{n}" for f, k, n in worst[:3])
            L.append(f"- **{a}** flagged Tier A entity fabrications in "
                     f"{len(worst)} of {len(fams)} families; highest: {top}")
        else:
            L.append(f"- **{a}** no Tier A entity fabrications in any family")
    L.append("")

    # ---------------- flag log ----------------
    L.append("## Flag log (every flag, with trace)\n")
    unc_rows = [g for g in grades if g.get("extraction_uncertain") and not _is_error(g)]
    L.append(f"### Uncertain extractions ({len(unc_rows)})\n")
    if not unc_rows:
        L.append("_None._\n")
    for g in unc_rows:
        exd = g.get("extraction") or {}
        L.append(f"- **`{g.get('item_id')}`** / `{g.get('arm')}` / `{g.get('family')}` -- "
                 f"method `{exd.get('method')}`, flags `{exd.get('flags')}`")
        L.append(f"  - extractor said: {exd.get('notes')}")
        L.append(f"  - value used: `{exd.get('value')}` | gold: `{g.get('gold_answer')}` | "
                 f"scored: {g.get('score')}")
        if exd.get("candidates"):
            L.append(f"  - other candidates seen: `{exd.get('candidates')}`")
    L.append("")

    fab_rows = [g for g in grades
                if ((g.get("fabrication") or {}).get("tier_a_count") or 0) > 0]
    L.append(f"### Tier A fabrication flags ({sum((g.get('fabrication') or {}).get('tier_a_count', 0) for g in fab_rows)} "
             f"across {len(fab_rows)} responses)\n")
    if not fab_rows:
        L.append("_None._\n")
    for g in fab_rows:
        L.append(f"- **`{g.get('item_id')}`** / `{g.get('arm')}` / `{g.get('family')}`")
        for f in (g.get("fabrication") or {}).get("flags", []):
            if f.get("tier") != "A":
                continue
            L.append(f"  - `{f.get('kind')}` **{f.get('surface')}** -- {f.get('reason')}")
            L.append(f"    - sentence: _{_fence(f.get('sentence'), 300)}_")
    L.append("")

    tierb_rows = [g for g in grades
                  if ((g.get("fabrication") or {}).get("tier_b_count") or 0) > 0]
    L.append(f"### Tier B ungrounded figures, advisory ({len(tierb_rows)} responses)\n")
    if not tierb_rows:
        L.append("_None._\n")
    for g in tierb_rows[:40]:
        flags = [f for f in (g.get("fabrication") or {}).get("flags", []) if f.get("tier") == "B"]
        surfaces = ", ".join(f"`{f.get('surface')}`(~{f.get('nearest_known')})" for f in flags[:8])
        L.append(f"- `{g.get('item_id')}` / `{g.get('arm')}`: {surfaces}")
    L.append("")

    # ---------------- errors ----------------
    errs = [g for g in grades if _is_error(g)]
    L.append(f"## Harness errors ({len(errs)})\n")
    L.append(
        "These are harness failures, not model failures. They are excluded from every "
        "denominator above and listed here by name. A clean run has zero.\n"
    )
    if not errs:
        L.append("_None._\n")
    for g in errs:
        L.append(f"- `{g.get('key') or g.get('item_id')}` / `{g.get('family')}` -- "
                 f"{(g.get('comparison') or '')}")
        L.append(f"  ```\n  {_fence(g.get('error'), 900)}\n  ```")
    L.append("")

    # ---------------- per item ----------------
    L.append("## Per-item results\n")
    L.append("| item | family | arm | score | verdict | extracted | gold | flags |")
    L.append("|---|---|---|---:|---|---|---|---|")
    for g in sorted(grades, key=lambda x: (x.get("family") or "", x.get("item_id") or "",
                                           x.get("arm") or "")):
        if g.get("_unparseable_line"):
            continue
        flags = ", ".join(g.get("flags") or []) or "-"
        L.append(
            f"| `{g.get('item_id')}` | {g.get('family')} | {g.get('arm')} | "
            f"{g.get('score')} | {g.get('verdict')} | "
            f"`{str(g.get('extracted_answer'))[:40]}` | `{str(g.get('gold_answer'))[:40]}` | "
            f"{flags} |"
        )
    L.append("")

    L.append("## Comparison traces\n")
    L.append("_What each grader actually compared, item by item._\n")
    for g in sorted(grades, key=lambda x: (x.get("family") or "", x.get("item_id") or "")):
        if g.get("_unparseable_line"):
            continue
        L.append(f"- `{g.get('item_id')}` / `{g.get('arm')}`: {g.get('comparison')}")
    L.append("")
    return "\n".join(L)


def write_report(path: str, text: str) -> str:
    assert_writable(path, what="write report")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def summary_payload(agg: dict, grades: list[dict], run_meta: dict | None = None) -> dict:
    """Machine-readable run summary.

    Emitted next to the markdown so a run can be charted or diffed against a
    previous run without re-parsing prose. Every number here is the same number
    the report prints -- both come from the same aggregation.
    """
    arms, fams, cells = agg["arms"], agg["families"], agg["cells"]
    out: dict = {
        "arms": arms,
        "baseline": agg.get("baseline"),
        "families": fams,
        "confidence": agg.get("conf"),
        "n_gradings": len([g for g in grades if not g.get("_unparseable_line")]),
        "n_harness_errors": len([g for g in grades if _is_error(g)]),
        "run": {k: (run_meta or {}).get(k) for k in
                ("run_started", "harness_git_sha", "yaml_parser", "zero_model_calls")},
        "cells": {},
        "comparisons": {},
    }
    for fam in fams:
        for arm in arms:
            c = cells[(fam, arm)]
            st = c["stat"]
            out["cells"][f"{fam}::{arm}"] = {
                "family": fam, "arm": arm,
                "n_scored": c["n_scored"], "n_errors": c["n_errors"],
                "binary": c["binary"],
                "point": c["point"],
                "ci": st.ci.to_dict() if st else None,
                "uncertain_rate": c["uncertain_rate"],
                "fabrication_tier_a_rate": c["fab_a_rate"],
                "fabrication_tier_b_flags": c["fab_b_total"],
            }
    for fam, comp in (agg.get("comparisons") or {}).items():
        out["comparisons"][fam] = comp.to_dict()
    return out


def write_summary_json(path: str, agg: dict, grades: list[dict],
                       run_meta: dict | None = None) -> str:
    assert_writable(path, what="write summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload(agg, grades, run_meta), fh, indent=2, default=str)
    return path
