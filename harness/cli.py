"""Command-line entry points.

    python -m harness.cli fixtures            # end-to-end fixture run + report
    python -m harness.cli run     --items ... --arms a,b --out ...
    python -m harness.cli report  --run results/<run_id>
    python -m harness.cli validate-ledger
    python -m harness.cli validate-templates  # resolves selectors; generates NOTHING
    python -m harness.cli allowlist           # fabrication allowlist summary
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from harness.arms import load_arms  # noqa: E402
from harness.ledger import LedgerUnavailable, load_ledger  # noqa: E402
from harness.paths import RefusedWrite, assert_writable  # noqa: E402
from harness.report import (  # noqa: E402
    aggregate,
    load_grades,
    render_markdown,
    write_report,
    write_summary_json,
)
from harness.runner import Runner, load_items  # noqa: E402
from harness.yamlio import load_file  # noqa: E402

CONFIG_DIR = os.path.join(REPO, "config")
ITEMS_DIR = os.path.join(REPO, "items")
RESULTS_DIR = os.path.join(REPO, "results")
DATA_DIR = os.path.join(REPO, "data")

#: The saturated v6 pool, copied from the corpus repo by
#: `tools/phase1_check_inputs.py`. This repo never regenerates it: the generator
#: needs `ledger-v4/` and the rendered corpus, and one population produced by two
#: implementations is two populations reported as one.
HOLDOUT_POOL = os.path.join(DATA_DIR, "holdout-pool-v6.json")

#: The SELECTED family-4 set, written by `holdout-pool` from the pool above.
#: v2 is the v1-era file and is preserved untouched.
HOLDOUT_SELECTED = os.path.join(DATA_DIR, "holdout-targets-v3.json")
HOLDOUT_SELECTED_V1 = os.path.join(DATA_DIR, "holdout-targets-v2.json")


def resolve_holdout_selected() -> str:
    """The item generator's holdout input: v3 if it exists, else the v1-era v2.

    One resolved constant rather than the same literal path repeated in three
    commands. Reporting capacity against one file while the generator draws from
    another puts two numbers in two artifacts and makes the smaller one look
    like the answer.
    """
    return (HOLDOUT_SELECTED if os.path.exists(HOLDOUT_SELECTED)
            else HOLDOUT_SELECTED_V1)


def load_pool_rows(family_doc: dict) -> tuple[list, dict]:
    """The v6 pool as ledger-shaped holdout rows, gated for capacity measurement.

    Two transformations, both of which the audit would otherwise get wrong:

    1. `rescued_by_rule` is folded into `driving_rule_ids`. Advance-class
       targets record their deciding rule under the former and the unit keyer
       reads only the latter. Without this every advance target is rejected as
       "no driving rule recorded" and the class measures zero.
    2. Discriminating targets falling inside an inferred boundary region are
       dropped. The advance-class rule's applicability boundaries are observed,
       not stated, so a target between the last observation on one side and the
       first on the other has gold no document evidences.
    """
    from harness.holdout_pool import boundary_gap, load_pool, normalise

    targets, sha = load_pool(HOLDOUT_POOL)
    rows, gaps = [], Counter()
    for t in targets:
        n = normalise(t)
        if str(n.get("bucket")) == "discriminating":
            g = boundary_gap(n.get("as_of"))
            if g:
                gaps[g] += 1
                continue
        rows.append(n)
    return rows, {"sha256": sha, "targets": len(targets), "admitted": len(rows),
                  "boundary_gap_drops": dict(gaps)}


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_grading_config() -> dict:
    """Grading config plus every checklist defined in the template files."""
    cfg = load_file(os.path.join(CONFIG_DIR, "grading.yaml")) or {}
    checklists: dict = {}
    cpath = os.path.join(ITEMS_DIR, "convention_conformance.yaml")
    if os.path.exists(cpath):
        doc = load_file(cpath) or {}
        for name, entry in (doc.get("checklists") or {}).items():
            checklists[name] = entry
    cfg["checklists"] = checklists
    return cfg


def resolve_corpus_dir() -> str | None:
    """Absolute path to the document tree, or None when unconfigured."""
    cfg = load_file(os.path.join(CONFIG_DIR, "ledger.yaml")) or {}
    raw = cfg.get("corpus_dir")
    if not raw:
        return None
    p = raw if os.path.isabs(raw) else os.path.join(REPO, raw)
    return os.path.abspath(p) if os.path.isdir(p) else None


def load_fabrication_config() -> dict:
    p = os.path.join(CONFIG_DIR, "fabrication.yaml")
    return (load_file(p) or {}) if os.path.exists(p) else {}


def resolve_ledger(explicit: str | None = None, quiet: bool = False):
    """Load the ledger named in config/ledger.yaml, with ordered fallbacks."""
    cfg = load_file(os.path.join(CONFIG_DIR, "ledger.yaml")) or {}
    candidates = []
    if explicit:
        candidates.append({"path": explicit, "schema": "auto", "label": "--ledger"})
    else:
        active = cfg.get("active")
        sources = cfg.get("sources") or {}
        if active and active in sources:
            s = dict(sources[active])
            s["label"] = active
            candidates.append(s)
        for name in (cfg.get("fallback_order") or []):
            if name in sources and name != active:
                s = dict(sources[name])
                s["label"] = name
                candidates.append(s)

    problems = []
    for cand in candidates:
        path = cand.get("path")
        if path and not os.path.isabs(path):
            path = os.path.join(REPO, path)
        try:
            led = load_ledger(path, cand.get("schema", "auto"),
                              cand.get("holdout_path"))
            if not quiet:
                print(f"ledger: loaded '{cand.get('label')}' schema={led.schema} "
                      f"deals={len(led.deals)} holdout={len(led.holdout_targets)} "
                      f"from {path}")
                for w in led.warnings:
                    print(f"  ledger warning: {w}")
            return led
        except LedgerUnavailable as exc:
            problems.append(f"  {cand.get('label')}: {exc}")
    raise LedgerUnavailable(
        "no ledger source could be loaded. Tried:\n" + "\n".join(problems)
    )


# --------------------------------------------------------------------------
def cmd_run(args) -> int:
    ledger = resolve_ledger(args.ledger)
    items = load_items(args.items)
    if args.family:
        wanted = set(args.family.split(","))
        items = [i for i in items if i.get("family") in wanted]
    arms = load_arms(args.arms_config, args.arms.split(",") if args.arms else None)

    out = args.out or os.path.join(RESULTS_DIR, f"run-{_stamp()}")
    runner = Runner(
        items=items, arms=arms, out_dir=out, ledger=ledger,
        config=load_grading_config(), fabrication_config=load_fabrication_config(),
        repo_root=REPO,
    )
    counts = runner.run(resume=not args.no_resume)
    print(json.dumps(counts, indent=2))
    if args.report:
        _render(out, args.report_out)
    return 1 if counts.get("errors") and args.strict else 0


def _render(run_dir: str, out_path: str | None = None) -> str:
    grades = load_grades(os.path.join(run_dir, "grades.jsonl"))
    meta = {}
    rj = os.path.join(run_dir, "run.json")
    if os.path.exists(rj):
        with io.open(rj, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    cfg = load_grading_config()
    rep_cfg = cfg.get("reporting") or {}
    agg = aggregate(grades,
                    conf=float(rep_cfg.get("confidence", 0.95)),
                    bootstrap_seed=int(rep_cfg.get("bootstrap_seed", 20260818)),
                    baseline_arm=rep_cfg.get("baseline_arm"))
    title = f"Benchmark run report -- {os.path.basename(run_dir)}"
    md = render_markdown(agg, grades, meta, cfg, title=title)
    path = out_path or os.path.join(run_dir, "report.md")
    write_report(path, md)
    # Machine-readable twin, always written next to the run so a result can be
    # charted or diffed without re-parsing the prose.
    sj = write_summary_json(os.path.join(run_dir, "summary.json"), agg, grades, meta)
    print(f"report written: {path}  ({len(grades)} grade records)")
    print(f"summary written: {sj}")
    return path


def cmd_report(args) -> int:
    _render(args.run, args.out)
    return 0


def cmd_fixtures(args) -> int:
    """The one command that proves the whole pipeline works end to end."""
    ledger = resolve_ledger(args.ledger or os.path.join(REPO, "fixtures", "mini_ledger.json"))
    items = load_items(os.path.join(REPO, "fixtures", "fixture_items.yaml"))
    arms = load_arms(os.path.join(CONFIG_DIR, "arms.yaml"),
                     ["fixture_strong", "fixture_weak"])
    out = args.out or os.path.join(RESULTS_DIR, f"fixtures-{_stamp()}")
    runner = Runner(items=items, arms=arms, out_dir=out, ledger=ledger,
                    config=load_grading_config(),
                    fabrication_config=load_fabrication_config(), repo_root=REPO)
    counts = runner.run(resume=not args.no_resume)
    print(json.dumps(counts, indent=2))
    path = _render(out, args.report_out)
    if args.copy_to:
        with io.open(path, "r", encoding="utf-8") as fh:
            body = fh.read()
        write_report(args.copy_to, body)
        print(f"report copied to: {args.copy_to}")
    return 1 if counts.get("errors") else 0


def cmd_validate_ledger(args) -> int:
    led = resolve_ledger(args.ledger)
    print(json.dumps(led.summary(), indent=2))
    return 0


def cmd_capacity(args) -> int:
    """Measure honest capacity per family and render CAPACITY.md + capacity.json.

    Generates no items. Reports what could honestly be generated, which is the
    number that sets n -- see PRE-REGISTRATION.md.
    """
    from harness.audit import render_capacity_md, run_audit
    from harness.baseline import baseline
    from harness.corpusindex import load_corpus_index
    from harness.templates import load_all_templates, load_template_file

    led = resolve_ledger(args.ledger)
    corpus = resolve_corpus_dir()
    if corpus is None:
        print("ERROR: corpus_dir is not configured or does not exist "
              "(config/ledger.yaml). The grounding and contradiction gates "
              "cannot run without the document tree, and a capacity number "
              "without them is only a ledger row count.", file=sys.stderr)
        return 1

    print(f"indexing {corpus} ...", file=sys.stderr)
    index = load_corpus_index(corpus, [d.get("codename") for d in led.deals])
    print(f"  {len(index.docs)} documents", file=sys.stderr)

    prov = baseline(led.source if os.path.isdir(str(led.source)) else
                    os.path.join(REPO, "..", "sediment-corpus", "ledger-v4"), corpus)
    templates = load_all_templates(ITEMS_DIR)
    # The per-family YAML document carries family-level blocks the templates
    # need but do not each repeat: the shared fact block and the lever map.
    family_docs = {}
    for fname in sorted(os.listdir(ITEMS_DIR)):
        if fname.endswith(".yaml"):
            fam, _t, doc = load_template_file(os.path.join(ITEMS_DIR, fname))
            family_docs[fam] = doc

    # Capacity for family 4 is measured against the FULL admissible pool -- what
    # could honestly be generated -- not against whatever was selected from it.
    # Measuring the selection would make capacity and n the same number by
    # construction and hide every unit the sampler left on the table.
    pool_meta = {}
    holdout_source = "ledger shard (no v6 pool present)"
    if os.path.exists(HOLDOUT_POOL):
        rows, pool_meta = load_pool_rows(family_docs.get("implicit_rule_application"))
        if rows:
            led.holdout_targets = rows
            holdout_source = (
                f"data/holdout-pool-v6.json ({pool_meta['targets']} targets, "
                f"{pool_meta['admitted']} after the boundary-gap gate, "
                f"sha256 {pool_meta['sha256'][:16]})")
    elif os.path.exists(HOLDOUT_SELECTED_V1):
        with io.open(HOLDOUT_SELECTED_V1, encoding="utf-8") as fh:
            hv2 = json.load(fh)
        rows = []
        for t in hv2.get("holdout_targets", []):
            r = dict(t)
            r["bucket"] = (t.get("provenance") or {}).get("bucket") or t.get("bucket")
            rows.append(r)
        if rows:
            led.holdout_targets = rows
            holdout_source = f"data/holdout-targets-v2.json ({len(rows)} targets)"
    print(f"holdout source: {holdout_source}", file=sys.stderr)

    audit = run_audit(led, templates, index, prov, family_docs)
    audit["holdout_source"] = holdout_source
    audit["holdout_pool"] = pool_meta

    md = render_capacity_md(audit)
    payload = {k: v for k, v in audit.items() if not k.startswith("_")}

    if args.md_out:
        assert_writable(args.md_out, what="write CAPACITY.md")
        with io.open(args.md_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md)
        print(f"wrote {args.md_out}", file=sys.stderr)
    if args.json_out:
        assert_writable(args.json_out, what="write capacity.json")
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        print(f"wrote {args.json_out}", file=sys.stderr)
    if not args.md_out and not args.json_out:
        print(md)

    for fam, d in payload["families"].items():
        print(f"{fam:28} capacity={d['capacity']:>6}  n={d['n_planned']:>4}  "
              f"+/-{d['wilson_halfwidth_at_n_planned']} pts", file=sys.stderr)
    return 0


def cmd_holdout_v2(args) -> int:
    """Regenerate the holdout set with the corpus's own generator."""
    from harness.baseline import baseline
    from harness.capacity import load_lever_fields, printed_fields
    from harness.corpusindex import load_corpus_index
    from harness.holdout import (
        GeneratorUnavailable, apply_rule_cap, balanced_controls,
        load_corpus_generator, saturate, saturate_controls,
    )
    from harness.templates import load_template_file

    led = resolve_ledger(args.ledger)
    corpus = resolve_corpus_dir()
    if corpus is None:
        print("ERROR: corpus_dir is not configured; the collision screen needs it.",
              file=sys.stderr)
        return 1
    corpus_repo = os.path.dirname(corpus)

    print(f"indexing {corpus} ...", file=sys.stderr)
    index = load_corpus_index(corpus, [d.get("codename") for d in led.deals])
    print(f"  {len(index.docs)} documents", file=sys.stderr)

    try:
        HO, B, _R = load_corpus_generator(corpus_repo)
    except GeneratorUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _fam, _tpls, doc = load_template_file(os.path.join(ITEMS_DIR,
                                                       "implicit_rule_application.yaml"))
    lever_map = load_lever_fields(doc)
    printed = printed_fields(doc.get("fact_block") or "")

    print("saturating the discriminating pool ...", file=sys.stderr)
    sat = saturate(HO, B, index, lever_map, printed)
    units = sat["units"]
    gold = Counter(str(t.get("ground_truth_outcome")) for t in units.values())

    capped = apply_rule_cap(units, 0.35)
    scored_keys = capped["selected"]
    want_controls = max(1, round(0.30 * len(scored_keys)))
    ctl = saturate_controls(HO, B, index, want_controls)
    control_keys = balanced_controls(ctl["units"], want_controls)

    gen_sha = _sha256(os.path.join(corpus_repo, "v4_holdout.py"))
    prov = baseline(os.path.join(corpus_repo, "ledger-v4"), corpus)

    targets = []
    for i, k in enumerate(scored_keys):
        t = dict(units[k])
        t["id"] = "holdout-v2-disc-%03d" % i
        t["unit_key"] = k
        t["provenance"] = {"generator": "v4_holdout.build", "generator_sha256": gen_sha,
                           "imported_read_only": True, "seed": prov["ledger"].get("index_seed"),
                           "bucket": "discriminating"}
        targets.append(t)
    for i, k in enumerate(control_keys):
        t = dict(ctl["units"][k])
        t["id"] = "holdout-v2-ctrl-%03d" % i
        t["unit_key"] = k
        t["provenance"] = {"generator": "v4_holdout.build", "generator_sha256": gen_sha,
                           "imported_read_only": True, "seed": prov["ledger"].get("index_seed"),
                           "bucket": t.get("bucket")}
        targets.append(t)

    payload = {
        "version": 2,
        "generated_against": {"generator_sha256": gen_sha, "provenance": prov},
        "policy": {"rule_cap_requested": 0.35, "rule_cap_used": capped["cap_used"],
                   "rule_cap_impossible": capped["impossible"],
                   "controls_at_pct_of_scored": 0.30},
        "saturation": {"rounds": sat["trace"], "saturated": sat["saturated"],
                       "total_generated": sat["total_generated"],
                       "ceiling_distinct_units": len(units)},
        "rejects": sat["rejected"],
        "gold_distribution": dict(gold),
        "rule_distribution": capped["per_rule"],
        "rule_capacity": capped.get("rule_capacity", {}),
        "n_scored": len(scored_keys), "n_control": len(control_keys),
        "holdout_targets": targets,
    }

    out = args.out or os.path.join(REPO, "data", "holdout-targets-v2.json")
    assert_writable(out, what="write holdout-targets-v2")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    total_rejects = sum(sat["rejected"].values())
    print("")
    print(f"CEILING (distinct admissible discriminating units): {len(units)}")
    print(f"  saturated: {sat['saturated']}  after {sat['total_generated']} generated records")
    print(f"  REJECTED: {total_rejects}")
    for reason, n in sat["rejected"].items():
        print(f"    {n:>7}  {reason}")
    print("")
    print(f"GOLD DISTRIBUTION across the ceiling: {dict(gold)}")
    if len(gold) <= 1:
        only = next(iter(gold), "?")
        print("")
        print("  *** EVERY DISCRIMINATING TARGET HAS GOLD '%s'. ***" % only)
        print("  A responder that ignores the facts and always answers '%s' scores" % only)
        print("  100% on this slice. The cause is structural, not a sampling accident:")
        print("  v4_holdout.py sets ground_truth_outcome to 'pass' whenever EITHER the")
        print("  written verdict or the implicit verdict is pass, and in this build of")
        print("  the engine no branch of implicit_verdict can produce an advance. A")
        print("  discriminating target is by definition one the written criteria")
        print("  advance and a rule declines, so its gold is 'pass' by construction.")
        print("  Until the rule engine gains a rule that can produce the advance")
        print("  class, this")
        print("  family's score cannot be read as evidence of learned behaviour, and")
        print("  the constant-answer baseline arm is reported beside it to make that")
        print("  visible in every run rather than only here.")
    print("")
    print(f"RULE DISTRIBUTION (cap {capped['cap_used']:.2f}"
          f"{' -- REQUESTED 0.35 IS IMPOSSIBLE: ' + capped['reason'] if capped['impossible'] else ''}):")
    for r, n in sorted(capped["per_rule"].items()):
        share = 100.0 * n / max(1, len(scored_keys))
        print(f"    {r:12} {n:>4}  {share:5.1f}%")
    print(f"  available per rule before the cap: {capped.get('rule_capacity')}")
    print("")
    print(f"scored: {len(scored_keys)}   controls: {len(control_keys)} "
          f"(30% of scored)")
    print(f"wrote {out}")
    return 0


def cmd_holdout_pool(args) -> int:
    """Select family 4's item set from the saturated v6 pool.

    Generates no targets. Reads `data/holdout-pool-v6.json`, applies every gate,
    and writes the balanced selection to `data/holdout-targets-v3.json` in the
    schema `harness.generate` already consumes.
    """
    from harness.baseline import baseline
    from harness.capacity import load_lever_fields, printed_fields
    from harness.corpusindex import load_corpus_index
    from harness.holdout_pool import (
        BALANCED_CAP, BOUNDARY_GAPS, N_CONTROL_PER_CLASS, RULE_CAP,
        admit, load_pool, make_screen, select,
    )
    from harness.templates import load_template_file

    led = resolve_ledger(args.ledger, quiet=True)
    corpus = resolve_corpus_dir()
    if corpus is None:
        print("ERROR: corpus_dir is not configured; the collision screen needs it.",
              file=sys.stderr)
        return 1

    print(f"indexing {corpus} ...", file=sys.stderr)
    index = load_corpus_index(corpus, [d.get("codename") for d in led.deals])
    print(f"  {len(index.docs)} documents", file=sys.stderr)

    _fam, tpls, doc = load_template_file(
        os.path.join(ITEMS_DIR, "implicit_rule_application.yaml"))
    lever_map = load_lever_fields(doc)
    fact_block = doc.get("fact_block") or ""
    printed = printed_fields(fact_block)
    slots = {}
    for t in tpls:
        slots.update(t.slots or {})

    targets, pool_sha = load_pool(HOLDOUT_POOL)
    screen = make_screen(index)
    res = admit(targets, lever_map, printed, index, screen, fact_block, slots)
    sel = select(res, args.seed, BALANCED_CAP, RULE_CAP, N_CONTROL_PER_CLASS)

    prov = baseline(os.path.join(os.path.dirname(corpus), "ledger-v4"), corpus)

    out_targets = []
    for i, k in enumerate(sel["advance_keys"]):
        out_targets.append(_pool_target(res["units"][k], k,
                                        "holdout-v3-adv-%03d" % i, "advance", prov))
    for i, k in enumerate(sel["pass_keys"]):
        out_targets.append(_pool_target(res["units"][k], k,
                                        "holdout-v3-pas-%03d" % i, "pass", prov))
    # `ctl-` in the id, not just the bucket's first letters: "clean_advance"[6:9]
    # is "adv", which collided head-on with the scored advance ids and produced
    # two items with one item_id. `load_items` refused the file, which is what it
    # is for, but the fix belongs here.
    for cname, keys in sorted(sel["control_keys"].items()):
        for i, k in enumerate(keys):
            short = "adv" if cname == "clean_advance" else "pas"
            out_targets.append(_pool_target(
                res["units"][k], k, "holdout-v3-ctl-%s-%03d" % (short, i),
                None, prov))

    payload = {
        "version": 3,
        "source_pool": {"path": os.path.relpath(HOLDOUT_POOL, REPO).replace("\\", "/"),
                        "sha256": pool_sha, "targets": len(targets)},
        "generated_against": {"provenance": prov},
        "policy": {
            "balanced_across_gold_class": True,
            "per_class_cap": BALANCED_CAP,
            "n_per_class": sel["n_per_class"],
            "chance_floor": sel["chance_floor"],
            "rule_cap_requested": RULE_CAP,
            "rule_cap_scope": "within gold class",
            "pass_rule_cap": sel["pass_rule_cap"],
            "advance_rule_cap_applied": sel["advance_rule_cap_applied"],
            "advance_rule_cap_note": sel["advance_rule_cap_note"],
            "controls_per_class": N_CONTROL_PER_CLASS,
            "boundary_gaps": [list(g) for g in BOUNDARY_GAPS],
            "seed": args.seed,
        },
        "availability": sel["available"],
        "admission": {
            "targets_seen": res["targets_seen"],
            "drops": res["drops"],
            "inadmissible_reasons": res["inadmissible_reasons"],
            "boundary_gap_drops": res["boundary_gap_drops"],
            "boundary_gap_detail": res["boundary_gap_detail"],
            "unit_key_collapse": res["unit_key_collapse"],
            "fact_block_collapse": res["fact_block_collapse"],
            "collisions": res["collisions"],
        },
        "rule_distribution": {"advance": sel["advance_rule_distribution"],
                              "pass": sel["pass_rule_cap"].get("per_rule")},
        "n_scored": sel["n_scored"],
        "n_control": sum(len(v) for v in sel["control_keys"].values()),
        "holdout_targets": sorted(out_targets, key=lambda t: t["id"]),
    }

    out = args.out or HOLDOUT_SELECTED
    assert_writable(out, what="write holdout-targets-v3")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("")
    print(f"POOL {len(targets)} targets, sha256 {pool_sha[:16]}")
    print("")
    print("ADMISSION")
    for reason, n in res["drops"].items():
        print(f"  {n:>6}  dropped -- {reason}")
    print(f"  {len(res['units']):>6}  admitted distinct units")
    print(f"          boundary gap: {res['boundary_gap_drops']}")
    print(f"          collapsed onto an occupied unit key: {res['unit_key_collapse']}")
    print(f"          collapsed onto an identical printed fact block: "
          f"{res['fact_block_collapse'] or '{}'}")
    print("")
    print("AVAILABILITY BY CLASS")
    for k, v in sorted(sel["available"].items()):
        print(f"  {k:26} {v:>5}")
    print("")
    print(f"n_advance = n_pass = min({sel['available']['discriminating_advance']}, "
          f"{sel['available']['discriminating_pass']}, {BALANCED_CAP}) = "
          f"{sel['n_per_class']}")
    print(f"SCORED {sel['n_scored']}   CONTROLS {payload['n_control']}   "
          f"chance floor {sel['chance_floor']}")
    print("")
    print("RULE DISTRIBUTION")
    print(f"  advance: {sel['advance_rule_distribution']}  (cap not applied: "
          f"structural)")
    print(f"  pass   : {sel['pass_rule_cap'].get('per_rule')}  cap used "
          f"{sel['pass_rule_cap'].get('cap_used')} "
          f"{'IMPOSSIBLE AT REQUESTED' if sel['pass_rule_cap'].get('impossible') else '(requested)'}")
    print("")
    print(f"wrote {out}")
    return 0


def _pool_target(t: dict, key: str, new_id: str, gold_class, prov: dict) -> dict:
    out = {k: v for k, v in t.items() if not k.startswith("_")}
    out["unit_key"] = key
    out["pool_id"] = t.get("pool_id") or t.get("id")
    out["id"] = new_id
    out["gold_class"] = gold_class or t.get("ground_truth_outcome")
    out["scored"] = gold_class is not None
    out["provenance"] = {
        "generator": "v4_holdout.build (upstream, via holdout-pool-v6.json)",
        "selected_by": "harness.holdout_pool.select",
        "imported_read_only": True,
        "seed": (prov.get("ledger") or {}).get("index_seed"),
        "bucket": t.get("bucket"),
    }
    return out


def cmd_generate(args) -> int:
    """Templates + ledger -> a concrete item file. Raises on any row reuse."""
    from harness.baseline import baseline
    from harness.corpusindex import load_corpus_index
    from harness.generate import SEED, build_items, write_items
    from harness.templates import load_all_templates, load_template_file

    led = resolve_ledger(args.ledger)
    corpus = resolve_corpus_dir()
    if corpus is None:
        print("ERROR: corpus_dir is not configured; the gates need the document tree.",
              file=sys.stderr)
        return 1
    print(f"indexing {corpus} ...", file=sys.stderr)
    index = load_corpus_index(corpus, [d.get("codename") for d in led.deals])

    family_docs = {}
    for fname in sorted(os.listdir(ITEMS_DIR)):
        if fname.endswith(".yaml"):
            fam, _t, doc = load_template_file(os.path.join(ITEMS_DIR, fname))
            family_docs[fam] = doc

    hv2 = None
    hpath = args.holdout or resolve_holdout_selected()
    if os.path.exists(hpath):
        with io.open(hpath, encoding="utf-8") as fh:
            hv2 = json.load(fh)
    else:
        print(f"WARNING: {hpath} not found; implicit_rule_application will emit "
              f"zero items. Run `holdout-v2` first.", file=sys.stderr)

    templates = load_all_templates(ITEMS_DIR)
    built = build_items(led, templates, index, family_docs, hv2, seed=args.seed)

    prov = baseline(os.path.join(os.path.dirname(corpus), "ledger-v4"), corpus)
    out = args.out or os.path.join(DATA_DIR, "items-v2.json")
    write_items(out, built, prov)

    rep = built["report"]
    print("")
    print(f"{'family':28}{'capacity':>10}{'n':>8}  notes")
    total = 0
    for fam, d in sorted(rep["families"].items()):
        total += d.get("n", 0)
        extra = []
        if d.get("field_cap_impossible"):
            extra.append(f"field cap 0.20 impossible; used {d['field_cap_used']}")
        if d.get("n_control"):
            extra.append(f"{d['n_control']} controls (unscored)")
        if d.get("distinct_subjects"):
            extra.append(f"{d['distinct_subjects']} subjects")
        print(f"{fam:28}{str(d.get('capacity')):>10}{d.get('n', 0):>8}  {'; '.join(extra)}")
    print(f"{'TOTAL SCORED ITEMS':28}{'':>10}{total:>8}")
    print(f"wrote {out}")
    return 0


def cmd_verify_items(args) -> int:
    """Regenerate the item file and prove it matches the committed one.

    The single command that makes "deterministic" checkable rather than
    asserted. Regenerates into results/ (never over the committed file) and
    compares sha256. A mismatch means the ledger, the corpus or the templates
    moved -- the diff of the provenance blocks says which.
    """
    import tempfile

    from harness.baseline import baseline
    from harness.corpusindex import load_corpus_index
    from harness.generate import build_items, write_items
    from harness.paths import permit_root
    from harness.templates import load_all_templates, load_template_file

    committed = args.items or os.path.join(DATA_DIR, "items-v2.json")
    if not os.path.exists(committed):
        print(f"ERROR: {committed} does not exist. Run `generate` first.", file=sys.stderr)
        return 1

    led = resolve_ledger(args.ledger, quiet=True)
    corpus = resolve_corpus_dir()
    if corpus is None:
        print("ERROR: corpus_dir is not configured.", file=sys.stderr)
        return 1
    index = load_corpus_index(corpus, [d.get("codename") for d in led.deals])

    family_docs = {}
    for fname in sorted(os.listdir(ITEMS_DIR)):
        if fname.endswith(".yaml"):
            fam, _t, doc = load_template_file(os.path.join(ITEMS_DIR, fname))
            family_docs[fam] = doc
    hpath = args.holdout or resolve_holdout_selected()
    hv2 = json.load(io.open(hpath, encoding="utf-8")) if os.path.exists(hpath) else None

    tmpdir = tempfile.mkdtemp(prefix="verify-items-")
    permit_root(tmpdir)
    built = build_items(led, load_all_templates(ITEMS_DIR), index, family_docs, hv2,
                        seed=args.seed)
    prov = baseline(os.path.join(os.path.dirname(corpus), "ledger-v4"), corpus)
    fresh = write_items(os.path.join(tmpdir, "items.json"), built, prov)

    a, b = _sha256(committed), _sha256(fresh)
    print("")
    print(f"committed:    {committed}")
    print(f"  sha256      {a}")
    print(f"regenerated:  {fresh}")
    print(f"  sha256      {b}")
    print("")
    if a == b:
        n = len(built["items"])
        print(f"IDENTICAL. {n} items reproduced byte for byte from seed {args.seed}, "
              f"ledger {prov['ledger']['shards']['deals']['sha256'][:16]}, "
              f"corpus digest {prov['corpus']['tree_digest'][:16]}.")
        return 0
    print("DIFFERENT. The committed item file was not produced by these inputs.")
    old = json.load(io.open(committed, encoding="utf-8"))
    for label, was, now in (
        ("seed", old.get("seed"), built["report"]["seed"]),
        ("deals shard", ((old.get("provenance") or {}).get("ledger") or {})
         .get("shards", {}).get("deals", {}).get("sha256"),
         prov["ledger"]["shards"]["deals"]["sha256"]),
        ("corpus digest", ((old.get("provenance") or {}).get("corpus") or {})
         .get("tree_digest"), prov["corpus"]["tree_digest"]),
        ("item count", len(old.get("items") or []), len(built["items"])),
    ):
        if was != now:
            print(f"  {label}: committed {was!r} -> now {now!r}")
    old_ids = {i["item_id"] for i in (old.get("items") or [])}
    new_ids = {i["item_id"] for i in built["items"]}
    if old_ids != new_ids:
        gone, added = sorted(old_ids - new_ids), sorted(new_ids - old_ids)
        print(f"  items removed: {len(gone)}  added: {len(added)}")
        for i in gone[:5]:
            print(f"    - {i}")
        for i in added[:5]:
            print(f"    + {i}")
    else:
        print("  the same items, but some field differs -- diff the two files to see it")
    return 1


def cmd_validate_templates(args) -> int:
    """Resolve every template's selector and gold spec against the ledger.

    THIS GENERATES NO QUESTIONS. It reports how many ledger rows each template
    would draw from and whether its gold answer is computable, which is what
    you need to know before generating anything -- and nothing more.
    """
    from harness.templates import load_template_file, validate_template

    led = resolve_ledger(args.ledger)
    files = sorted(f for f in os.listdir(ITEMS_DIR) if f.endswith(".yaml"))
    total, ok, bad = 0, 0, 0
    results = []
    print(f"\nvalidating templates against ledger schema={led.schema} "
          f"({len(led.deals)} deals, {len(led.holdout_targets)} holdout targets)\n")
    for fname in files:
        path = os.path.join(ITEMS_DIR, fname)
        try:
            fam, templates, _doc = load_template_file(path)
        except ValueError as exc:
            print(f"--- {fname} --- FAILED TO LOAD: {exc}\n")
            bad += 1
            continue
        if not templates:
            continue
        print(f"--- {fname}  (family={fam}, {len(templates)} templates) ---")
        for tpl in templates:
            total += 1
            res = validate_template(tpl, led)
            results.append(res)
            print(res.line())
            if res.ok:
                ok += 1
            else:
                bad += 1
        print()
    print(f"templates: {total} total, {ok} resolvable, {bad} empty-or-broken")
    print("NOTE: no questions were generated. This reports candidate counts and "
          "gold computability only.")
    if args.json_out:
        assert_writable(args.json_out, what="write --json-out")
        with io.open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump([r.__dict__ for r in results], fh, indent=2, default=str)
        print(f"wrote {args.json_out}")
    return 1 if bad and args.strict else 0


def cmd_validate_items(args) -> int:
    """Check a generated item file before spending a run on it.

    Two jobs. First, shape: does every item carry what its family's grader
    needs. Second, and more useful, POWER: it prints the Wilson half-width each
    family's item count actually buys, so the consequence of a sampling choice
    is visible at generation time rather than after the run.
    """
    from collections import Counter

    from graders.base import FAMILIES, GRADEABLE
    from graders.numeric import UNITS
    from harness.stats import ProportionStat, halfwidth_table, wilson_ci

    items = load_items(args.items)
    cfg = load_grading_config()
    checklists = cfg.get("checklists") or {}
    min_n = int((cfg.get("reporting") or {}).get("min_items_for_claim", 25))

    errors: list[str] = []
    warnings: list[str] = []

    for it in items:
        iid = it.get("item_id")
        fam = it.get("family")
        if fam not in GRADEABLE:
            errors.append(f"{iid}: unknown family {fam!r}")
            continue
        if not str(it.get("question") or "").strip():
            errors.append(f"{iid}: empty question")
        gold = it.get("gold")
        if not isinstance(gold, dict):
            errors.append(f"{iid}: missing or malformed `gold` block")
            continue

        if fam == "convention_conformance":
            ref = gold.get("checklist_ref") or gold.get("checklist")
            if not gold.get("checks") and not ref:
                errors.append(f"{iid}: needs gold.checks or gold.checklist_ref")
            elif ref and ref not in checklists:
                errors.append(
                    f"{iid}: checklist_ref {ref!r} not found "
                    f"(available: {sorted(checklists)})")
        else:
            if gold.get("value") is None:
                errors.append(f"{iid}: gold.value is missing")
            unit = str(gold.get("unit") or "none")
            if unit not in UNITS:
                errors.append(f"{iid}: gold.unit {unit!r} not one of {list(UNITS)}")
            atype = str(it.get("answer_type") or "").lower()
            if not atype:
                warnings.append(f"{iid}: no answer_type; the grader will assume numeric")

        if fam == "implicit_rule_application":
            meta = it.get("meta") or {}
            for key in ("holdout_id", "written_criteria_verdict", "discriminates"):
                if key not in meta:
                    warnings.append(
                        f"{iid}: meta.{key} missing -- this item will not appear in the "
                        f"discriminating/clean breakout that this family reports")

    print(f"\nitems file: {args.items}")
    print(f"{len(items)} items, {len(errors)} error(s), {len(warnings)} warning(s)\n")

    # Count SCORED items. Family 4 carries 20 controls that are excluded from
    # its score by the pre-registration; folding them into n would print an
    # interval for a number nobody reports.
    def _scored(it) -> bool:
        return (it.get("meta") or {}).get("scored", True) is not False

    by_fam = Counter(it.get("family") for it in items if _scored(it))
    by_fam_all = Counter(it.get("family") for it in items)
    print("| family | n scored | difficulty mix | 95% CI half-width at this n |")
    print("|---|---:|---|---|")
    for fam in GRADEABLE:
        n = by_fam.get(fam, 0)
        diffs = Counter((it.get("meta") or {}).get("difficulty", "?")
                        for it in items if it.get("family") == fam and _scored(it))
        mix = ", ".join(f"{k}:{v}" for k, v in sorted(diffs.items())) or "-"
        if n:
            hw50 = 100 * wilson_ci(round(0.5 * n), n).halfwidth
            hw80 = 100 * wilson_ci(round(0.8 * n), n).halfwidth
            band = f"+/-{hw50:.1f} pts at 50%, +/-{hw80:.1f} at 80%"
        else:
            band = "no items"
        flag = "  <-- below min_items_for_claim" if 0 < n < min_n else ""
        label = fam if fam in FAMILIES else f"{fam} (reported separately, not one of the six)"
        extra = by_fam_all.get(fam, 0) - n
        if extra:
            label += f" [+{extra} unscored controls]"
        print(f"| {label} | {n} | {mix} | {band}{flag} |")
    print()
    print("Six families of equal standing. They are never averaged: no row above may "
          "be combined with another.")
    print()

    ira = [it for it in items if it.get("family") == "implicit_rule_application"]
    if ira:
        scored = [it for it in ira if _scored(it)]
        ctl = [it for it in ira if not _scored(it)]
        buckets = Counter((it.get("meta") or {}).get("bucket", "?") for it in ira)
        gold = Counter(str((it.get("gold") or {}).get("value")) for it in scored)
        floors = {(it.get("meta") or {}).get("chance_floor") for it in scored}
        floor = next(iter(floors)) if len(floors) == 1 else None

        print("=" * 74)
        print("FAMILY 4 -- implicit_rule_application")
        print("=" * 74)
        print()
        print("The advance class rests on a SINGLE RULE with one behavioural degree")
        print("of freedom: RULE-A1 is the only rule in this corpus that produces the")
        print("advance class, and every advance item is that one rule applied to a")
        print("different deal. The interval below is an ITEM-LEVEL interval and is")
        print("NOT a claim about generalisation across rules.")
        print()
        # These two lines predate the family-4 block and are kept verbatim:
        # tests/test_cli.py asserts them, and the new detail below is an
        # addition to what this command reported, not a replacement for it.
        disc = sum(1 for it in ira if (it.get("meta") or {}).get("discriminates") is True)
        print(f"implicit_rule_application buckets: {dict(buckets)}")
        print(f"  discriminating: {disc}/{len(ira)}")
        print()
        print(f"  n scored          {len(scored)}")
        print(f"  GOLD SPLIT        {dict(sorted(gold.items()))}")
        if gold:
            top, k = gold.most_common(1)[0]
            print(f"  majority class    {top!r} at {k}/{len(scored)} = "
                  f"{100.0 * k / max(1, len(scored)):.1f}%")
        print(f"  CHANCE FLOOR      {floor if floor is not None else 'NOT RECORDED'}"
              f"{'  (50%)' if floor == 0.5 else ''}")
        if len(scored):
            hw = 100 * wilson_ci(round(0.5 * len(scored)), len(scored)).halfwidth
            print(f"  WILSON half-width +/-{hw:.2f} points at n={len(scored)}, p=0.5")
        print()

        print("  RULE DISTRIBUTION WITHIN EACH GOLD CLASS")
        print("  (the 35% cap applies within class, not across the family)")
        print()
        for cls in sorted({str((it.get("gold") or {}).get("value")) for it in scored}):
            sub = [it for it in scored
                   if str((it.get("gold") or {}).get("value")) == cls]
            rc: Counter = Counter()
            for it in sub:
                m = it.get("meta") or {}
                # `driving_rule_ids` already carries the rescue -- `normalise`
                # folds it in, which is what makes the advance class keyable at
                # all. Adding `rescued_by_rule` again counted the rule twice and
                # printed it at 200% of its own class.
                rules = list(dict.fromkeys(
                    list(m.get("driving_rule_ids") or [])
                    + list(m.get("rescued_by_rule") or [])))
                for r in rules:
                    rc[r] += 1
            print(f"    gold `{cls}`  n={len(sub)}")
            for r, k in rc.most_common():
                share = 100.0 * k / max(1, len(sub))
                over = "   <-- ABOVE 35%" if share > 35.0 and len(rc) > 1 else ""
                print(f"      {r:12} {k:>4}  {share:5.1f}%{over}")
            if len(rc) == 1:
                only = next(iter(rc))
                print(f"      {only} supplies 100% of this class. Structural, not a")
                print(f"      sampling choice: it is the only rule in the corpus that")
                print(f"      can produce this verdict. Disclosed, not capped.")
            print()

        if ctl:
            cg = Counter(str((it.get("gold") or {}).get("value")) for it in ctl)
            cb = Counter((it.get("meta") or {}).get("bucket") for it in ctl)
            print(f"  CONTROLS {len(ctl)} (excluded from the score, own line): "
                  f"{dict(cb)} gold {dict(cg)}")
            print()

        if disc and disc < min_n:
            print(f"  NOTE: {disc} discriminating items is below min_items_for_claim "
                  f"({min_n}). The discriminating slice is the narrowest in the set; "
                  f"consider oversampling that bucket.")
            print()
        elif len(scored) < min_n:
            print(f"  NOTE: {len(scored)} scored items is below min_items_for_claim "
                  f"({min_n}).")
            print()
        print("=" * 74)
        print()

    rec = [it for it in items if (it.get("meta") or {}).get("recency")]
    if rec:
        print(f"recency slice: {len(rec)} items over "
              f"{len({(it.get('meta') or {}).get('deal_codename') for it in rec})} subjects, "
              f"scored separately from single_fact_recall")
        print()

    # What a responder that reads nothing would score. Printed here because a
    # family a constant can beat is not measuring the model, and the number is
    # only obvious once someone computes it.
    gold_by_fam: dict = {}
    for it in items:
        if not (it.get("meta") or {}).get("scored", True):
            continue
        g = (it.get("gold") or {}).get("value")
        if isinstance(g, str):
            gold_by_fam.setdefault(it["family"], Counter())[g] += 1
    degenerate = []
    for fam, c in sorted(gold_by_fam.items()):
        top, k = c.most_common(1)[0]
        tot = sum(c.values())
        if tot and k / tot >= 0.9:
            degenerate.append((fam, top, k, tot))
    if degenerate:
        print("CONSTANT-ANSWER BASELINE -- families a fixed answer scores well on:")
        for fam, top, k, tot in degenerate:
            print(f"  {fam}: always answering {top!r} scores {k}/{tot} = "
                  f"{100.0*k/tot:.0f}%")
        print("  Run the `constant_pass` arm alongside the real arms; a family score "
              "at or below this line is not evidence about the model.")
        print()

    for w in warnings[:40]:
        print(f"  WARN  {w}")
    if len(warnings) > 40:
        print(f"  ... and {len(warnings) - 40} more warnings")
    for e in errors[:40]:
        print(f"  ERROR {e}")
    if len(errors) > 40:
        print(f"  ... and {len(errors) - 40} more errors")

    print()
    print("OK" if not errors else f"{len(errors)} error(s) -- fix before running")
    return 1 if errors and args.strict else 0


def cmd_allowlist(args) -> int:
    from graders.fabrication import FabricationScorer

    led = resolve_ledger(args.ledger)
    sc = FabricationScorer(led, load_fabrication_config())
    print(json.dumps(sc.allowlist_summary(), indent=2))
    if args.probe:
        rep = sc.score(args.probe)
        print(json.dumps(rep.to_dict(), indent=2)[:6000])
    return 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness.cli", description="benchmark-v1 harness")
    p.add_argument("--ledger", default=None, help="override the ledger path")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run items against arms")
    r.add_argument("--items", required=True)
    r.add_argument("--arms", default=None, help="comma-separated arm names")
    r.add_argument("--arms-config", default=os.path.join(CONFIG_DIR, "arms.yaml"))
    r.add_argument("--family", default=None, help="restrict to comma-separated families")
    r.add_argument("--out", default=None)
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--report", action="store_true")
    r.add_argument("--report-out", default=None)
    r.add_argument("--strict", action="store_true", help="exit non-zero on harness errors")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="render a report from a finished run")
    rp.add_argument("--run", required=True)
    rp.add_argument("--out", default=None)
    rp.set_defaults(func=cmd_report)

    f = sub.add_parser("fixtures", help="end-to-end fixture run and report")
    f.add_argument("--out", default=None)
    f.add_argument("--report-out", default=None)
    f.add_argument("--copy-to", default=None)
    f.add_argument("--no-resume", action="store_true")
    f.set_defaults(func=cmd_fixtures)

    vl = sub.add_parser("validate-ledger")
    vl.set_defaults(func=cmd_validate_ledger)

    vt = sub.add_parser("validate-templates")
    vt.add_argument("--strict", action="store_true")
    vt.add_argument("--json-out", default=None)
    vt.set_defaults(func=cmd_validate_templates)

    vi = sub.add_parser("validate-items",
                        help="check a generated item file's shape and its statistical power")
    vi.add_argument("--items", required=True)
    vi.add_argument("--strict", action="store_true")
    vi.set_defaults(func=cmd_validate_items)

    vi2 = sub.add_parser("verify-items",
                         help="regenerate the item file and prove it matches the committed one")
    vi2.add_argument("--items", default=None)
    vi2.add_argument("--seed", type=int, default=20260819)
    vi2.add_argument("--holdout", default=None)
    vi2.set_defaults(func=cmd_verify_items)

    gen = sub.add_parser("generate", help="build the item file from templates + ledger")
    gen.add_argument("--out", default=None)
    gen.add_argument("--holdout", default=None)
    gen.add_argument("--seed", type=int, default=20260819)
    gen.set_defaults(func=cmd_generate)

    hp = sub.add_parser("holdout-pool",
                        help="select family 4's item set from the saturated v6 pool")
    hp.add_argument("--out", default=None)
    hp.add_argument("--seed", type=int, default=20260819)
    hp.set_defaults(func=cmd_holdout_pool)

    hv = sub.add_parser("holdout-v2", help="regenerate holdout targets to saturation")
    hv.add_argument("--out", default=None)
    hv.set_defaults(func=cmd_holdout_v2)

    cap = sub.add_parser("capacity", help="measure honest capacity per family")
    cap.add_argument("--md-out", default=None)
    cap.add_argument("--json-out", default=None)
    cap.set_defaults(func=cmd_capacity)

    al = sub.add_parser("allowlist")
    al.add_argument("--probe", default=None, help="score a string against the allowlist")
    al.set_defaults(func=cmd_allowlist)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except RefusedWrite as exc:
        # A refused write is a guard doing its job, not a crash. Print the
        # reason and exit 2 rather than dumping a traceback -- exit 2 so it is
        # distinguishable from an ordinary failure (1).
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
