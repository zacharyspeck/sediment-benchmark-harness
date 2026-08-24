"""Family 5 -- convention_conformance.

Draft a document. Grade against a deterministic checklist built from the
ledger: section order matches the canonical sequence, required elements
present, house terminology used, forbidden elements absent. The score is the
fraction of checklist items passed.

NO LLM JUDGE. NOT ANYWHERE. Every check below is a string or regex operation.
If a convention cannot be expressed as one, it does not belong in this
benchmark -- "reads like the house voice" is not gradeable and inviting a model
to arbitrate it would make the headline number unfalsifiable.

CHECK KINDS
-----------
  must_contain       every pattern (or `any_of` group) must match
  must_not_contain   no pattern may match          -- forbidden elements
  section_order      named sections appear, in the canonical order
  number_format      every money/percent figure matches the house format
  max_count          a pattern occurs at most N times   -- e.g. superlatives
  min_count          a pattern occurs at least N times
  requires_nearby    if A appears, B must appear within N characters
                     -- "name the risk paired with mitigants" (HR-07)
  first_line_matches the verdict is in line one -- Westbrook's memo convention

CONDITIONAL CHECKS
------------------
A check may carry `applies_when: {pattern: ...}`. If the trigger does not
appear, the check is marked not-applicable and drops out of the denominator
rather than being scored as a pass. Inapplicable checks that score as passes
inflate conformance for a document that simply said less.

SCORING
-------
score = passed / applicable. This is a FRACTION, not a Bernoulli trial, so the
grade is marked `binary=False` and the stats layer gives it a bootstrap
interval instead of a Wilson interval -- see harness/stats.py.
"""

from __future__ import annotations

import re

from harness.extractor import extract_answer
from .base import Grade, GradeContext, VERDICT_CORRECT, VERDICT_INCORRECT, VERDICT_PARTIAL, register

GRADER = "convention_conformance"
VERSION = "1.0.0"

_MONEY = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:M|m|million|B|bn|billion|K|k)?\b")
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s?%")


class CheckError(ValueError):
    pass


def _flags(spec: dict) -> int:
    f = re.MULTILINE
    if spec.get("case_sensitive"):
        return f
    return f | re.IGNORECASE


def _find(pattern: str, text: str, spec: dict) -> list[dict]:
    try:
        rx = re.compile(pattern, _flags(spec))
    except re.error as exc:
        raise CheckError(f"bad regex {pattern!r}: {exc}") from exc
    return [
        {"pattern": pattern, "match": m.group(0)[:120], "start": m.start(), "end": m.end()}
        for m in rx.finditer(text or "")
    ]


def _evidence_line(text: str, pos: int, width: int = 110) -> str:
    lo = max(0, pos - width // 2)
    return (text[lo: pos + width // 2] or "").replace("\n", " ⏎ ").strip()


# --------------------------------------------------------------------------
# individual check kinds
# --------------------------------------------------------------------------
def _check_must_contain(spec, text):
    groups = spec.get("any_of")
    if groups:
        hits = []
        for p in groups:
            hits.extend(_find(p, text, spec))
        ok = bool(hits)
        return ok, {"matched": hits[:6], "mode": "any_of", "patterns": groups}, (
            "found " + hits[0]["match"] if ok else f"none of {len(groups)} accepted patterns appeared"
        )
    pats = spec.get("patterns") or ([spec["pattern"]] if spec.get("pattern") else [])
    if not pats:
        raise CheckError(f"must_contain check {spec.get('id')!r} has no patterns")
    missing, matched = [], []
    for p in pats:
        h = _find(p, text, spec)
        (matched.extend(h) if h else missing.append(p))
    ok = not missing
    return ok, {"matched": matched[:6], "missing": missing, "patterns": pats}, (
        f"all {len(pats)} required pattern(s) present" if ok else f"missing: {missing}"
    )


def _check_must_not_contain(spec, text):
    pats = spec.get("patterns") or ([spec["pattern"]] if spec.get("pattern") else [])
    if not pats:
        raise CheckError(f"must_not_contain check {spec.get('id')!r} has no patterns")
    found = []
    for p in pats:
        found.extend(_find(p, text, spec))
    ok = not found
    ev = [{**f, "context": _evidence_line(text, f["start"])} for f in found[:6]]
    return ok, {"forbidden_found": ev, "patterns": pats}, (
        "no forbidden pattern present" if ok
        else f"forbidden text present: {[f['match'] for f in found[:4]]}"
    )


def _check_section_order(spec, text):
    sections = spec.get("sections") or []
    if not sections:
        raise CheckError(f"section_order check {spec.get('id')!r} has no sections")
    positions, missing = [], []
    for sec in sections:
        name = sec.get("name", "<unnamed>")
        pats = sec.get("any_of") or sec.get("patterns") or []
        first = None
        for p in pats:
            for h in _find(p, text, spec):
                first = h["start"] if first is None else min(first, h["start"])
        if first is None:
            missing.append(name)
        else:
            positions.append({"name": name, "at": first})
    if missing:
        return False, {"positions": positions, "missing": missing}, (
            f"sections not found: {missing}"
        )
    order = [p["at"] for p in positions]
    ok = all(order[i] < order[i + 1] for i in range(len(order) - 1))
    seen = [p["name"] for p in sorted(positions, key=lambda p: p["at"])]
    return ok, {"positions": positions, "observed_order": seen,
                "canonical_order": [s.get("name") for s in sections]}, (
        "sections appear in canonical order" if ok
        else f"order was {seen}, canonical is {[s.get('name') for s in sections]}"
    )


def _check_number_format(spec, text):
    """House format: money in $M with one decimal; margins to one decimal."""
    which = spec.get("target", "money")
    decimals = int(spec.get("decimals", 1))
    rx = _MONEY if which == "money" else _PERCENT
    bad, good = [], []
    for m in rx.finditer(text or ""):
        tok = m.group(0)
        num = re.search(r"\d[\d,]*(?:\.(\d+))?", tok)
        dec = num.group(1) if num and num.group(1) else ""
        if len(dec) == decimals:
            good.append(tok)
        else:
            bad.append({"match": tok, "start": m.start(), "decimals": len(dec),
                        "context": _evidence_line(text, m.start())})
    if not good and not bad:
        return None, {"target": which, "note": "no figures of this kind appeared"}, "not applicable"
    ok = not bad
    return ok, {"target": which, "required_decimals": decimals,
                "conforming": good[:8], "nonconforming": bad[:8]}, (
        f"all {len(good)} {which} figure(s) carry {decimals} decimal(s)" if ok
        else f"{len(bad)} {which} figure(s) off-format: {[b['match'] for b in bad[:4]]}"
    )


def _check_max_count(spec, text):
    pats = spec.get("patterns") or ([spec["pattern"]] if spec.get("pattern") else [])
    limit = int(spec.get("max", 0))
    hits = []
    for p in pats:
        hits.extend(_find(p, text, spec))
    ok = len(hits) <= limit
    ev = [{**h, "context": _evidence_line(text, h["start"])} for h in hits[:8]]
    return ok, {"count": len(hits), "max": limit, "hits": ev}, (
        f"{len(hits)} occurrence(s) <= limit {limit}" if ok
        else f"{len(hits)} occurrence(s) exceeds limit {limit}: {[h['match'] for h in hits[:5]]}"
    )


def _check_min_count(spec, text):
    pats = spec.get("patterns") or ([spec["pattern"]] if spec.get("pattern") else [])
    need = int(spec.get("min", 1))
    hits = []
    for p in pats:
        hits.extend(_find(p, text, spec))
    ok = len(hits) >= need
    return ok, {"count": len(hits), "min": need, "hits": hits[:8]}, (
        f"{len(hits)} occurrence(s) >= required {need}" if ok
        else f"only {len(hits)} occurrence(s), need {need}"
    )


def _check_requires_nearby(spec, text):
    trig = spec.get("trigger") or spec.get("if_pattern")
    need = spec.get("require") or spec.get("then_patterns") or []
    window = int(spec.get("window", 400))
    if not trig or not need:
        raise CheckError(f"requires_nearby check {spec.get('id')!r} needs `trigger` and `require`")
    triggers = _find(trig, text, spec)
    if not triggers:
        return None, {"trigger": trig, "note": "trigger absent"}, "not applicable"
    unmet = []
    for t in triggers:
        lo, hi = max(0, t["start"] - window), min(len(text), t["end"] + window)
        window_text = text[lo:hi]
        if not any(_find(p, window_text, spec) for p in need):
            unmet.append({**t, "context": _evidence_line(text, t["start"])})
    ok = not unmet
    return ok, {"trigger": trig, "require": need, "window": window,
                "triggers": len(triggers), "unpaired": unmet[:5]}, (
        f"all {len(triggers)} trigger(s) paired within {window} chars" if ok
        else f"{len(unmet)} trigger(s) with no required pairing nearby"
    )


def _check_first_line_matches(spec, text):
    pats = spec.get("patterns") or spec.get("any_of") or ([spec["pattern"]] if spec.get("pattern") else [])
    n_lines = int(spec.get("lines", 1))
    lines = [l for l in (text or "").splitlines() if l.strip()][:n_lines]
    head = "\n".join(lines)
    hits = []
    for p in pats:
        hits.extend(_find(p, head, spec))
    ok = bool(hits)
    return ok, {"head": head[:200], "patterns": pats, "matched": hits[:4]}, (
        f"required opener present in first {n_lines} line(s)" if ok
        else f"first {n_lines} line(s) did not match any required opener"
    )


_KINDS = {
    "must_contain": _check_must_contain,
    "must_not_contain": _check_must_not_contain,
    "section_order": _check_section_order,
    "number_format": _check_number_format,
    "max_count": _check_max_count,
    "min_count": _check_min_count,
    "requires_nearby": _check_requires_nearby,
    "first_line_matches": _check_first_line_matches,
}


# --------------------------------------------------------------------------
# checklist resolution + ledger drift guard
# --------------------------------------------------------------------------
def _resolve_checklist(item: dict, ctx: GradeContext) -> tuple[list, dict]:
    gold = item.get("gold") or {}
    meta = {"source": "inline"}
    checks = gold.get("checks")
    if checks:
        return list(checks), meta
    ref = gold.get("checklist_ref") or gold.get("checklist")
    if ref:
        lib = (ctx.config or {}).get("checklists") or {}
        entry = lib.get(ref)
        if not entry:
            raise CheckError(
                f"checklist {ref!r} not found; available: {sorted(lib)}. "
                f"Checklists are loaded from items/convention_conformance.yaml."
            )
        meta = {"source": "library", "checklist_ref": ref,
                "checklist_version": entry.get("version")}
        return list(entry.get("checks") or []), meta
    raise CheckError(f"item {item.get('item_id')!r} has neither gold.checks nor gold.checklist_ref")


def _ledger_drift(checks, ctx: GradeContext) -> list[dict]:
    """Assert a checklist's canonical order still matches the ledger.

    A checklist is only trustworthy while it agrees with the ledger it claims
    to encode. If someone edits the house rules, this surfaces as a flag rather
    than as a silently wrong grade.
    """
    from harness.query import MISSING, get_path

    out = []
    led = ctx.ledger
    if led is None:
        return out
    for c in checks:
        lc = c.get("ledger_check")
        if not lc:
            continue
        path = lc.get("path")
        val = get_path({"canon": led.canon, "house_rules": led.house_rules,
                        "meta": led.meta}, path) if path else MISSING
        if val is MISSING or val is None:
            out.append({"check": c.get("id"), "path": path, "problem": "ledger path missing"})
            continue
        if "expect_len" in lc and isinstance(val, (list, tuple)) and len(val) != int(lc["expect_len"]):
            out.append({"check": c.get("id"), "path": path, "problem": "length drift",
                        "ledger_len": len(val), "checklist_len": int(lc["expect_len"])})
        if "expect_values" in lc and list(val) != list(lc["expect_values"]):
            out.append({"check": c.get("id"), "path": path, "problem": "value drift",
                        "ledger": list(val), "checklist": list(lc["expect_values"])})
    return out


# --------------------------------------------------------------------------
@register(GRADER)
def grade(item: dict, response: str, ctx: GradeContext) -> Grade:
    text = response or ""
    checks, cl_meta = _resolve_checklist(item, ctx)
    if not checks:
        raise CheckError(f"item {item.get('item_id')!r} resolved to an empty checklist")

    # The drafted document is the whole response. An ANSWER line is still
    # required by the format, and is excluded from the graded body so the
    # word "ANSWER" cannot satisfy or trip a check.
    ex = extract_answer(text, answer_type="string")
    body = text
    if ex.raw_line:
        body = text.replace(ex.raw_line, " ", 1)

    flags = list(ex.flags)
    if ex.uncertain:
        flags.append("extraction_uncertain")

    records, passed, applicable = [], 0, 0
    for spec in checks:
        cid = spec.get("id") or f"<unnamed:{spec.get('kind')}>"
        kind = spec.get("kind")
        rec = {"id": cid, "kind": kind, "description": spec.get("description", ""),
               "weight": float(spec.get("weight", 1.0))}
        try:
            gate = spec.get("applies_when")
            if gate:
                gpat = gate.get("pattern") or gate.get("patterns")
                gpats = [gpat] if isinstance(gpat, str) else list(gpat or [])
                if not any(_find(p, body, spec) for p in gpats):
                    rec.update(passed=None, applicable=False, why="trigger absent; check not applicable",
                               evidence={"applies_when": gpats})
                    records.append(rec)
                    continue
            fn = _KINDS.get(kind)
            if fn is None:
                raise CheckError(
                    f"unknown check kind {kind!r}; available: {sorted(_KINDS)}"
                )
            ok, evidence, why = fn(spec, body)
            if ok is None:
                rec.update(passed=None, applicable=False, why=why, evidence=evidence)
            else:
                rec.update(passed=bool(ok), applicable=True, why=why, evidence=evidence)
                applicable += 1
                passed += 1 if ok else 0
        except CheckError as exc:
            # A malformed check is a harness defect, not a model failure.
            rec.update(passed=None, applicable=False, why=f"CHECK ERROR: {exc}",
                       evidence={}, error=str(exc))
            flags.append("checklist_error")
        records.append(rec)

    drift = _ledger_drift(checks, ctx)
    if drift:
        flags.append("checklist_ledger_drift")

    score = (passed / applicable) if applicable else 0.0
    if applicable == 0:
        flags.append("no_applicable_checks")

    verdict = (
        VERDICT_CORRECT if applicable and passed == applicable
        else VERDICT_INCORRECT if passed == 0
        else VERDICT_PARTIAL
    )

    failed = [r["id"] for r in records if r.get("passed") is False]
    return Grade(
        item_id=item.get("item_id", "<no-id>"),
        family=item.get("family", GRADER),
        arm=ctx.arm,
        grader=GRADER,
        grader_version=VERSION,
        score=score,
        binary=False,  # fractional: Wilson does not apply, see harness/stats.py
        verdict=verdict,
        extracted_answer=ex.value,
        gold_answer=f"{applicable} applicable checklist items",
        comparison=(
            f"{passed}/{applicable} checklist items passed "
            f"({len(records) - applicable} not applicable); failed: {failed or 'none'}"
        ),
        flags=flags,
        detail={
            "checklist": cl_meta,
            "checks_total": len(records),
            "checks_applicable": applicable,
            "checks_passed": passed,
            "failed_check_ids": failed,
            "checks": records,
            "ledger_drift": drift,
            "perfect_conformance": bool(applicable and passed == applicable),
        },
        extraction=ex.to_trace(),
        gold_trace=dict((item.get("gold") or {}).get("trace") or {}),
        raw_response=text,
        meta=dict(item.get("meta") or {}),
    )
