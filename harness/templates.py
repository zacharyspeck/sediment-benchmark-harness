"""Load, bind and validate question templates.

Templates are DATA. A template says:
  - how the question is phrased (one or more patterns)
  - which ledger rows it draws from       (`selector`)
  - which fields fill the slots           (`slots`)
  - which field or aggregation supplies the gold answer (`gold`)
  - the difficulty tier and the answer type the grader expects

Slots are written `{{name}}` and may appear in the question text AND inside the
selector, so one template covers "healthcare deals screened in 2023" and
"business services deals screened in 2021" without duplication.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not generate questions. The ledger the questions depend on is still
being built, so generation is out of scope. `validate()` resolves a template's
selector and gold spec and reports how many candidate rows exist -- enough to
prove a template works, without emitting a single benchmark item. The hook
where generation would attach is marked at the bottom of this file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from .query import MISSING, QueryError, aggregate, get_path, resolve_gold, select
from .yamlio import load_file

__all__ = ["Template", "load_template_file", "load_all_templates", "substitute",
           "default_bindings", "validate_template", "ValidationResult"]

_SLOT_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass
class Template:
    id: str
    family: str
    source_file: str
    raw: dict = field(default_factory=dict)

    @property
    def selector(self) -> dict:
        return self.raw.get("selector") or {}

    @property
    def gold(self) -> dict:
        return self.raw.get("gold") or {}

    @property
    def slots(self) -> dict:
        return self.raw.get("slots") or {}

    @property
    def answer_type(self) -> str:
        return str(self.raw.get("answer_type") or "numeric")

    @property
    def difficulty(self) -> str:
        return str(self.raw.get("difficulty") or "unspecified")

    @property
    def questions(self) -> list[str]:
        q = self.raw.get("question")
        if isinstance(q, str):
            return [q]
        return list(self.raw.get("questions") or q or [])


def load_template_file(path: str) -> tuple[str, list[Template], dict]:
    doc = load_file(path) or {}
    family = doc.get("family") or os.path.basename(path).replace(".yaml", "")
    out = []
    for t in (doc.get("templates") or []):
        if not isinstance(t, dict):
            raise ValueError(f"{path}: a template entry is not a mapping")
        tid = t.get("id")
        if not tid:
            raise ValueError(f"{path}: a template has no `id`")
        out.append(Template(id=tid, family=t.get("family", family),
                            source_file=path, raw=t))
    return family, out, doc


def load_all_templates(items_dir: str) -> list[Template]:
    templates: list[Template] = []
    for fname in sorted(os.listdir(items_dir)):
        if not fname.endswith(".yaml"):
            continue
        _, ts, _ = load_template_file(os.path.join(items_dir, fname))
        templates.extend(ts)
    return templates


# --------------------------------------------------------------------------
def substitute(obj: Any, bindings: dict) -> Any:
    """Replace {{slot}} placeholders throughout a nested structure.

    A string that is exactly one placeholder becomes the bound VALUE (keeping
    its type -- an int year stays an int). A placeholder inside a longer string
    is interpolated as text.
    """
    if isinstance(obj, str):
        m = _SLOT_RE.fullmatch(obj.strip())
        if m:
            return bindings.get(m.group(1), obj)
        return _SLOT_RE.sub(lambda mm: str(bindings.get(mm.group(1), mm.group(0))), obj)
    if isinstance(obj, dict):
        return {k: substitute(v, bindings) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, bindings) for v in obj]
    return obj


def default_bindings(tpl: Template, ledger) -> dict:
    """Pick one concrete value per slot, so a template can be validated.

    Preference: the first entry of the slot's `choose` list; otherwise the most
    common distinct value of the slot's `from` path in the ledger.
    """
    out: dict = {}
    for name, spec in (tpl.slots or {}).items():
        if not isinstance(spec, dict):
            # shorthand: slot maps directly to a row field, filled per-row later
            continue
        if spec.get("choose"):
            out[name] = spec["choose"][0]
            continue
        frm = spec.get("from")
        if not frm:
            continue
        coll, _, path = str(frm).partition(".")
        try:
            rows = ledger.collection(coll)
        except KeyError:
            continue
        counts: dict[Any, int] = {}
        for r in rows:
            v = get_path(r, path)
            if v is MISSING or v is None or isinstance(v, (list, dict)):
                continue
            counts[v] = counts.get(v, 0) + 1
        if counts:
            out[name] = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
    return out


@dataclass
class ValidationResult:
    template_id: str
    family: str
    ok: bool
    n_candidates: int
    bindings: dict
    gold_preview: Any = None
    gold_n: int | None = None
    note: str = ""
    problem: str | None = None

    def line(self) -> str:
        status = "OK " if self.ok else ("EMPTY" if self.problem is None else "FAIL ")
        tail = self.problem or self.note
        binds = ", ".join(f"{k}={v!r}" for k, v in self.bindings.items())
        return (f"  [{status}] {self.template_id:24s} candidates={self.n_candidates:<6d}"
                + (f" [{binds}]" if binds else "")
                + (f"  {tail}" if tail else ""))


def validate_template(tpl: Template, ledger) -> ValidationResult:
    """Resolve a template's selector and gold spec. Generates nothing."""
    binds = default_bindings(tpl, ledger)
    sel = substitute(tpl.selector, binds)
    source = sel.get("source", "deals")
    try:
        rows = select(ledger.collection(source), sel.get("where"))
    except (KeyError, QueryError, ValueError) as exc:
        return ValidationResult(tpl.id, tpl.family, False, 0, binds,
                                problem=f"{type(exc).__name__}: {exc}")

    n = len(rows)
    if n == 0:
        return ValidationResult(tpl.id, tpl.family, False, 0, binds,
                                note="selector matched no ledger rows")

    gold = substitute(tpl.gold, binds)
    try:
        if gold.get("aggregate"):
            res = aggregate(rows, gold["aggregate"])
            if res.value is None:
                return ValidationResult(tpl.id, tpl.family, False, n, binds,
                                        problem=f"gold not computable: {res.note}")
            return ValidationResult(tpl.id, tpl.family, True, n, binds,
                                    gold_preview=res.value, gold_n=res.n,
                                    note=f"gold={res.value!r} over n={res.n}")
        if gold.get("path"):
            path = gold["path"]
            have = sum(1 for r in rows if get_path(r, path) not in (MISSING, None))
            if have == 0:
                return ValidationResult(tpl.id, tpl.family, False, n, binds,
                                        problem=f"no candidate row has `{path}`")
            sample = next(get_path(r, path) for r in rows
                          if get_path(r, path) not in (MISSING, None))
            return ValidationResult(tpl.id, tpl.family, True, n, binds,
                                    gold_preview=sample, gold_n=have,
                                    note=f"gold `{path}` resolvable on {have}/{n}; e.g. {sample!r}")
        if gold.get("literal") is not None:
            return ValidationResult(tpl.id, tpl.family, True, n, binds,
                                    gold_preview=gold["literal"], gold_n=n,
                                    note="literal gold")
        if gold.get("value") is not None:
            # A fixed verdict label that is the same for every row the selector
            # matches -- how absence_and_abstention states its gold.
            return ValidationResult(tpl.id, tpl.family, True, n, binds,
                                    gold_preview=gold["value"], gold_n=n,
                                    note=f"fixed verdict gold {gold['value']!r}")
        if gold.get("checklist_ref") or gold.get("checks"):
            ref = gold.get("checklist_ref", "<inline>")
            return ValidationResult(tpl.id, tpl.family, True, n, binds,
                                    gold_preview=ref, gold_n=n,
                                    note=f"checklist gold ({ref})")
        return ValidationResult(tpl.id, tpl.family, False, n, binds,
                                problem="gold block has none of: aggregate/path/literal/checks")
    except (QueryError, ValueError, KeyError) as exc:
        return ValidationResult(tpl.id, tpl.family, False, n, binds,
                                problem=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# QUESTION GENERATION -- INTENTIONALLY NOT IMPLEMENTED
# ---------------------------------------------------
# Generation is out of scope: the ledger these questions depend on is still
# being built, and generating against a moving ledger would bake in answers
# that stop being true.
#
# When the ledger is frozen, generation attaches HERE. Everything it needs
# already exists:
#
#   rows      = select(ledger.collection(sel['source']), sel['where'])
#   bindings  = per-row slot values via tpl.slots (row field -> slot)
#   question  = substitute(random.choice(tpl.questions), bindings)
#   gold      = resolve_gold(rows_for_this_item, substitute(tpl.gold, bindings))
#   item      = {item_id, family, question, answer_type, gold: {value, unit,
#                accept, trace: gold.as_trace()}, grading, meta}
#
# The graders already consume exactly that item shape -- it is the shape the
# fixtures use. Sampling policy (how many items per template, how to spread
# difficulty, how to avoid reusing the same deal) is a design decision for the
# person who owns the question design, which is why it is not guessed at here.
