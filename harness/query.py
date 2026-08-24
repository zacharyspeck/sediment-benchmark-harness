"""A tiny declarative query/aggregation layer over normalized ledger rows.

WHY THIS EXISTS
---------------
Question templates are data, not code. A template says which rows it draws
from and how the gold answer is computed; this module is what executes that
description. The point is that revising a question -- changing a filter, a
sector, an aggregation -- is a YAML edit, never a Python edit.

Filters (`where`) are a mapping of dotted field path -> condition:

    sector: healthcare services            # shorthand for equality
    firm: {op: in, values: [westbrook]}
    year: {op: between, lo: 2023, hi: 2023}
    numbers.entry_multiple: {op: not_null}
    outcome: {op: ne, value: pass}
    codename: {op: regex, value: "^K"}
    kill_code: {op: exists}
    sector: {op: contains, value: health}   # substring, case-insensitive

Aggregations (`gold.aggregate`):

    {op: count}
    {op: mean,   field: numbers.adj_ebitda_margin_pct, round: 1}
    {op: median, field: _derived.hold_period_years,    round: 2}
    {op: sum | min | max | count_distinct | pct_of, ...}

Every aggregation returns both the value and the row ids that produced it, so
a disputed gold answer can be audited down to the individual deals.
"""

from __future__ import annotations

import re
import statistics
# `field` is imported under an alias because AggResult below has an attribute
# literally named `field` (the ledger path an aggregation ran over). Without the
# alias, that attribute shadows dataclasses.field inside the class body and the
# next default_factory call resolves to None.
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Iterable, Sequence

MISSING = object()


class QueryError(ValueError):
    """Raised for a malformed selector/aggregation. Never swallowed."""


# --------------------------------------------------------------------------
# path access
# --------------------------------------------------------------------------
def get_path(row: Any, path: str, default=MISSING):
    """Resolve a dotted path. Returns `default` (MISSING) when absent."""
    cur = row
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return cur


def has_path(row: Any, path: str) -> bool:
    return get_path(row, path) is not MISSING


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
def _norm(v):
    return v.strip().casefold() if isinstance(v, str) else v


def _cmp_numeric(a, b, fn) -> bool:
    try:
        return fn(float(a), float(b))
    except (TypeError, ValueError):
        return False


def _match_one(value, cond) -> bool:
    if not isinstance(cond, dict) or "op" not in cond:
        # shorthand equality (list shorthand = membership)
        if isinstance(cond, list):
            return _norm(value) in [_norm(c) for c in cond]
        return _norm(value) == _norm(cond)

    op = str(cond["op"]).lower()

    if op == "exists":
        want = cond.get("value", True)
        return (value is not MISSING) == bool(want)
    if op == "is_null":
        # An absent field and an explicit null mean the same thing to a
        # template author ("this deal has no entry multiple"). The ledger uses
        # both -- thin screening records omit the key entirely while thick ones
        # carry an explicit null -- so is_null must cover both or half the
        # population silently disappears from the selector.
        return value is MISSING or value is None
    if value is MISSING:
        # every other operator is false against an absent field
        return op in ("not_exists",)
    if op == "not_exists":
        return False
    if op == "eq":
        return _norm(value) == _norm(cond["value"])
    if op == "ne":
        return _norm(value) != _norm(cond["value"])
    if op == "in":
        return _norm(value) in [_norm(v) for v in cond["values"]]
    if op == "not_in":
        return _norm(value) not in [_norm(v) for v in cond["values"]]
    if op == "not_null":
        return value is not None
    if op == "gt":
        return _cmp_numeric(value, cond["value"], lambda a, b: a > b)
    if op == "gte":
        return _cmp_numeric(value, cond["value"], lambda a, b: a >= b)
    if op == "lt":
        return _cmp_numeric(value, cond["value"], lambda a, b: a < b)
    if op == "lte":
        return _cmp_numeric(value, cond["value"], lambda a, b: a <= b)
    if op == "between":
        return _cmp_numeric(value, cond["lo"], lambda a, b: a >= b) and _cmp_numeric(
            value, cond["hi"], lambda a, b: a <= b
        )
    if op == "contains":
        return isinstance(value, str) and str(cond["value"]).casefold() in value.casefold()
    if op == "regex":
        return isinstance(value, str) and re.search(str(cond["value"]), value) is not None
    if op == "list_contains":
        return isinstance(value, (list, tuple)) and any(
            _norm(x) == _norm(cond["value"]) for x in value
        )
    if op == "list_len":
        return isinstance(value, (list, tuple)) and len(value) == int(cond["value"])
    if op == "truthy":
        return bool(value) == bool(cond.get("value", True))
    raise QueryError(f"unknown filter op: {op!r}")


def matches(row: dict, where: dict | None) -> bool:
    if not where:
        return True
    for path, cond in where.items():
        if not _match_one(get_path(row, path), cond):
            return False
    return True


def select(rows: Iterable[dict], where: dict | None = None) -> list[dict]:
    return [r for r in rows if matches(r, where)]


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
@dataclass
class AggResult:
    op: str
    value: Any
    n: int
    field: str | None = None
    contributing_ids: list = dc_field(default_factory=list)
    contributing_values: list = dc_field(default_factory=list)
    note: str = ""

    def as_trace(self) -> dict:
        return {
            "op": self.op,
            "field": self.field,
            "value": self.value,
            "n": self.n,
            "contributing_ids": self.contributing_ids[:200],
            "contributing_values": self.contributing_values[:200],
            "truncated": len(self.contributing_ids) > 200,
            "note": self.note,
        }


_ID_KEYS = ("id", "codename", "name")

#: Collections whose rows carry no single id field but ARE uniquely keyed by a
#: combination. Without this every `pc_periods` row returns "<no-id>", which
#: collapses 199 distinct quarters into one indistinguishable element: every
#: aggregate over them becomes the same one-element set, the trace lists 199
#: identical ids, and the overlap measurement silently reads them as duplicates.
_COMPOSITE_ID_KEYS = (
    ("pc_id", "quarter"),
    ("company", "quarter"),
)


def _row_id(row: dict) -> str:
    for k in _ID_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    for combo in _COMPOSITE_ID_KEYS:
        if all(row.get(k) not in (None, "") for k in combo):
            return ":".join(str(row[k]) for k in combo)
    return "<no-id>"


def aggregate(rows: Sequence[dict], spec: dict) -> AggResult:
    """Execute an aggregation spec against already-filtered rows."""
    if not isinstance(spec, dict) or "op" not in spec:
        raise QueryError(f"aggregate spec needs an 'op': {spec!r}")
    op = str(spec["op"]).lower()
    fld = spec.get("field")
    rnd = spec.get("round")

    if op == "count":
        return AggResult(op, len(rows), len(rows), None, [_row_id(r) for r in rows])

    if op == "count_distinct":
        if not fld:
            raise QueryError("count_distinct requires 'field'")
        vals = [get_path(r, fld) for r in rows]
        vals = [v for v in vals if v is not MISSING and v is not None]
        uniq = sorted({str(v) for v in vals})
        return AggResult(op, len(uniq), len(vals), fld, [_row_id(r) for r in rows], uniq)

    if op in ("mean", "median", "sum", "min", "max", "stdev", "pct_of"):
        if not fld:
            raise QueryError(f"{op} requires 'field'")
        pairs = []
        for r in rows:
            v = get_path(r, fld)
            if v is MISSING or v is None:
                continue
            try:
                pairs.append((_row_id(r), float(v)))
            except (TypeError, ValueError):
                continue
        vals = [v for _, v in pairs]
        ids = [i for i, _ in pairs]
        if not vals:
            return AggResult(op, None, 0, fld, [], [], note="no non-null numeric values matched")
        if op == "mean":
            val = statistics.fmean(vals)
        elif op == "median":
            val = statistics.median(vals)
        elif op == "sum":
            val = sum(vals)
        elif op == "min":
            val = min(vals)
        elif op == "max":
            val = max(vals)
        elif op == "stdev":
            val = statistics.stdev(vals) if len(vals) > 1 else 0.0
        else:  # pct_of
            denom = float(spec.get("denominator", 0) or 0)
            if denom == 0:
                raise QueryError("pct_of requires a non-zero 'denominator'")
            val = 100.0 * sum(vals) / denom
        if rnd is not None:
            val = round(val, int(rnd))
        return AggResult(op, val, len(vals), fld, ids, vals)

    if op == "list":
        if not fld:
            raise QueryError("list requires 'field'")
        vals = []
        for r in rows:
            v = get_path(r, fld)
            if v is MISSING or v is None:
                continue
            vals.append(v)
        if spec.get("sort", True):
            vals = sorted(vals, key=lambda x: str(x))
        if spec.get("distinct", True):
            seen, out = set(), []
            for v in vals:
                k = str(v).casefold()
                if k not in seen:
                    seen.add(k)
                    out.append(v)
            vals = out
        return AggResult(op, vals, len(vals), fld, [_row_id(r) for r in rows], vals)

    if op in ("mode", "group_counts"):
        # "Which sector do we pass on most?" -- a corpus-wide question that
        # needs a group-by, not a numeric reduction.
        if not fld:
            raise QueryError(f"{op} requires 'field'")
        counts: dict[str, int] = {}
        for r in rows:
            v = get_path(r, fld)
            if v is MISSING or v is None:
                continue
            key = str(v)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return AggResult(op, None, 0, fld, [], [], note="no non-null values matched")
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if op == "group_counts":
            return AggResult(op, dict(ordered), len(rows), fld,
                             [_row_id(r) for r in rows], ordered)
        top_n = ordered[0][1]
        tied = [k for k, c in ordered if c == top_n]
        return AggResult(
            op, ordered[0][0], len(rows), fld, [_row_id(r) for r in rows], ordered,
            note=(f"tie between {tied}" if len(tied) > 1 else f"{ordered[0][0]!r} with {top_n}"),
        )

    if op == "value":
        # single-row scalar lookup; errors loudly if the filter is not unique
        if not fld:
            raise QueryError("value requires 'field'")
        if len(rows) != 1:
            return AggResult(
                op, None, len(rows), fld, [_row_id(r) for r in rows], [],
                note=f"selector matched {len(rows)} rows; 'value' needs exactly 1",
            )
        v = get_path(rows[0], fld, None)
        if isinstance(v, float) and rnd is not None:
            v = round(v, int(rnd))
        return AggResult(op, v, 1, fld, [_row_id(rows[0])], [v])

    raise QueryError(f"unknown aggregate op: {op!r}")


def resolve_gold(rows: Sequence[dict], gold_spec: dict) -> AggResult:
    """gold_spec is either {aggregate: {...}} or {path: 'a.b'} on a unique row."""
    if "aggregate" in gold_spec:
        return aggregate(rows, gold_spec["aggregate"])
    if "path" in gold_spec:
        return aggregate(rows, {"op": "value", "field": gold_spec["path"], "round": gold_spec.get("round")})
    if "literal" in gold_spec:
        return AggResult("literal", gold_spec["literal"], 1, None, [], [gold_spec["literal"]])
    raise QueryError(f"gold spec needs 'aggregate', 'path' or 'literal': {gold_spec!r}")
