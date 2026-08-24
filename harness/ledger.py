"""Normalized, read-only access to the fact ledger.

WHY AN ADAPTER RATHER THAN `json.load`
--------------------------------------
As of the night this was written the corpus repo carries TWO ledger schemas at
once:

  v3  a single `ledger-v3.json` (416 deals) with a sibling
      `holdout-targets.json` whose records live under the key `targets`
  v4  a sharded `ledger-v4/` directory (index.json + core/deals/holdout/
      implicit_rules/portfolio/halloran/constraints shards, 1670 deals) whose
      holdout records live under the key `holdout_targets`

The corpus build is in flight and will keep moving. Nothing above this module
should know or care which layout is on disk, so everything downstream --
templates, graders, the fabrication allowlist -- is written against ONE
normalized shape. Adding a v5 means adding a loader here and nothing else.

READ-ONLY, ALWAYS
-----------------
This module opens files for reading and never writes, never locks, never runs
git. The corpus repo is another agent's working tree. Loads are retried on
JSONDecodeError because a file can legitimately be mid-write when we read it.
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from .query import MISSING, aggregate, get_path, matches, select

__all__ = ["Ledger", "load_ledger", "LedgerUnavailable"]


class LedgerUnavailable(RuntimeError):
    """The configured ledger could not be read. Never swallowed."""


# --------------------------------------------------------------------------
# resilient read
# --------------------------------------------------------------------------
def _read_json(path: str, attempts: int = 4, pause: float = 0.35) -> Any:
    """Read JSON, tolerating a file that is being rewritten underneath us."""
    last = None
    for i in range(attempts):
        try:
            with io.open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            raise LedgerUnavailable(f"ledger file not found: {path}")
        except (json.JSONDecodeError, PermissionError, OSError) as exc:
            last = exc
            if i < attempts - 1:
                time.sleep(pause)
    raise LedgerUnavailable(
        f"could not read {path} after {attempts} attempts "
        f"(the corpus build may be rewriting it): {type(last).__name__}: {last}"
    )


def _parse_date(s: Any) -> date | None:
    if not isinstance(s, str) or len(s) < 10:
        return None
    try:
        return date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
# the normalized object
# --------------------------------------------------------------------------
@dataclass
class Ledger:
    """One shape, whatever the on-disk layout was."""

    schema: str = "unknown"          # 'v3' | 'v4' | 'normalized'
    source: str = ""                 # path it came from
    meta: dict = field(default_factory=dict)
    firms: dict = field(default_factory=dict)
    people: list = field(default_factory=list)
    funds: list = field(default_factory=list)
    deals: list = field(default_factory=list)
    hooks: list = field(default_factory=list)
    house_rules: list = field(default_factory=list)
    criteria_versions: list = field(default_factory=list)
    timeline: list = field(default_factory=list)
    canon: dict = field(default_factory=dict)
    implicit_rules: list = field(default_factory=list)
    holdout_targets: list = field(default_factory=list)
    portfolio_companies: list = field(default_factory=list)
    pc_periods: list = field(default_factory=list)
    clients: list = field(default_factory=list)
    pitches: list = field(default_factory=list)
    buyers: list = field(default_factory=list)
    analyst_never_signs: list = field(default_factory=list)
    stated_counts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # -- collection access ------------------------------------------------
    COLLECTIONS = (
        "deals", "people", "funds", "hooks", "house_rules", "criteria_versions",
        "timeline", "implicit_rules", "holdout_targets", "portfolio_companies",
        "pc_periods", "clients", "pitches", "buyers", "stated_counts",
    )

    def collection(self, name: str) -> list:
        if name not in self.COLLECTIONS:
            raise KeyError(
                f"unknown ledger collection {name!r}; available: {', '.join(self.COLLECTIONS)}"
            )
        return getattr(self, name)

    def select(self, source: str, where: dict | None = None) -> list[dict]:
        return select(self.collection(source), where)

    # -- convenience used by graders and the fabrication allowlist --------
    def deal_by_codename(self, codename: str) -> dict | None:
        cf = str(codename).strip().casefold()
        for d in self.deals:
            if str(d.get("codename", "")).casefold() == cf:
                return d
        return None

    def holdout_by_id(self, tid: str) -> dict | None:
        for t in self.holdout_targets:
            if t.get("id") == tid:
                return t
        return None

    def implicit_rule(self, rid: str) -> dict | None:
        for r in self.implicit_rules:
            if r.get("id") == rid:
                return r
        return None

    def frozen_facts(self) -> dict:
        return dict(self.canon.get("frozen_facts") or {})

    def written_screens(self) -> dict:
        return dict(self.frozen_facts().get("westbrook_screens") or {})

    def mandate(self) -> dict:
        return dict(self.frozen_facts().get("westbrook_mandate") or {})

    def firm_names(self) -> list[str]:
        return [f.get("name", "") for f in self.firms.values() if f.get("name")]

    def summary(self) -> dict:
        return {
            "schema": self.schema,
            "source": self.source,
            "version": (self.meta or {}).get("version"),
            "counts": {k: len(self.collection(k)) for k in self.COLLECTIONS},
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# derived fields
# --------------------------------------------------------------------------
def _derive_deals(deals: list[dict]) -> None:
    """Attach a `_derived` block to every deal.

    Templates reference these by dotted path exactly like native fields. They
    live under `_derived` so it is always obvious in a trace which numbers came
    off the ledger and which the harness computed.
    """
    for d in deals:
        der: dict[str, Any] = {}
        entry = _parse_date(d.get("date"))
        ex = d.get("exit") if isinstance(d.get("exit"), dict) else None
        if entry and ex:
            xd = _parse_date(ex.get("date"))
            if xd:
                days = (xd - entry).days
                der["hold_period_days"] = days
                der["hold_period_years"] = round(days / 365.25, 3)
                der["hold_period_months"] = round(days / 30.4375, 2)
        if ex:
            for k in ("multiple", "moic", "ebitda_at_exit", "buyer_type"):
                if ex.get(k) is not None:
                    der[f"exit_{k}"] = ex[k]
        der["has_exit"] = bool(ex)
        der["is_pass"] = str(d.get("outcome", "")).casefold() == "pass"
        der["is_screen"] = str(d.get("stage", "")).casefold() == "screen"
        der["is_thin"] = str(d.get("tier", "")).casefold() == "thin"

        nums = d.get("numbers") or {}
        rev, eb = nums.get("revenue_m"), nums.get("adj_ebitda_m")
        try:
            if rev not in (None, 0) and eb is not None:
                der["implied_margin_pct"] = round(100.0 * float(eb) / float(rev), 2)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        em = nums.get("entry_multiple")
        try:
            if em is not None and eb is not None:
                der["implied_ev_m"] = round(float(em) * float(eb), 2)
        except (TypeError, ValueError):
            pass
        d["_derived"] = der


def _derive_portfolio(companies: list[dict], periods: list[dict]) -> None:
    """Attach each portfolio company's CURRENT operating figures.

    `pc_periods` is a quarterly time series; the company row carries only entry
    figures. Without this there is no ledger-side answer to "what is X's
    retention now", and the recency slice -- a fact two documents disagree on
    where a later one supersedes an earlier one -- has no gold to check against.

    `_derived.current.<field>` is the value at the company's LATEST recorded
    quarter, and `_derived.latest_quarter` names which quarter that is, so the
    gold is always attributable rather than floating.
    """
    if not companies or not periods:
        return
    by_pc: dict[str, list[dict]] = {}
    for row in periods:
        pid = row.get("pc_id")
        if pid:
            by_pc.setdefault(pid, []).append(row)
    for pc in companies:
        ser = sorted(by_pc.get(pc.get("id"), []), key=lambda r: str(r.get("quarter") or ""))
        if not ser:
            continue
        last = ser[-1]
        d = pc.setdefault("_derived", {})
        d["latest_quarter"] = last.get("quarter")
        d["n_quarters"] = len(ser)
        d["first_quarter"] = ser[0].get("quarter")
        cur = {k: v for k, v in last.items() if k not in ("pc_id", "quarter")}
        d["current"] = cur


def _derive_entity_keys(led) -> None:
    """One stable key per real-world entity, across collections.

    `portfolio_companies` and `deals` describe the SAME eleven companies from
    two angles, and `clients` describes the same mandates as the Halloran deal
    rows. Keyed by their own row ids they look like disjoint populations, so an
    aggregate over the portfolio and an aggregate over the platform deals would
    both be admitted as independent while being the same eleven companies asked
    two ways -- exactly the duplication the overlap rule exists to catch.

    Row ids stay untouched (traces still name the row that was read); this adds
    a parallel key the overlap measurement uses.
    """
    for d in led.deals or []:
        cn = d.get("codename")
        if cn:
            d.setdefault("_derived", {})["entity_key"] = f"deal:{cn}"
    for pc in led.portfolio_companies or []:
        cn = pc.get("deal_codename")
        if cn:
            pc.setdefault("_derived", {})["entity_key"] = f"deal:{cn}"
    by_mandate = {d.get("id"): d.get("codename") for d in (led.deals or [])}
    for c in led.clients or []:
        cn = by_mandate.get(c.get("mandate_id"))
        if cn:
            c.setdefault("_derived", {})["entity_key"] = f"deal:{cn}"
    # A company-quarter is a finer entity than the company: an aggregate over
    # one company's quarters is a different question from one over companies.
    pc_name = {pc.get("id"): pc.get("deal_codename") for pc in (led.portfolio_companies or [])}
    for r in led.pc_periods or []:
        cn = pc_name.get(r.get("pc_id"))
        if cn and r.get("quarter"):
            r.setdefault("_derived", {})["entity_key"] = f"deal:{cn}@{r['quarter']}"


def _derive_holdouts(targets: list[dict]) -> None:
    """Attach `_derived` to holdout targets.

    `discriminates` is the field `implicit_rule_application` reports on: it marks
    targets where the written criteria and the firm's actual behaviour give
    different answers. We recompute it rather than trusting the flag, and warn
    if the two disagree.
    """
    for t in targets:
        w = str(t.get("written_criteria_verdict", "")).strip().casefold()
        g = str(t.get("ground_truth_outcome", "")).strip().casefold()
        der = {
            "written_verdict": w,
            "gold_verdict": g,
            "computed_discriminates": bool(w and g and w != g),
            "declared_discriminates": bool(t.get("discriminates")),
        }
        der["flag_mismatch"] = der["computed_discriminates"] != der["declared_discriminates"]
        t["_derived"] = der


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def _load_v3(path: str, holdout_path: str | None) -> Ledger:
    raw = _read_json(path)
    led = Ledger(schema="v3", source=os.path.abspath(path))
    led.meta = raw.get("meta") or {}
    led.firms = raw.get("firms") or {}
    for k in ("people", "funds", "deals", "hooks", "house_rules",
              "criteria_versions", "timeline"):
        setattr(led, k, list(raw.get(k) or []))
    led.canon = raw.get("canon") or {}

    # v3 keeps holdout targets in a sibling file under the key `targets`
    if holdout_path is None:
        guess = os.path.join(os.path.dirname(os.path.abspath(path)), "holdout-targets.json")
        holdout_path = guess if os.path.exists(guess) else None
    if holdout_path and os.path.exists(holdout_path):
        hraw = _read_json(holdout_path)
        if isinstance(hraw, dict):
            led.holdout_targets = list(
                hraw.get("holdout_targets") or hraw.get("targets") or []
            )
        elif isinstance(hraw, list):
            led.holdout_targets = list(hraw)
    else:
        led.warnings.append(
            "no holdout targets file found next to the v3 ledger; "
            "implicit_rule_application has no gold source"
        )
    return led


_V4_SHARD_KEYS = {
    "core": ("meta", "firms", "people", "funds", "hooks", "house_rules",
             "criteria_versions", "timeline", "canon", "analyst_never_signs"),
    "implicit_rules": ("implicit_rules",),
    "deals": ("deals",),
    "holdout": ("holdout_targets",),
    "portfolio": ("portfolio_companies", "pc_periods"),
    "halloran": ("clients", "pitches", "buyers"),
    "constraints": ("stated_counts",),
}


def _load_v4(dirpath: str) -> Ledger:
    led = Ledger(schema="v4", source=os.path.abspath(dirpath))
    index_path = os.path.join(dirpath, "index.json")
    shards: Iterable[str]
    if os.path.exists(index_path):
        idx = _read_json(index_path)
        led.meta = {"version": idx.get("version"), "seed": idx.get("seed")}
        shards = list((idx.get("shards") or {}).keys())
    else:
        led.warnings.append("ledger-v4/index.json absent; discovering shards by filename")
        shards = [os.path.splitext(f)[0] for f in os.listdir(dirpath) if f.endswith(".json")]

    for shard in shards:
        if shard == "index":
            continue
        spath = os.path.join(dirpath, f"{shard}.json")
        if not os.path.exists(spath):
            led.warnings.append(f"shard declared in index but missing on disk: {shard}.json")
            continue
        try:
            raw = _read_json(spath)
        except LedgerUnavailable as exc:
            led.warnings.append(f"shard unreadable, skipped: {shard}.json ({exc})")
            continue
        if not isinstance(raw, dict):
            led.warnings.append(f"shard is not an object, skipped: {shard}.json")
            continue
        for key, val in raw.items():
            if key == "meta":
                led.meta = {**(val or {}), **led.meta}
                continue
            if key == "firms":
                led.firms = val or {}
                continue
            if key == "canon":
                led.canon = val or {}
                continue
            if hasattr(led, key) and isinstance(getattr(led, key), list):
                setattr(led, key, list(val or []))
    return led


def _load_normalized(path: str) -> Ledger:
    """Load a ledger already written in the normalized shape (fixtures use this)."""
    raw = _read_json(path)
    led = Ledger(schema=raw.get("schema", "normalized"), source=os.path.abspath(path))
    led.meta = raw.get("meta") or {}
    led.firms = raw.get("firms") or {}
    led.canon = raw.get("canon") or {}
    for k in Ledger.COLLECTIONS:
        setattr(led, k, list(raw.get(k) or []))
    led.analyst_never_signs = list(raw.get("analyst_never_signs") or [])
    return led


def detect_schema(path: str) -> str:
    if os.path.isdir(path):
        return "v4"
    base = os.path.basename(path).casefold()
    if "ledger-v3" in base:
        return "v3"
    try:
        head = _read_json(path)
    except LedgerUnavailable:
        raise
    if isinstance(head, dict):
        if "schema" in head or ("deals" in head and "holdout_targets" in head and "canon" in head):
            return "normalized"
        if "canon" in head and "deals" in head:
            return "v3"
    return "normalized"


def load_ledger(path: str, schema: str = "auto", holdout_path: str | None = None) -> Ledger:
    """Load and normalize a ledger from disk. Read-only.

    `schema` is 'auto' | 'v3' | 'v4' | 'normalized'.
    """
    if not os.path.exists(path):
        raise LedgerUnavailable(
            f"ledger path does not exist: {path}\n"
            f"Point config/ledger.yaml at a real ledger, or run against "
            f"fixtures/mini_ledger.json which is self-contained."
        )
    kind = detect_schema(path) if schema in (None, "", "auto") else schema
    if kind == "v3":
        led = _load_v3(path, holdout_path)
    elif kind == "v4":
        led = _load_v4(path)
    elif kind == "normalized":
        led = _load_normalized(path)
    else:
        raise LedgerUnavailable(f"unknown ledger schema {schema!r}")

    _derive_deals(led.deals)
    _derive_holdouts(led.holdout_targets)
    _derive_portfolio(led.portfolio_companies, led.pc_periods)
    _derive_entity_keys(led)

    mismatched = [t["id"] for t in led.holdout_targets
                  if t.get("_derived", {}).get("flag_mismatch")]
    if mismatched:
        led.warnings.append(
            f"{len(mismatched)} holdout target(s) whose `discriminates` flag disagrees with "
            f"written_criteria_verdict vs ground_truth_outcome: {mismatched[:10]}"
        )
    if not led.deals:
        led.warnings.append("ledger contains zero deals")
    return led
