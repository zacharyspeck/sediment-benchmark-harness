"""The fabrication scorer. Runs across every family, independent of correctness.

For each response: extract every capitalized entity, deal codename, person name
and dollar/multiple figure from the prose; check each against a ledger
allowlist; anything not in the ledger and not in the stoplist is a candidate
fabrication. Every flag is logged with its surrounding sentence for manual
audit, and the report gives a fabrication rate per arm per family.

This number matters as much as accuracy. A buyer's real fear is invented deal
history -- a system that is 5 points more accurate but invents a portfolio
company is not the better system.

TWO TIERS, REPORTED SEPARATELY
------------------------------
Tier A -- ENTITY fabrication (the headline).
    A deal codename, person, firm or portfolio company that does not exist in
    the ledger. High precision: entities are closed-world. If the model says
    "Project Vellvale" and no such deal exists, that is an invention, full stop.

Tier B -- UNGROUNDED FIGURE (advisory).
    A dollar amount or multiple that matches no ledger figure. Deliberately
    NOT folded into the headline rate, because this family of benchmark asks
    models to compute things -- an average margin, a median hold period -- and
    a correctly derived statistic is a number that appears nowhere in the
    ledger. Counting that as fabrication would punish the right answer. Tier B
    is high-recall and is meant to be read with the trace open, not quoted.

Reporting them as one number would be the easiest way to produce an impressive
and dishonest chart.

PER-ITEM ALLOWLISTING
---------------------
Whatever the question itself supplies is allowlisted for that item. Family 4
hands the model a holdout target's facts in the prompt; echoing those figures
back is quotation, not invention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from harness.extractor import split_sentences

__all__ = ["FabricationScorer", "FabricationReport", "FabricationFlag", "DEFAULT_STOPLIST"]

# --------------------------------------------------------------------------
# defaults (config/fabrication.yaml overrides/extends all of these)
# --------------------------------------------------------------------------
DEFAULT_STOPLIST = {
    # The first-person pronoun. Capitalised mid-sentence by definition, so
    # without this every "which I have rounded to" is a proper-noun flag.
    # It also rescues "Fund I" / "Fund I's", where the roman numeral is a real
    # part of a real entity name.
    # KNOWN COST: an invented "Fund IV" is not caught, because every token in
    # it is accounted for. Judged worth it -- the alternative fires on ordinary
    # English in nearly every response, and a detector nobody trusts is worse
    # than one with a documented blind spot.
    "i", "ii", "iii", "iv",
    # sentence furniture / pronouns / determiners
    "the", "this", "that", "these", "those", "a", "an", "we", "our", "us", "it", "its",
    "they", "their", "he", "she", "his", "her", "there", "here", "if", "when", "while",
    "and", "or", "but", "so", "because", "however", "although", "though", "both",
    "no", "not", "none", "nothing", "any", "all", "each", "every", "some", "such",
    "yes", "answer", "note", "notes", "summary", "conclusion", "background", "overview",
    "question", "response", "reasoning", "source", "sources", "caveat", "assumption",
    "assumptions", "method", "methodology", "step", "steps", "first", "second", "third",
    "fourth", "fifth", "finally", "next", "then", "also", "in", "on", "at", "to", "for",
    "from", "with", "without", "under", "over", "above", "below", "between", "across",
    "per", "via", "by", "as", "of", "into", "than", "about", "after", "before", "during",
    # calendar
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "q1", "q2", "q3", "q4", "fy", "ytd", "ltm", "ttm", "cagr", "yoy",
    # ordinary finance / deal vocabulary
    "ebitda", "ebit", "ebitdar", "revenue", "revenues", "margin", "margins", "adjusted",
    "adj", "multiple", "multiples", "enterprise", "value", "equity", "debt", "leverage",
    "cash", "flow", "capex", "opex", "working", "capital", "gross", "net", "operating",
    "income", "earnings", "profit", "loss", "balance", "sheet", "statement",
    "loi", "ioi", "cim", "msa", "nda", "ic", "qoe", "moic", "irr", "dpi", "tvpi", "npv",
    "lp", "lps", "gp", "gps", "pe", "vc", "ma", "mna", "roll", "up", "addon", "add",
    "platform", "diligence", "screening", "screen", "memo", "memorandum", "mandate",
    "process", "auction", "buyer", "buyers", "seller", "sellers", "owner", "owners",
    "founder", "founders", "management", "team", "board", "committee", "investment",
    "criteria", "criterion", "thesis", "portfolio", "company", "companies", "business",
    "services", "manufacturing", "healthcare", "consumer", "products", "software",
    "saas", "vertical", "industrial", "specialty", "distribution", "recurring",
    "retention", "churn", "concentration", "customer", "customers", "contract",
    "contracts", "contracted", "dependence", "rollover", "close", "closed", "closing",
    "exit", "exits", "exited", "hold", "period", "fund", "funds", "vintage",
    "sector", "sectors", "deal", "deals", "target", "targets", "pass", "passed",
    "advance", "advanced", "walk", "walked", "withdrawn", "lost", "pipeline",
    "strategic", "sponsor", "financial", "control", "minority", "majority",
    "us", "u.s.", "usa", "united", "states", "american", "north", "america",
    "midwest", "northeast", "southeast", "southwest", "west", "east", "south", "central",
    "gaap", "sec", "irs", "phase", "environmental", "insurance", "loss", "run",
    "quality", "restatement", "restated", "actuals", "projections", "projection",
    "forecast", "budget", "plan", "growth", "organic", "inorganic", "tuck",
    "million", "billion", "thousand", "percent", "percentage", "points", "basis",
    "true", "false", "n", "a", "na", "tbd", "unknown", "unclear", "insufficient",
    # roles / titles
    "managing", "partner", "partners", "principal", "vice", "president", "associate",
    "analyst", "director", "chief", "officer", "cfo", "ceo", "coo", "cto", "md", "vp",
    "co", "cofounder", "founder",
}

# Kinds of figure Tier B looks at.
_MONEY_RE = re.compile(
    r"\$\s?(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?(?P<mag>[MmBbKk]|million|billion|thousand)?\b"
)
_MULTIPLE_RE = re.compile(r"(?<![\w.])(?P<num>\d{1,3}(?:\.\d+)?)\s?(?:x|×)(?![\w])", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(?<![\w.])(?P<num>\d{1,3}(?:\.\d+)?)\s?%")

# Entity surfaces.
_PROJECT_RE = re.compile(r"\b(?:Project|Deal|Target|Codename)\s+(?P<name>[A-Z][\w'\-]*)")

# Runs are joined by spaces/tabs only, never newlines. Allowing `\s+` let a run
# swallow a paragraph break -- "Martin Ashcbourne\n\nThe" became a single
# three-token "entity", which both mangled the reported surface and defeated the
# two-token person-name test.
_CAP_RUN_RE = re.compile(
    r"\b[A-Z][\w'&\-]*(?:[ \t]+(?:of|and|the|&|de|van|von)[ \t]+[A-Z][\w'&\-]*"
    r"|[ \t]+[A-Z][\w'&\-]*)*"
)
_ACRONYM_RE = re.compile(r"^[A-Z0-9&.\-]{1,6}$")


@dataclass
class FabricationFlag:
    kind: str                 # deal_codename | person_name | proper_noun | money | multiple | percent
    tier: str                 # A (entity) | B (figure)
    surface: str
    reason: str
    sentence: str
    start: int
    end: int
    nearest_known: str | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "tier": self.tier, "surface": self.surface,
            "reason": self.reason, "sentence": self.sentence,
            "start": self.start, "end": self.end,
            "nearest_known": self.nearest_known,
        }


@dataclass
class FabricationReport:
    flags: list[FabricationFlag] = field(default_factory=list)
    n_entities_checked: int = 0
    n_figures_checked: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def tier_a(self) -> list[FabricationFlag]:
        return [f for f in self.flags if f.tier == "A"]

    @property
    def tier_b(self) -> list[FabricationFlag]:
        return [f for f in self.flags if f.tier == "B"]

    @property
    def fabricated(self) -> bool:
        """Headline definition: at least one Tier A entity flag."""
        return bool(self.tier_a)

    def to_dict(self) -> dict:
        return {
            "fabricated": self.fabricated,
            "tier_a_count": len(self.tier_a),
            "tier_b_count": len(self.tier_b),
            "entities_checked": self.n_entities_checked,
            "figures_checked": self.n_figures_checked,
            "flags": [f.to_dict() for f in self.flags],
            "notes": list(self.notes),
        }


_POSSESSIVE = re.compile(r"[’']s\b")


def _norm(s: str) -> str:
    """Normalize a surface form for allowlist comparison.

    Strips the possessive, because "Cedacombe's" and "Fund I's" are the same
    entities as "Cedacombe" and "Fund I". Tuned against the fixtures: both of
    those were false positives before the possessive was handled, and a
    fabrication detector that fires on ordinary English is one nobody reads.
    """
    t = re.sub(r"\s+", " ", str(s or "")).strip().replace("’", "'")
    t = _POSSESSIVE.sub("", t)
    return t.casefold().strip(".,;:'\"")


def _is_acronym(tok: str) -> bool:
    """ALLCAPS, 2-6 chars, optionally pluralised: OEM, OEMs, MSA, LOI, L.P.

    Skipped rather than flagged. An invented acronym is not the failure this
    scorer exists to catch, and industry vocabulary is unbounded -- no
    allowlist will ever contain every one a finance document uses.
    """
    core = tok[:-1] if (len(tok) > 2 and tok.endswith("s")) else tok
    core = core.replace(".", "").replace("-", "").replace("&", "")
    return 2 <= len(core) <= 6 and core.isupper() and core.isalnum()


def _fmt_num(v: float) -> str:
    return f"{v:.10g}"


class FabricationScorer:
    """Builds a ledger allowlist once, then scores many responses against it."""

    def __init__(self, ledger: Any = None, config: dict | None = None):
        cfg = dict(config or {})
        self.config = cfg
        self.ledger = ledger

        self.stoplist = set(DEFAULT_STOPLIST)
        self.stoplist.update(_norm(w) for w in (cfg.get("stoplist_add") or []))
        for w in (cfg.get("stoplist_remove") or []):
            self.stoplist.discard(_norm(w))

        self.money_tol = float(cfg.get("money_tolerance", 0.051))
        self.multiple_tol = float(cfg.get("multiple_tolerance", 0.051))
        self.percent_tol = float(cfg.get("percent_tolerance", 0.051))
        self.check_percent = bool(cfg.get("check_percent", True))
        self.min_token_len = int(cfg.get("min_token_len", 2))

        self.entities: set[str] = set()
        self.entity_tokens: set[str] = set()
        self.money: set[float] = set()
        self.multiples: set[float] = set()
        self.percents: set[float] = set()
        self._build_allowlist(ledger)
        for extra in (cfg.get("allowlist_add") or []):
            self._add_entity(str(extra))

    # ------------------------------------------------------------------
    def _add_entity(self, name: Any) -> None:
        if not isinstance(name, str):
            return
        n = _norm(name)
        if not n:
            return
        self.entities.add(n)
        for tok in re.split(r"[\s/,&\-]+", n):
            t = tok.strip(".'\"")
            if len(t) >= self.min_token_len:
                self.entity_tokens.add(t)

    def _add_number(self, value: Any, kind: str) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        bucket = {"money": self.money, "multiple": self.multiples, "percent": self.percents}[kind]
        bucket.add(round(v, 4))
        bucket.add(round(v, 1))
        bucket.add(round(v, 0))

    def _build_allowlist(self, led) -> None:
        if led is None:
            return

        for f in (led.firms or {}).values():
            self._add_entity(f.get("name"))
        for p in led.people or []:
            self._add_entity(p.get("name"))
            self._add_entity(p.get("title"))
        for fu in led.funds or []:
            self._add_entity(fu.get("name"))
            self._add_number(fu.get("size_m"), "money")
        for pc in led.portfolio_companies or []:
            self._add_entity(pc.get("name"))
            self._add_entity(pc.get("deal_codename"))
            self._add_entity(pc.get("sector"))
            self._add_number(pc.get("entry_revenue_m"), "money")
            self._add_number(pc.get("entry_margin_pct"), "percent")
        for h in led.hooks or []:
            self._add_entity(h.get("id"))
            self._add_entity(h.get("origin_deal"))
        for r in led.implicit_rules or []:
            self._add_entity(r.get("id"))
        for c in (led.clients or []) + (led.buyers or []) + (led.pitches or []):
            if isinstance(c, dict):
                for k in ("name", "codename", "buyer_name", "client_name"):
                    self._add_entity(c.get(k))

        for d in led.deals or []:
            self._add_entity(d.get("codename"))
            self._add_entity(d.get("sector"))
            self._add_entity(d.get("kill_code"))
            for rel in (d.get("related") or []):
                self._add_entity(rel)
            self._add_entity(d.get("parent"))
            nums = d.get("numbers") or {}
            for k in ("revenue_m", "adj_ebitda_m", "entry_ev_m"):
                self._add_number(nums.get(k), "money")
            self._add_number(nums.get("entry_multiple"), "multiple")
            for k in ("adj_ebitda_margin_pct", "top_customer_pct", "top3_customer_pct",
                      "gross_logo_retention_pct", "logo_churn_pct", "growth_pct",
                      "recurring_pct", "capex_pct_revenue"):
                self._add_number(nums.get(k), "percent")
            ex = d.get("exit")
            if isinstance(ex, dict):
                self._add_number(ex.get("multiple"), "multiple")
                self._add_number(ex.get("moic"), "multiple")
                self._add_number(ex.get("ebitda_at_exit"), "money")
                self._add_entity(ex.get("buyer_type"))
            lm = d.get("loi_multiple")
            if lm is not None:
                self._add_number(lm, "multiple")

        for t in led.holdout_targets or []:
            self._add_entity(t.get("id"))
            self._add_entity(t.get("sector"))
            for k in ("revenue_m", "adj_ebitda_m"):
                self._add_number(t.get(k), "money")
            for k in ("adj_ebitda_margin_pct", "top_customer_pct", "top3_customer_pct",
                      "gross_logo_retention_pct", "logo_churn_pct", "growth_pct",
                      "recurring_pct", "capex_pct_revenue"):
                self._add_number(t.get(k), "percent")
            for k in ("owner_expectation_multiple", "indicative_high_multiple"):
                self._add_number(t.get(k), "multiple")

        for pcp in led.pc_periods or []:
            for k, v in pcp.items():
                if k.endswith("_m"):
                    self._add_number(v, "money")
                elif k.endswith("_pct"):
                    self._add_number(v, "percent")
                elif k.endswith("_x"):
                    self._add_number(v, "multiple")

        facts = led.frozen_facts() if hasattr(led, "frozen_facts") else {}
        mand = facts.get("westbrook_mandate") or {}
        for s in (mand.get("sectors_in") or []) + (mand.get("sectors_out") or []):
            self._add_entity(s)
        for v in (mand.get("revenue_range_m") or []):
            self._add_number(v, "money")
        self._add_number(mand.get("min_adj_ebitda_m"), "money")
        for code in (facts.get("westbrook_screens") or {}):
            self._add_entity(code)

    # ------------------------------------------------------------------
    def _known_entity(self, surface: str, extra: set[str] | None = None) -> bool:
        """Is this surface accounted for by the ledger, the stoplist, or the item?

        `extra` carries the per-item allowlist (entities the question itself
        supplied). It is consulted at BOTH the whole-surface and the token
        level -- otherwise allowlisting "Bracholt" would still leave
        "Project Bracholt" flagged, which is the same claim wearing a prefix.
        """
        extra = extra or set()
        n = _norm(surface)
        if not n:
            return True
        if n in self.entities or n in self.stoplist or n in extra:
            return True
        toks = [t for t in re.split(r"[\s/,&\-]+", n) if t]
        if not toks:
            return True
        # every token individually accounted for => not an invention
        return all(
            t in self.entity_tokens or t in self.stoplist or t in extra for t in toks
        )

    def _nearest(self, value: float, pool: Iterable[float]) -> str | None:
        best, bestd = None, None
        for v in pool:
            d = abs(v - value)
            if bestd is None or d < bestd:
                best, bestd = v, d
        return _fmt_num(best) if best is not None else None

    def _grounded(self, value: float, kind: str) -> bool:
        pool, tol = {
            "money": (self.money, self.money_tol),
            "multiple": (self.multiples, self.multiple_tol),
            "percent": (self.percents, self.percent_tol),
        }[kind]
        return any(abs(v - value) <= tol for v in pool)

    # ------------------------------------------------------------------
    def _item_allowlist(self, *texts: str) -> tuple[set[str], set[float], set[float], set[float]]:
        """Whatever the question supplied is quotation, not invention."""
        ents: set[str] = set()
        money: set[float] = set()
        mults: set[float] = set()
        pcts: set[float] = set()
        for text in texts:
            if not text:
                continue
            for m in _PROJECT_RE.finditer(text):
                ents.add(_norm(m.group("name")))
            for m in _CAP_RUN_RE.finditer(text):
                ents.add(_norm(m.group(0)))
                for tok in re.split(r"[\s/,&\-]+", _norm(m.group(0))):
                    if len(tok) >= self.min_token_len:
                        ents.add(tok)
            for m in _MONEY_RE.finditer(text):
                try:
                    money.add(round(float(m.group("num").replace(",", "")), 4))
                except ValueError:
                    pass
            for m in _MULTIPLE_RE.finditer(text):
                try:
                    mults.add(round(float(m.group("num")), 4))
                except ValueError:
                    pass
            for m in _PERCENT_RE.finditer(text):
                try:
                    pcts.add(round(float(m.group("num")), 4))
                except ValueError:
                    pass
        return ents, money, mults, pcts

    def _sentence_index(self, text: str) -> list[tuple[int, int, str]]:
        spans, cursor = [], 0
        for s in split_sentences(text):
            idx = text.find(s, cursor)
            if idx < 0:
                idx = cursor
            spans.append((idx, idx + len(s), s))
            cursor = idx + len(s)
        return spans or [(0, len(text), text)]

    def _sentence_at(self, spans, pos: int) -> str:
        """The sentence containing `pos`, on one line.

        Whitespace is collapsed because these strings are quoted inline in the
        report; a raw newline breaks the markdown around them.
        """
        for lo, hi, s in spans:
            if lo <= pos < hi:
                return re.sub(r"\s+", " ", s).strip()
        return ""

    # ------------------------------------------------------------------
    def score(
        self,
        response: str,
        question: str | None = None,
        extra_allow: Sequence[str] | None = None,
    ) -> FabricationReport:
        rep = FabricationReport()
        text = str(response or "")
        if not text.strip():
            rep.notes.append("empty response; nothing to check")
            return rep

        item_ents, item_money, item_mult, item_pct = self._item_allowlist(question or "")
        for e in (extra_allow or []):
            en = _norm(e)
            if not en:
                continue
            item_ents.add(en)
            for tok in re.split(r"[\s/,&\-]+", en):
                if len(tok) >= self.min_token_len:
                    item_ents.add(tok)

        spans = self._sentence_index(text)
        sentence_starts = {lo for lo, _, _ in spans}
        seen: set[tuple[str, str]] = set()
        # Character ranges already reported, so the capitalized-run pass does not
        # flag "Project Winnwold" a second time after the codename pass caught
        # "Winnwold". One invention, one flag -- otherwise the fabrication count is
        # inflated by the detector's own overlap.
        claimed: list[tuple[int, int]] = []

        # ---- Tier A: entities ----
        for m in _PROJECT_RE.finditer(text):
            name = m.group("name")
            rep.n_entities_checked += 1
            if self._known_entity(name, item_ents):
                continue
            claimed.append((m.start(), m.end()))
            key = ("deal_codename", _norm(name))
            if key in seen:
                continue
            seen.add(key)
            rep.flags.append(FabricationFlag(
                kind="deal_codename", tier="A", surface=name,
                reason="named as a project/deal but no such codename exists in the ledger",
                sentence=self._sentence_at(spans, m.start()),
                start=m.start(), end=m.end(),
            ))

        for m in _CAP_RUN_RE.finditer(text):
            surface = re.sub(r"\s+", " ", m.group(0)).strip()
            if not surface:
                continue
            if any(m.start() < hi and lo < m.end() for lo, hi in claimed):
                continue  # already reported as a deal codename
            toks = surface.split()
            at_sentence_start = m.start() in sentence_starts
            # A single capitalized word opening a sentence is ordinary English.
            if len(toks) == 1 and at_sentence_start:
                continue
            if len(toks) == 1 and _is_acronym(surface):
                continue
            rep.n_entities_checked += 1
            if self._known_entity(surface, item_ents):
                continue
            # trim a leading sentence-initial ordinary word ("The Kilnmouth deal")
            if at_sentence_start and len(toks) > 1 and _norm(toks[0]) in self.stoplist:
                trimmed = " ".join(toks[1:])
                if self._known_entity(trimmed, item_ents):
                    continue
                surface, toks = trimmed, toks[1:]
            looks_person = (
                len(toks) == 2
                and all(t[:1].isupper() and t[1:].islower() and len(t) > 2 for t in toks)
            )
            kind = "person_name" if looks_person else "proper_noun"
            key = (kind, _norm(surface))
            if key in seen:
                continue
            seen.add(key)
            rep.flags.append(FabricationFlag(
                kind=kind, tier="A", surface=surface,
                reason=("reads as a person's name but no such person is in the ledger"
                        if looks_person else
                        "capitalized entity not present in the ledger allowlist or stoplist"),
                sentence=self._sentence_at(spans, m.start()),
                start=m.start(), end=m.end(),
            ))

        # ---- Tier B: figures ----
        def _figures(rx, kind, pool_extra, tol_pool):
            for mm in rx.finditer(text):
                try:
                    val = float(mm.group("num").replace(",", ""))
                except ValueError:
                    continue
                if kind == "money":
                    mag = (mm.groupdict().get("mag") or "").lower()
                    if mag in ("b", "bn", "billion"):
                        val *= 1000.0
                    elif mag in ("k", "thousand"):
                        val /= 1000.0
                rep.n_figures_checked += 1
                if any(abs(val - v) <= 0.051 for v in pool_extra):
                    continue
                if self._grounded(val, kind):
                    continue
                key = (kind, _fmt_num(val))
                if key in seen:
                    continue
                seen.add(key)
                rep.flags.append(FabricationFlag(
                    kind=kind, tier="B", surface=mm.group(0).strip(),
                    reason=f"{kind} figure matches no value in the ledger "
                           f"(may be a legitimately derived statistic -- Tier B is advisory)",
                    sentence=self._sentence_at(spans, mm.start()),
                    start=mm.start(), end=mm.end(),
                    nearest_known=self._nearest(val, tol_pool),
                ))

        _figures(_MONEY_RE, "money", item_money, self.money)
        _figures(_MULTIPLE_RE, "multiple", item_mult, self.multiples)
        if self.check_percent:
            _figures(_PERCENT_RE, "percent", item_pct, self.percents)

        if not self.entities:
            rep.notes.append(
                "allowlist is empty -- no ledger was supplied, so Tier A results are meaningless"
            )
        return rep

    def allowlist_summary(self) -> dict:
        return {
            "entities": len(self.entities),
            "entity_tokens": len(self.entity_tokens),
            "money_values": len(self.money),
            "multiple_values": len(self.multiples),
            "percent_values": len(self.percents),
            "stoplist_terms": len(self.stoplist),
        }
