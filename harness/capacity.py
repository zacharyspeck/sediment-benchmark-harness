"""Honest capacity measurement.

The question this answers is not "how many items could I emit" but "how many
items can I emit that are each a distinct, answerable, correctly-golded test".
Those are very different numbers, and the gap between them is where a benchmark
quietly becomes 30 facts asked three ways.

FOUR STAGES, EACH STRICTLY SMALLER THAN THE LAST
------------------------------------------------
  raw          rows the selector matches, summed over every slot binding
  distinct     after collapsing to the family's unit of independence
  grounded     after dropping units no corpus document can answer
  defensible   after dropping units a corpus document contradicts

Only `defensible` sets n. The other three are reported so the shrinkage is
visible and arguable rather than hidden.

WHY GROUNDING IS A GATE AND NOT A WARNING
-----------------------------------------
Every arm searches the same corpus. An item whose answer appears in no document
is answered by no arm; it contributes a near-constant score to every arm and
cannot separate them. It is not a hard item, it is an absent measurement. Same
logic in reverse for contradiction: if a document refutes the gold, the arm that
reads the corpus faithfully is scored wrong.

OVERLAP, FOR THE TWO AGGREGATE FAMILIES
---------------------------------------
Jaccard alone is blind to containment. A 46-row set sitting entirely inside a
1,445-row set scores J = 0.03 and looks independent, while one item's answer
hands you a large slice of the other's. Admission therefore requires BOTH
Jaccard <= 0.25 and overlap coefficient |A n B| / min(|A|,|B|) <= 0.25, and
capacity is the exact maximum independent set of the conflict graph -- exact,
because greedy was measured off by 3-7 on this graph.

Nothing here calls a model. Every gate is a regex, a substring test or a number
comparison.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from .query import aggregate, select
from .templates import substitute
from .stats import wilson_ci

# Families whose unit is a set of rows rather than a row.
AGGREGATE_FAMILIES = {"multi_doc_aggregation", "corpus_wide_statistic"}

# The one family whose units are not ledger rows and whose "grounding" runs
# backwards: a holdout target MUST appear in no document.
HOLDOUT_FAMILY = "implicit_rule_application"

# Recency is NOT one of the six. Its unit is (subject, field) -- one item per
# company per figure -- because six figures about one company are six different
# facts with six different supersessions, not one fact asked six ways. Each
# template IS a field, so the unit key is (template, row).
#
# The clustering is real and must be reported with the number: 57 units over 11
# companies share document sets, so the effective independent sample for
# anything company-level is nearer 11 than 57.
RECENCY_FAMILY = "recency"

JACCARD_MAX = 0.25
OVERLAP_MAX = 0.25


# --------------------------------------------------------------------- slots
def expand_bindings(tpl, ledger) -> list[dict]:
    """Every declared slot combination, as an explicit cartesian product.

    `default_bindings` in templates.py takes choose[0] -- one binding per
    template -- which under-reports an aggregate family by the size of its
    cross-product. This is the enumeration the audit needs.

    Per-row shorthand slots (`codename: codename`) carry no `choose` and are
    filled at generation time from the selected row, so they contribute no
    combinations here.
    """
    axes: list[tuple[str, list]] = []
    for name, spec in (tpl.slots or {}).items():
        if isinstance(spec, dict) and spec.get("choose"):
            axes.append((name, list(spec["choose"])))
    if not axes:
        return [{}]
    names = [a[0] for a in axes]
    return [dict(zip(names, combo)) for combo in itertools.product(*[a[1] for a in axes])]


def slot_cardinality(tpl) -> tuple[int, str]:
    """(product, "4 sectors x 4 years = 16") for the report."""
    parts, total = [], 1
    for name, spec in (tpl.slots or {}).items():
        if isinstance(spec, dict) and spec.get("choose"):
            k = len(spec["choose"])
            parts.append(f"{k} {name}")
            total *= k
    if not parts:
        return 1, "no slots = 1"
    return total, " x ".join(parts) + f" = {total}"


# --------------------------------------------------------------- gold render
_MONEY = re.compile
def _num_variants(v: float, unit: str) -> list[str]:
    """The surface forms the corpus uses for a value.

    Deliberately narrow. A loose match (bare `6.5` anywhere) would count a
    coincidental figure as grounding and inflate capacity, which is the exact
    failure this whole module exists to prevent.
    """
    out: list[str] = []
    try:
        f = float(v)
    except (TypeError, ValueError):
        return [str(v)]
    one = f"{f:.1f}"
    two = f"{f:.2f}"
    whole = f"{f:.0f}"
    if unit == "money_m":
        out += [f"${one}M", f"${two}M", f"${one} million", f"${whole}M"]
    elif unit == "percent":
        out += [f"{one}%", f"{whole}%", f"{two}%"]
    elif unit == "multiple":
        out += [f"{one}x", f"{two}x", f"{whole}x"]
    elif unit in ("years", "count"):
        out += [one, whole]
    else:
        out += [one, whole, two]
    return [o for o in out if o]


def gold_is_rendered(row: dict, gold_value: Any, unit: str, accept: dict | None,
                     index, codename: str | None) -> bool:
    """Does at least one of this deal's own documents actually state the gold?

    For a categorical gold the accept map supplies the phrasings the corpus is
    allowed to use; for a numeric gold the formatted surface forms above.
    """
    docs = index.docs_for(codename)
    if not docs:
        return False
    needles: list[str] = []
    if accept:
        key = str(gold_value)
        needles.append(key)
        for phrase in (accept.get(key) or []):
            needles.append(str(phrase))
    elif isinstance(gold_value, (int, float)) and not isinstance(gold_value, bool):
        needles = _num_variants(gold_value, unit or "")
    else:
        needles = [str(gold_value)]
    low = [n.lower() for n in needles if n]
    for d in docs:
        t = d.text.lower()
        for n in low:
            if n in t:
                return True
    return False


# ------------------------------------------------------------------- recency
def stated_values_by_date(index, codename, patterns) -> dict:
    """{document date -> set of values} for one labelled figure.

    The patterns must each capture exactly one numeric group and be anchored on
    the label, not on the bare number. A loose pattern collects unrelated
    figures and manufactures a supersession that is not there.
    """
    rx = [p if hasattr(p, "search") else re.compile(p) for p in patterns or ()]
    out: dict[str, set] = {}
    for d in index.docs_for(codename):
        for r in rx:
            for m in r.finditer(d.text):
                try:
                    v = float(m.group(1).replace(",", ""))
                except (TypeError, ValueError, IndexError):
                    continue
                out.setdefault(d.date, set()).add(v)
    return out


def _superseded_ok(cfg, row, index, codename, gold_value) -> tuple[bool, str]:
    """A recency unit is admissible only when all four hold.

    1. the figure is stated on at least two distinct document dates
    2. at least two DIFFERENT values are stated (there is a real disagreement)
    3. the latest date states exactly ONE value (the current answer is not itself
       ambiguous)
    4. that latest value equals the ledger's current value

    Anything weaker manufactures recency out of noise. (3) in particular matters:
    a later document that states two different figures for the same fact cannot
    be the source of a single gold answer.
    """
    pats = cfg.get("patterns") if isinstance(cfg, dict) else None
    if not pats:
        return False, "recency gate has no value patterns"
    obs = stated_values_by_date(index, codename, pats)
    if len(obs) < 2:
        return False, "figure is stated on fewer than two document dates"
    values = {v for s in obs.values() for v in s}
    if len(values) < 2:
        return False, "every document states the same value -- nothing is superseded"
    latest = obs[max(obs)]
    if len(latest) != 1:
        return False, "the latest document states more than one value for this figure"
    late = next(iter(latest))
    earliest = obs[min(obs)]
    if late in earliest:
        return False, "the latest value already appears in the earliest document"
    if gold_value is None:
        return False, "no gold to check the latest stated value against"
    try:
        if abs(float(late) - float(gold_value)) > 1e-6:
            return False, "the latest stated value does not match the ledger current value"
    except (TypeError, ValueError):
        return False, "gold is not numeric, so recency cannot be checked"
    return True, ""


# --------------------------------------------------------------------- gates
def _grader_invention_patterns(tpl) -> list:
    """Exactly the pattern set graders/absence_and_abstention.py would apply."""
    from graders.absence_and_abstention import DEFAULT_THESIS_PATTERNS

    grading = (getattr(tpl, "raw", None) or {}).get("grading") or {}
    gold = (getattr(tpl, "raw", None) or {}).get("gold") or {}
    if grading.get("thesis_patterns"):
        pats = list(grading["thesis_patterns"])
    else:
        pats = list(DEFAULT_THESIS_PATTERNS) + list(grading.get("extra_thesis_patterns") or [])
    pats += list(grading.get("forbidden_claims") or gold.get("forbidden_claims") or [])
    return pats


def _gate_cfg(tpl) -> dict:
    raw = getattr(tpl, "raw", None) or {}
    return raw.get("gates") or {}


def unit_passes_gates(tpl, row: dict, index, gold_value=None, unit="", accept=None
                      ) -> tuple[bool, bool, str]:
    """(grounded, contradicted, reason).

    Reads the `gates:` block on the template. Absent config means the
    conservative default: require a document, do not require the gold to be
    rendered, contradict nothing.
    """
    cfg = _gate_cfg(tpl)
    g = cfg.get("grounding") or {}
    c = cfg.get("contradiction") or {}
    # Which field on THIS collection's rows names the documents' subject. Deals
    # use `codename`; portfolio companies carry `deal_codename`. Getting this
    # wrong silently reports every unit as ungrounded, which reads exactly like
    # a real corpus gap -- so it is configured, not guessed.
    subject_field = g.get("codename_field") or c.get("codename_field") or "codename"
    codename = row.get(subject_field)

    grounded, reason = True, ""
    if g.get("require_doc", True) and not index.has_doc(codename):
        grounded, reason = False, "no corpus document names this deal"
    if grounded and g.get("require_doctype"):
        if not index.has_doctype(codename, g["require_doctype"]):
            grounded, reason = False, "no document of the required type"
    if grounded and g.get("require_patterns"):
        if not index.matches_any(codename, g["require_patterns"]):
            grounded, reason = False, "no document matches the required evidence pattern"
    if grounded and g.get("require_gold_rendered") and gold_value is not None:
        if not gold_is_rendered(row, gold_value, unit, accept, index, codename):
            grounded, reason = False, "gold value is not stated in any of its documents"
    if grounded and g.get("require_superseded"):
        ok, why = _superseded_ok(g["require_superseded"], row, index, codename, gold_value)
        if not ok:
            grounded, reason = False, why

    contradicted = False
    if c:
        # The family's OWN grader, mirrored onto the corpus. This is not a
        # judgement call and not an extra opinion about what counts as a
        # contradiction: graders/absence_and_abstention.py scores a response as
        # `invented_thesis` when it matches one of these patterns. If a pattern
        # already appears in the deal's own documents, an arm that reads the
        # corpus faithfully and quotes it is scored 0 WITH THE CORRECT VERDICT.
        # Such an item cannot be answered well, so it is not capacity.
        #
        # Pulling the patterns from the grader rather than restating them is the
        # point: a restated list drifts, and a drifted list silently re-admits
        # broken items.
        if not contradicted and c.get("grader_mirror"):
            pats = _grader_invention_patterns(tpl)
            if index.matches_any(codename, pats):
                contradicted, reason = True, (
                    "a document already contains language this family's grader "
                    "scores as an invention")
        if not contradicted and c.get("doctypes") and index.has_doctype(codename, c["doctypes"]):
            contradicted, reason = True, "a document of a refuting type exists"
        elif c.get("patterns") and index.matches_any(codename, c["patterns"]):
            contradicted, reason = True, "a document matches a refuting pattern"
    return grounded, contradicted, reason


# ------------------------------------------------------------------- overlap
def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def overlap_coefficient(a: frozenset, b: frozenset) -> float:
    m = min(len(a), len(b))
    return (len(a & b) / m) if m else 0.0


def conflicts(a: frozenset, b: frozenset) -> bool:
    return jaccard(a, b) > JACCARD_MAX or overlap_coefficient(a, b) > OVERLAP_MAX


def max_independent_set(sets: list[frozenset]) -> list[int]:
    """Exact maximum independent set over the conflict graph.

    Branch and bound with a greedy warm start and a degree-ordered branch
    variable. Exact matters here: on the shipped aggregate pool a greedy
    selection was measured 3-7 units short of optimal depending on the
    threshold, and capacity is the number that sets n.
    """
    n = len(sets)
    if n == 0:
        return []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if conflicts(sets[i], sets[j]):
                adj[i].add(j)
                adj[j].add(i)

    # Several admissible sets are usually the same size, so a pure count
    # objective picks arbitrarily between them. Tie-break on total rows covered:
    # among equally large admissible sets, prefer the one that asks about more
    # of the corpus. Deterministic, and it stops a five-row question from
    # displacing an eleven-row one for no reason.
    sizes = [len(x) for x in sets]

    def score(chosen: list[int]) -> tuple[int, int]:
        return (len(chosen), sum(sizes[i] for i in chosen))

    order = sorted(range(n), key=lambda i: (len(adj[i]), -sizes[i], i))
    greedy: list[int] = []
    blocked: set[int] = set()
    for i in order:
        if i not in blocked:
            greedy.append(i)
            blocked |= adj[i]
    best = list(greedy)

    def expand(candidates: list[int], chosen: list[int]) -> None:
        nonlocal best
        # Prune on cardinality with <=, not <. Relaxing this to explore ties for
        # the coverage tie-break removes the only strong bound the search has
        # and turns an 85-node sparse graph into an exponential walk -- measured,
        # it did not terminate. The tie-break is handled instead by the warm
        # start, which is ordered to prefer larger sets, so ties are broken well
        # without searching every one of them.
        if len(chosen) + len(candidates) <= len(best):
            return
        if not candidates:
            if score(chosen) > score(best):
                best = list(chosen)
            return
        # Branch on the most-constrained candidate.
        pivot = max(candidates, key=lambda i: (len(adj[i] & set(candidates)), -i))
        rest = [c for c in candidates if c != pivot]
        # include pivot
        expand([c for c in rest if c not in adj[pivot]], chosen + [pivot])
        # exclude pivot
        expand(rest, chosen)

    expand(sorted(range(n), key=lambda i: (len(adj[i]), -sizes[i], i)), [])
    return sorted(best)


def overlap_histogram(sets: list[frozenset], metric) -> dict:
    buckets = {"0": 0, "(0,0.10]": 0, "(0.10,0.25]": 0, "(0.25,0.50]": 0,
               "(0.50,0.75]": 0, "(0.75,0.99]": 0, "1.0": 0}
    pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            v = metric(sets[i], sets[j])
            pairs += 1
            if v == 0:
                buckets["0"] += 1
            elif v <= 0.10:
                buckets["(0,0.10]"] += 1
            elif v <= 0.25:
                buckets["(0.10,0.25]"] += 1
            elif v <= 0.50:
                buckets["(0.25,0.50]"] += 1
            elif v <= 0.75:
                buckets["(0.50,0.75]"] += 1
            elif v < 1.0:
                buckets["(0.75,0.99]"] += 1
            else:
                buckets["1.0"] += 1
    return {"pairs": pairs, "buckets": buckets}


def top_overlapping(labelled: list[tuple[str, frozenset]], k: int = 10) -> list[dict]:
    rows = []
    for i in range(len(labelled)):
        for j in range(i + 1, len(labelled)):
            (la, a), (lb, b) = labelled[i], labelled[j]
            rows.append({
                "a": la, "b": lb, "n_a": len(a), "n_b": len(b),
                "jaccard": round(jaccard(a, b), 4),
                "overlap_coef": round(overlap_coefficient(a, b), 4),
                "subset": a < b or b < a,
            })
    rows.sort(key=lambda r: (-max(r["jaccard"], r["overlap_coef"]), r["a"], r["b"]))
    return rows[:k]


# --------------------------------------------------------------------- units
@dataclass
class UnitRecord:
    key: str
    template_id: str
    family: str
    row_id: str | None = None
    codename: str | None = None
    binding: dict = field(default_factory=dict)
    row_set: frozenset = frozenset()
    grounded: bool = True
    contradicted: bool = False
    reason: str = ""
    gold: Any = None

    @property
    def defensible(self) -> bool:
        return self.grounded and not self.contradicted


# ----------------------------------------------------- family 4 unit keying
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def printed_fields(fact_block: str) -> set[str]:
    """Fields the item actually SHOWS the model.

    Derived from the fact block rather than hand-listed, so it cannot drift out
    of sync with the template. A rule whose deciding field is absent here makes
    its targets undecidable: two targets with identical printed facts can carry
    opposite gold, and no amount of studying the firm resolves that.
    """
    return set(_PLACEHOLDER.findall(fact_block or ""))


def lever_bin(value, step: float):
    """Bin a lever to the coarsest step a reader can still act on.

    Two targets differing by 0.1pp of retention are the same item in every way
    a model or a reader can perceive. Binning is what stops near-duplicates
    from being counted as distinct capacity.
    """
    if value is None:
        return None                       # genuinely absent -- inadmissible
    if not step:
        return str(value)                 # categorical lever: no binning
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return round(round(f / step) * step, 6)


def load_lever_fields(doc: dict) -> dict:
    """The lever map for family 4, merged from the private side-file if present.

    The map records which fields each implicit rule reads. That is a property
    of the corpus, not of the harness, so it is not committed here: the family
    document ships an empty stub and the real map arrives beside the private
    pool as `config/lever_fields.yaml`. When the file is absent the map is
    empty and every discriminating target rejects as "lever not printed", which
    makes the family measure zero rather than measure the wrong population.
    """
    import os as _os
    from .yamlio import load_file
    out = dict(doc.get("lever_fields") or {})
    p = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "config", "lever_fields.yaml")
    if _os.path.exists(p):
        side = load_file(p) or {}
        out.update(side.get("lever_fields") or side)
    return out


def holdout_unit_key(target: dict, lever_map: dict, printed: set[str]
                     ) -> tuple[str | None, str]:
    """(unit key, reason-if-inadmissible) for one holdout target.

    Unit = (firing rule-set, binned lever values), admitted only when EVERY
    firing rule's lever is printed. A target driven by a rule whose lever is
    hidden is not a hard item, it is an unanswerable one.

    CONTROLS ARE KEYED DIFFERENTLY. A clean_advance / clean_pass target fires no
    implicit rule -- that is what makes it clean -- so it has no lever and the
    rule-set key does not apply. It is decidable from the written criteria the
    fact block prints, so each control is its own unit. Controls are never
    scored (pre-registration), they are reported on their own line, so this only
    governs how many are available to generate.
    """
    bucket = str(target.get("bucket") or "")
    rules = sorted(target.get("driving_rule_ids") or [])
    if bucket in ("clean_advance", "clean_pass"):
        return "control:" + str(target.get("id") or id(target)), ""
    if not rules:
        return None, "discriminating target with no driving rule recorded"
    parts = []
    for rid in rules:
        spec = lever_map.get(rid) or {}
        # A rule may need MORE THAN ONE field to be decidable: a rule that
        # compares one figure against a threshold derived from another is not
        # determined by the headline figure alone, so printing it is not enough.
        # Every field a rule reads must be printed or the target is undecidable
        # regardless of how visible the headline lever is.
        fields = spec.get("fields") or ([spec["field"]] if spec.get("field") else [])
        if not fields:
            return None, f"{rid}: no lever field declared"
        bins = spec.get("bins") or {}
        vals = []
        for fieldname in fields:
            if fieldname not in printed:
                return None, f"{rid}: lever `{fieldname}` is not printed in the fact block"
            step = bins.get(fieldname, spec.get("bin"))
            b = lever_bin(target.get(fieldname), step)
            if b is None:
                return None, f"{rid}: lever `{fieldname}` is absent on this target"
            vals.append(f"{fieldname}={b}")
        parts.append(rid + "[" + ",".join(vals) + "]")
    return "|".join(parts), ""


def halfwidth_at(n: int, p: float = 0.5) -> float:
    """Wilson 95% half-width in points, via the repo's own function."""
    if n <= 0:
        return 100.0
    return round(100 * wilson_ci(round(p * n), n).halfwidth, 2)
