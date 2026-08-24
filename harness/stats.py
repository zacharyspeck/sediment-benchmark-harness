"""Confidence intervals, arm comparison, and honest power arithmetic.

Built in, not bolted on: no score is ever reported without an interval, and the
comparison verdict is computed here rather than written by hand into the
report.

WHY THE INTERVALS ARE SO WIDE
-----------------------------
People expect +/-10 points at n=25. The actual Wilson half-width at n=25 is
about +/-18 points near p=0.5 and about +/-15 points near p=0.8. That is not a
detail -- it means a 25-item family cannot distinguish a 60% arm from a 75% arm
at all. `halfwidth_table()` prints the real numbers so nobody has to take that
on faith, and `sample_size_for_difference()` says how many items a given gap
would actually need.

TWO TESTS, AND WHY THE HEADLINE USES THE STRICTER ONE
-----------------------------------------------------
`intervals_overlap` is the rule the report headlines: if the two arms' Wilson
intervals overlap, print "no significant difference". This rule is CONSERVATIVE
-- non-overlapping intervals always imply a real difference, but overlapping
intervals do NOT imply the absence of one. It errs toward silence.

`newcombe_diff_ci` is the statistically proper test for a difference of two
independent proportions. It is computed and printed alongside, because when the
two disagree (overlap says "no difference", Newcombe excludes zero) the honest
statement is "suggestive, underpowered" -- not a winner, and not silence
either. The report labels that case explicitly instead of picking whichever
answer flatters the result.

WILSON DOES NOT APPLY TO EVERYTHING
-----------------------------------
The Wilson interval is for a binomial proportion. Five of the six families
score each item 0 or 1, so it is exactly right. convention_conformance scores a
FRACTION of a checklist per item, which is not a Bernoulli trial; using Wilson
there would understate the uncertainty. That family gets a seeded bootstrap
interval over item scores instead, and the report says which is which.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "Interval", "ProportionStat", "MeanStat",
    "wilson_ci", "bootstrap_mean_ci", "newcombe_diff_ci",
    "intervals_overlap", "compare_arms", "ArmComparison", "halfwidth_table",
    "sample_size_for_difference", "detectable_difference", "n_to_separate", "z_for",
]


# --------------------------------------------------------------------------
# normal quantile (no scipy dependency)
# --------------------------------------------------------------------------
def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1.15e-9 across (0,1), which is far beyond what a benchmark
    with 25 items per family could possibly need.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def z_for(conf: float = 0.95) -> float:
    """Two-sided z multiplier for a confidence level."""
    return _norm_ppf(1 - (1 - conf) / 2.0)


# --------------------------------------------------------------------------
@dataclass
class Interval:
    lo: float
    hi: float
    method: str = ""
    conf: float = 0.95

    @property
    def halfwidth(self) -> float:
        return (self.hi - self.lo) / 2.0

    def as_pct(self) -> tuple[float, float]:
        return (100.0 * self.lo, 100.0 * self.hi)

    def fmt(self, pct: bool = True) -> str:
        if pct:
            return f"[{100*self.lo:.1f}, {100*self.hi:.1f}]"
        return f"[{self.lo:.3f}, {self.hi:.3f}]"

    def to_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "method": self.method,
                "conf": self.conf, "halfwidth": self.halfwidth}


@dataclass
class ProportionStat:
    """A binomial family score: k successes out of n items."""

    k: float
    n: int
    conf: float = 0.95
    label: str = ""
    ci: Interval = field(init=False)

    def __post_init__(self):
        self.ci = wilson_ci(self.k, self.n, self.conf)

    @property
    def point(self) -> float:
        return (self.k / self.n) if self.n else 0.0

    def to_dict(self) -> dict:
        return {"kind": "proportion", "k": self.k, "n": self.n,
                "point": self.point, "ci": self.ci.to_dict(), "label": self.label}


@dataclass
class MeanStat:
    """A fractional-score family: mean of per-item scores in [0,1]."""

    values: Sequence[float]
    conf: float = 0.95
    seed: int = 20260818
    iters: int = 10000
    label: str = ""
    ci: Interval = field(init=False)

    def __post_init__(self):
        self.ci = bootstrap_mean_ci(self.values, conf=self.conf, seed=self.seed, iters=self.iters)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def point(self) -> float:
        return (sum(self.values) / len(self.values)) if self.values else 0.0

    def to_dict(self) -> dict:
        return {"kind": "mean", "n": self.n, "point": self.point,
                "ci": self.ci.to_dict(), "label": self.label,
                "seed": self.seed, "iters": self.iters}


# --------------------------------------------------------------------------
def wilson_ci(k: float, n: int, conf: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because it behaves at the extremes:
    at k=n it does not produce an upper bound of 1.0 with zero width, and at
    k=0 it does not collapse to a point. Both cases happen constantly at n=25.
    """
    if n <= 0:
        return Interval(0.0, 1.0, "wilson (n=0: no information)", conf)
    z = z_for(conf)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(max(0.0, centre - margin), min(1.0, centre + margin), "wilson", conf)


def bootstrap_mean_ci(values: Sequence[float], conf: float = 0.95,
                      seed: int = 20260818, iters: int = 10000) -> Interval:
    """Seeded percentile bootstrap for the mean of per-item scores.

    Deterministic by construction: the same run always yields the same
    interval, so a number in the report can be reproduced exactly.
    """
    vals = [float(v) for v in (values or [])]
    if not vals:
        return Interval(0.0, 1.0, "bootstrap (n=0: no information)", conf)
    if len(vals) == 1:
        return Interval(vals[0], vals[0], "bootstrap (n=1: point estimate only)", conf)
    if len(set(vals)) == 1:
        # Every item scored identically, so every resample has the same mean and
        # the percentile interval collapses to a point. A zero-width interval
        # reads as certainty and is nothing of the kind -- it says the sample
        # showed no variation, not that the true rate is known exactly. Labelled
        # so the report can surface it rather than printing [100.0, 100.0] as
        # though it were a measurement.
        return Interval(
            vals[0], vals[0],
            f"percentile bootstrap (DEGENERATE: all {len(vals)} item scores identical; "
            f"zero width reflects no observed variation, not certainty)",
            conf,
        )
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(iters):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    alpha = (1 - conf) / 2.0
    lo = means[max(0, int(math.floor(alpha * iters)))]
    hi = means[min(iters - 1, int(math.ceil((1 - alpha) * iters)) - 1)]
    return Interval(lo, hi, f"percentile bootstrap (seed={seed}, {iters} resamples)", conf)


def newcombe_diff_ci(k1: float, n1: int, k2: float, n2: int, conf: float = 0.95) -> Interval:
    """Newcombe hybrid-score interval for p1 - p2.

    The proper test for a difference of two independent proportions, and the
    right thing to quote when someone asks "is arm A actually better".
    """
    if n1 <= 0 or n2 <= 0:
        return Interval(-1.0, 1.0, "newcombe (insufficient n)", conf)
    a = wilson_ci(k1, n1, conf)
    b = wilson_ci(k2, n2, conf)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - a.lo) ** 2 + (b.hi - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((a.hi - p1) ** 2 + (p2 - b.lo) ** 2)
    return Interval(max(-1.0, lo), min(1.0, hi), "newcombe hybrid score", conf)


def intervals_overlap(a: Interval, b: Interval) -> bool:
    return not (a.hi < b.lo or b.hi < a.lo)


# --------------------------------------------------------------------------
@dataclass
class ArmComparison:
    arm_a: str
    arm_b: str
    point_a: float
    point_b: float
    ci_a: Interval
    ci_b: Interval
    diff_ci: Interval | None
    overlap: bool
    verdict: str
    headline: str
    caveat: str = ""

    def to_dict(self) -> dict:
        return {
            "arm_a": self.arm_a, "arm_b": self.arm_b,
            "point_a": self.point_a, "point_b": self.point_b,
            "ci_a": self.ci_a.to_dict(), "ci_b": self.ci_b.to_dict(),
            "diff_ci": self.diff_ci.to_dict() if self.diff_ci else None,
            "intervals_overlap": self.overlap,
            "verdict": self.verdict, "headline": self.headline, "caveat": self.caveat,
        }


def compare_arms(stat_a, stat_b, name_a: str, name_b: str) -> ArmComparison:
    """Compare two arms on one family. The overlap rule decides the headline."""
    ci_a, ci_b = stat_a.ci, stat_b.ci
    pa, pb = stat_a.point, stat_b.point
    overlap = intervals_overlap(ci_a, ci_b)

    diff = None
    if isinstance(stat_a, ProportionStat) and isinstance(stat_b, ProportionStat):
        diff = newcombe_diff_ci(stat_a.k, stat_a.n, stat_b.k, stat_b.n, stat_a.conf)

    gap_pts = 100.0 * (pa - pb)

    if overlap:
        verdict = "no_significant_difference"
        headline = "no significant difference"
        caveat = (
            f"{name_a} scored {100*pa:.1f}% and {name_b} scored {100*pb:.1f}% "
            f"(a {abs(gap_pts):.1f} point gap), but the intervals overlap. "
            f"At this n that gap is indistinguishable from noise."
        )
        if diff and (diff.lo > 0 or diff.hi < 0):
            verdict = "suggestive_underpowered"
            headline = "no significant difference (suggestive, underpowered)"
            caveat += (
                f" The paired-difference interval {diff.fmt()} does exclude zero, so a real "
                f"effect may be present -- but the overlap rule is the one this report "
                f"headlines, and it declines to call it. Treat as a hypothesis, not a result."
            )
    else:
        winner, loser = (name_a, name_b) if pa > pb else (name_b, name_a)
        verdict = "significant"
        headline = f"{winner} > {loser}"
        caveat = (
            f"Intervals do not overlap ({ci_a.fmt()} vs {ci_b.fmt()}), so the "
            f"{abs(gap_pts):.1f} point gap survives at this n."
        )
    return ArmComparison(name_a, name_b, pa, pb, ci_a, ci_b, diff, overlap,
                         verdict, headline, caveat)


# --------------------------------------------------------------------------
# power arithmetic, printed in the report so nobody has to trust a claim
# --------------------------------------------------------------------------
def halfwidth_table(ns: Sequence[int] = (10, 25, 50, 100, 200, 400),
                    ps: Sequence[float] = (0.5, 0.7, 0.9),
                    conf: float = 0.95) -> list[dict]:
    """Wilson half-width in percentage points, by n and by true rate."""
    rows = []
    for n in ns:
        row = {"n": n}
        for p in ps:
            ci = wilson_ci(round(p * n), n, conf)
            row[f"p={p:.0%}"] = round(100 * ci.halfwidth, 1)
        rows.append(row)
    return rows


def sample_size_for_difference(p1: float, p2: float, alpha: float = 0.05,
                               power: float = 0.80) -> int:
    """Items PER ARM needed to detect p1 vs p2 (two-sided, unpaired)."""
    if p1 == p2:
        return -1
    z_a = z_for(1 - alpha)
    z_b = _norm_ppf(power)
    pbar = (p1 + p2) / 2.0
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar))
           + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / ((p1 - p2) ** 2)))


def n_to_separate(p1: float, p2: float, conf: float = 0.95, max_n: int = 5000) -> int | None:
    """Smallest n per arm at which THIS REPORT would call the difference.

    The report's headline rule is interval overlap, so "how many items would I
    need" must be answered under that same rule. Answering it with the
    classical two-proportion power formula produces numbers that contradict the
    verdict printed above them -- at p=1.0 vs p=0.2 the power formula says n=5
    is plenty while the Wilson intervals at n=5 still overlap, because the
    normal approximation the power formula rests on is invalid at the extremes
    and because the overlap rule is strictly more conservative.

    Returns None if the rates are equal or separation needs more than `max_n`.
    """
    if p1 == p2:
        return None
    n = 2
    while n <= max_n:
        a = wilson_ci(round(p1 * n), n, conf)
        b = wilson_ci(round(p2 * n), n, conf)
        if not intervals_overlap(a, b):
            return n
        n += 1 if n < 60 else (5 if n < 400 else 25)
    return None


def detectable_difference(n: int, p_base: float = 0.6, alpha: float = 0.05,
                          power: float = 0.80, step: float = 0.005) -> float | None:
    """Smallest gap detectable at this n. Returns points, or None if > 40."""
    d = step
    while d <= 0.40:
        p2 = min(0.999, p_base + d)
        if sample_size_for_difference(p2, p_base, alpha, power) <= n:
            return round(100 * d, 1)
        d += step
    return None
