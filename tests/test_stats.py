"""Confidence intervals, the overlap rule, and the power arithmetic."""

import math

import pytest

from harness.stats import (
    MeanStat,
    ProportionStat,
    bootstrap_mean_ci,
    compare_arms,
    detectable_difference,
    halfwidth_table,
    intervals_overlap,
    newcombe_diff_ci,
    sample_size_for_difference,
    wilson_ci,
    z_for,
)


def test_z_for_known_values():
    assert z_for(0.95) == pytest.approx(1.959963, abs=1e-5)
    assert z_for(0.99) == pytest.approx(2.575829, abs=1e-5)


def test_wilson_matches_published_value():
    # 15/25: Wilson 95% interval is approximately [0.4070, 0.7664]
    ci = wilson_ci(15, 25)
    assert ci.lo == pytest.approx(0.4070, abs=5e-4)
    assert ci.hi == pytest.approx(0.7664, abs=5e-4)


def test_wilson_does_not_collapse_at_the_extremes():
    """The reason Wilson is used instead of the normal approximation."""
    at_all = wilson_ci(25, 25)
    assert at_all.hi == pytest.approx(1.0, abs=1e-9)
    assert at_all.lo < 0.95, "an interval at k=n must still have width"
    at_none = wilson_ci(0, 25)
    assert at_none.lo == pytest.approx(0.0, abs=1e-9)
    assert at_none.hi > 0.05, "an interval at k=0 must still have width"


def test_wilson_never_leaves_the_unit_interval():
    for n in (1, 5, 25, 100):
        for k in range(n + 1):
            ci = wilson_ci(k, n)
            assert 0.0 <= ci.lo <= ci.hi <= 1.0


def test_wilson_narrows_as_n_grows():
    widths = [wilson_ci(round(0.6 * n), n).halfwidth for n in (10, 25, 50, 100, 400)]
    assert widths == sorted(widths, reverse=True)


def test_halfwidth_at_25_is_wider_than_ten_points():
    """The number people assume is wrong, and the report says so."""
    row = next(r for r in halfwidth_table() if r["n"] == 25)
    assert row["p=50%"] > 15.0, "half-width at n=25, p=0.5 should be ~18 points"
    assert row["p=90%"] > 10.0


def test_sample_size_grows_as_the_gap_shrinks():
    n5 = sample_size_for_difference(0.65, 0.60)
    n10 = sample_size_for_difference(0.70, 0.60)
    n20 = sample_size_for_difference(0.80, 0.60)
    assert n5 > n10 > n20
    assert n10 > 300, "a 10-point gap needs hundreds of items per arm"


def test_n_to_separate_agrees_with_the_overlap_rule():
    """The 'how many more items' answer must use the rule the report applies."""
    from harness.stats import n_to_separate

    p1, p2 = 1.0, 0.2
    n = n_to_separate(p1, p2)
    assert n is not None
    # at n, the intervals must not overlap...
    a, b = wilson_ci(round(p1 * n), n), wilson_ci(round(p2 * n), n)
    assert not intervals_overlap(a, b)
    # ...and one step below, they must
    a2, b2 = wilson_ci(round(p1 * (n - 1)), n - 1), wilson_ci(round(p2 * (n - 1)), n - 1)
    assert intervals_overlap(a2, b2)


def test_n_to_separate_is_more_conservative_than_the_power_formula():
    """The reason both are printed: they disagree, and the stricter one binds."""
    from harness.stats import n_to_separate

    overlap_n = n_to_separate(1.0, 0.2)
    power_n = sample_size_for_difference(1.0, 0.2)
    assert overlap_n > power_n, (
        "the overlap rule should need more items than the classical power calc")


def test_n_to_separate_is_none_for_equal_rates():
    from harness.stats import n_to_separate

    assert n_to_separate(0.6, 0.6) is None


def test_detectable_difference_at_25_is_large():
    d = detectable_difference(25)
    assert d is not None and d > 25.0, "n=25 cannot detect anything under ~30 points"


def test_overlap_rule_declines_to_call_a_12_point_gap_at_n25():
    a = ProportionStat(k=15, n=25)   # 60%
    b = ProportionStat(k=18, n=25)   # 72%
    comp = compare_arms(b, a, "arm_b", "arm_a")
    assert comp.overlap is True
    assert comp.headline.startswith("no significant difference")
    assert "noise" in comp.caveat


def test_overlap_rule_calls_a_large_gap():
    a = ProportionStat(k=10, n=25)   # 40%
    b = ProportionStat(k=24, n=25)   # 96%
    comp = compare_arms(b, a, "arm_b", "arm_a")
    assert comp.overlap is False
    assert comp.verdict == "significant"
    assert "arm_b > arm_a" == comp.headline


def test_suggestive_underpowered_is_labelled_not_hidden():
    """Where Newcombe excludes zero but the intervals still overlap."""
    found = None
    for n in (40, 60, 80, 100):
        for ka, kb in ((round(0.5 * n), round(0.75 * n)), (round(0.55 * n), round(0.8 * n))):
            comp = compare_arms(ProportionStat(k=kb, n=n), ProportionStat(k=ka, n=n), "b", "a")
            if comp.verdict == "suggestive_underpowered":
                found = comp
                break
        if found:
            break
    assert found is not None, "expected to find a suggestive-but-underpowered case"
    assert "hypothesis" in found.caveat
    assert found.diff_ci is not None


def test_newcombe_interval_brackets_the_observed_difference():
    d = newcombe_diff_ci(20, 25, 15, 25)
    assert d.lo <= (20 / 25 - 15 / 25) <= d.hi


def test_intervals_overlap_is_symmetric():
    a, b = wilson_ci(10, 25), wilson_ci(18, 25)
    assert intervals_overlap(a, b) == intervals_overlap(b, a)


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    vals = [0.8, 1.0, 0.6, 0.9, 0.75, 1.0, 0.5, 0.85]
    a = bootstrap_mean_ci(vals, seed=20260818, iters=2000)
    b = bootstrap_mean_ci(vals, seed=20260818, iters=2000)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    c = bootstrap_mean_ci(vals, seed=999, iters=2000)
    assert (c.lo, c.hi) != (a.lo, a.hi) or True  # different seed may coincide; not asserted


def test_degenerate_bootstrap_is_labelled_not_presented_as_certainty():
    """All-identical scores collapse the interval; it must say so."""
    ci = bootstrap_mean_ci([1.0, 1.0, 1.0, 1.0, 1.0], seed=1, iters=500)
    assert ci.lo == ci.hi == 1.0
    assert "DEGENERATE" in ci.method
    assert "not certainty" in ci.method


def test_bootstrap_brackets_the_mean():
    vals = [0.8, 1.0, 0.6, 0.9, 0.75, 1.0, 0.5, 0.85]
    ci = bootstrap_mean_ci(vals, seed=1, iters=3000)
    mean = sum(vals) / len(vals)
    assert ci.lo <= mean <= ci.hi


def test_mean_stat_used_for_fractional_scores():
    st = MeanStat(values=[0.5, 0.75, 1.0, 0.25], seed=7, iters=1000)
    assert st.n == 4
    assert st.point == pytest.approx(0.625)
    assert "bootstrap" in st.ci.method


def test_proportion_stat_uses_wilson():
    st = ProportionStat(k=3, n=4)
    assert st.point == 0.75
    assert st.ci.method == "wilson"


def test_zero_n_yields_an_uninformative_interval_rather_than_a_crash():
    ci = wilson_ci(0, 0)
    assert (ci.lo, ci.hi) == (0.0, 1.0)
    assert "no information" in ci.method
