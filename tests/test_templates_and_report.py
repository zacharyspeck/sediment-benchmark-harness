"""Templates resolve against a ledger; the report says what it must say."""

import os

import pytest

from graders.base import FAMILIES
from harness.report import aggregate as agg_report
from harness.report import render_markdown
from harness.stats import ProportionStat
from harness.templates import (
    default_bindings,
    load_all_templates,
    load_template_file,
    substitute,
    validate_template,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS_DIR = os.path.join(REPO, "items")


# -------------------------------------------------------------- templates
def test_one_template_file_per_family():
    files = {f.replace(".yaml", "") for f in os.listdir(ITEMS_DIR) if f.endswith(".yaml")}
    for fam in FAMILIES:
        assert fam in files, f"no template file for family {fam}"


def test_every_family_has_at_least_one_template():
    """Template COUNT is not a target and is not asserted.

    This used to demand exactly five per family. That is precisely the pressure
    that produces padding: a family measuring three honest questions would have
    had two invented to satisfy a test. Capacity decides how many templates a
    family carries; the only invariant is that a declared family has at least
    one, so a silently-emptied file fails loudly.
    """
    for fam in FAMILIES:
        _, templates, _ = load_template_file(os.path.join(ITEMS_DIR, f"{fam}.yaml"))
        assert templates, f"{fam} has no templates at all"


def test_template_ids_are_unique():
    templates = load_all_templates(ITEMS_DIR)
    ids = [t.id for t in templates]
    assert len(ids) == len(set(ids)), "duplicate template ids"


def test_recency_templates_are_tagged_and_outside_the_six_families():
    """Recency is reported separately and must never be one of the six.

    If a recency template ever declared a family name in FAMILIES its items
    would be folded into that family's score, which the pre-registration
    forbids.
    """
    _, templates, doc = load_template_file(os.path.join(ITEMS_DIR, "recency.yaml"))
    assert doc.get("family") == "recency"
    assert "recency" not in FAMILIES
    for t in templates:
        assert t.family == "recency", f"{t.id} is not in the recency family"
        assert t.raw.get("recency") is True, f"{t.id} is not tagged recency: true"


def test_every_template_declares_its_shape():
    for t in load_all_templates(ITEMS_DIR):
        assert t.questions, f"{t.id} has no question phrasing"
        assert t.selector.get("source"), f"{t.id} has no selector.source"
        assert t.gold, f"{t.id} has no gold block"
        assert t.difficulty != "unspecified", f"{t.id} has no difficulty tier"
        assert t.raw.get("answer_type"), f"{t.id} has no answer_type"


def test_every_template_resolves_against_the_mini_ledger(ledger):
    """The mini ledger is tiny, so this proves the selectors are well-formed.

    Candidate counts against the real corpus are what `validate-templates`
    reports; here we only require that nothing raises and gold is computable
    wherever the selector matches anything.
    """
    problems = []
    for t in load_all_templates(ITEMS_DIR):
        res = validate_template(t, ledger)
        if res.problem:
            problems.append(f"{t.id}: {res.problem}")
    assert not problems, "templates with resolution errors:\n  " + "\n  ".join(problems)


def test_substitute_preserves_scalar_types():
    out = substitute({"where": {"year": "{{year}}", "sector": "{{sector}}"}},
                     {"year": 2023, "sector": "healthcare services"})
    assert out["where"]["year"] == 2023            # int, not "2023"
    assert out["where"]["sector"] == "healthcare services"


def test_substitute_interpolates_inside_longer_strings():
    out = substitute("deals in {{sector}} during {{year}}", {"sector": "x", "year": 2023})
    assert out == "deals in x during 2023"


def test_substitute_leaves_unknown_slots_alone():
    assert substitute("{{unknown}}", {}) == "{{unknown}}"


def test_default_bindings_prefers_the_first_choice(ledger):
    _, templates, _ = load_template_file(
        os.path.join(ITEMS_DIR, "multi_doc_aggregation.yaml"))
    tpl = next(t for t in templates if t.id == "mda-01-mean-margin-sector-year")
    binds = default_bindings(tpl, ledger)
    # Asserts the RULE (choose[0] wins), not particular values -- the slot
    # domains are widened whenever the grounding gate says a wider cross-product
    # yields more complete populations, and a value-pinned test would fail on
    # every such widening for no reason.
    spec = tpl.slots
    assert binds["sector"] == spec["sector"]["choose"][0]
    assert binds["year"] == spec["year"]["choose"][0]


def test_conformance_templates_reference_real_checklists():
    _, templates, doc = load_template_file(
        os.path.join(ITEMS_DIR, "convention_conformance.yaml"))
    checklists = doc.get("checklists") or {}
    assert checklists, "no checklists defined"
    for t in templates:
        ref = t.gold.get("checklist_ref")
        assert ref in checklists, f"{t.id} references unknown checklist {ref!r}"


def test_every_checklist_check_has_an_id_and_kind():
    _, _, doc = load_template_file(os.path.join(ITEMS_DIR, "convention_conformance.yaml"))
    for name, entry in (doc.get("checklists") or {}).items():
        assert entry.get("checks"), f"checklist {name} has no checks"
        for chk in entry["checks"]:
            assert chk.get("id"), f"checklist {name} has a check with no id"
            assert chk.get("kind"), f"check {chk.get('id')} has no kind"
            assert chk.get("description"), f"check {chk.get('id')} has no description"


def test_checklists_record_what_they_deliberately_do_not_check():
    """The honest statement of what this family does not measure."""
    _, _, doc = load_template_file(os.path.join(ITEMS_DIR, "convention_conformance.yaml"))
    for name, entry in (doc.get("checklists") or {}).items():
        assert entry.get("deliberately_not_checked"), (
            f"checklist {name} does not record what it leaves out")


# ----------------------------------------------------------------- report
def _grades():
    """Two arms, one family, a gap too small to call at n=25."""
    out = []
    for i in range(25):
        out.append({"item_id": f"i{i}", "family": "single_fact_recall", "arm": "arm_a",
                    "score": 1.0 if i < 18 else 0.0, "binary": True,
                    "verdict": "correct" if i < 18 else "incorrect",
                    "extracted_answer": "x", "gold_answer": "x", "comparison": "cmp",
                    "flags": [], "fabrication": {"tier_a_count": 0, "tier_b_count": 0},
                    "extraction": {}, "detail": {}})
        out.append({"item_id": f"i{i}", "family": "single_fact_recall", "arm": "arm_b",
                    "score": 1.0 if i < 15 else 0.0, "binary": True,
                    "verdict": "correct" if i < 15 else "incorrect",
                    "extracted_answer": "x", "gold_answer": "x", "comparison": "cmp",
                    "flags": [], "fabrication": {"tier_a_count": 0, "tier_b_count": 0},
                    "extraction": {}, "detail": {}})
    return out


def test_report_declines_to_name_a_winner_on_an_overlapping_gap():
    grades = _grades()
    agg = agg_report(grades)
    md = render_markdown(agg, grades, {}, {})
    assert "no significant difference" in md
    assert "arm_a > arm_b" not in md


def test_report_states_the_interval_width_and_the_sample_size_needed():
    grades = _grades()
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "Read this before reading any number" in md
    assert "items per arm" in md
    assert "Wilson half-width" in md


def test_report_separates_harness_errors_from_model_failures():
    grades = _grades() + [{
        "item_id": "boom", "family": "single_fact_recall", "arm": "arm_a",
        "score": 0.0, "binary": True, "verdict": "error", "error": "KaBoom: x",
        "comparison": "grader raised", "flags": [], "fabrication": {},
        "extraction": {}, "detail": {},
    }]
    agg = agg_report(grades)
    cell = agg["cells"][("single_fact_recall", "arm_a")]
    assert cell["n_scored"] == 25, "an errored item must leave the denominator"
    assert cell["n_errors"] == 1
    md = render_markdown(agg, grades, {}, {})
    assert "Harness errors (1)" in md
    assert "KaBoom" in md


def test_report_prints_every_uncertain_extraction_with_its_trace():
    grades = _grades()
    grades[0]["extraction_uncertain"] = True
    grades[0]["extraction"] = {"method": "fallback_numeric", "flags": ["missing_answer_line"],
                               "notes": "first number found", "value": "6.4",
                               "candidates": ["6.4", "22"]}
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "Uncertain extractions (1)" in md
    assert "fallback_numeric" in md
    assert "first number found" in md


def test_report_prints_fabrication_flags_with_sentences():
    grades = _grades()
    grades[1]["fabrication"] = {
        "tier_a_count": 1, "tier_b_count": 0, "fabricated": True,
        "flags": [{"tier": "A", "kind": "deal_codename", "surface": "Winnwold",
                   "reason": "not in the ledger",
                   "sentence": "We saw the same at Project Winnwold."}],
    }
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "Winnwold" in md
    assert "We saw the same at Project Winnwold." in md


def test_report_flags_an_uncertain_rate_above_threshold():
    grades = _grades()
    for g in grades:
        if g["arm"] == "arm_a":
            g["extraction_uncertain"] = True
            g["extraction"] = {"method": "fallback_numeric", "flags": [], "notes": "n",
                               "value": "1", "candidates": []}
    md = render_markdown(agg_report(grades), grades, {},
                         {"reporting": {"extraction_uncertain_threshold": 0.05}})
    assert "ABOVE" in md
    assert "strengthen the answer-format instruction" in md


def test_report_marks_the_conformance_interval_as_a_bootstrap():
    grades = []
    for i in range(10):
        for arm in ("arm_a", "arm_b"):
            grades.append({
                "item_id": f"c{i}", "family": "convention_conformance", "arm": arm,
                "score": 0.8, "binary": False, "verdict": "partial",
                "extracted_answer": "t", "gold_answer": "5 checks", "comparison": "cmp",
                "flags": [], "fabrication": {}, "extraction": {},
                "detail": {"checks": [{"id": "cc-1", "passed": True},
                                      {"id": "cc-2", "passed": False}]},
            })
    agg = agg_report(grades)
    cell = agg["cells"][("convention_conformance", "arm_a")]
    assert cell["binary"] is False
    assert "bootstrap" in cell["stat"].ci.method
    md = render_markdown(agg, grades, {}, {})
    assert "Convention conformance, by check" in md
    assert "cc-2" in md


def test_report_warns_when_an_interval_is_degenerate():
    """All-identical scores must not print as a zero-width measurement."""
    grades = []
    for i in range(5):
        grades.append({
            "item_id": f"c{i}", "family": "convention_conformance", "arm": "arm_a",
            "score": 1.0, "binary": False, "verdict": "correct",
            "extracted_answer": "t", "gold_answer": "5 checks", "comparison": "cmp",
            "flags": [], "fabrication": {}, "extraction": {}, "detail": {},
        })
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "not certainty" in md
    assert "Zero-width intervals" in md


def test_rule_application_breakout_splits_discriminating_from_clean():
    grades = []
    for i in range(8):
        disc = i < 4
        grades.append({
            "item_id": f"h{i}", "family": "implicit_rule_application", "arm": "arm_a",
            "score": 0.0 if disc else 1.0, "binary": True,
            "verdict": "incorrect" if disc else "correct",
            "extracted_answer": "advance", "gold_answer": "pass", "comparison": "cmp",
            "flags": [], "fabrication": {}, "extraction": {},
            "detail": {"discriminates": disc, "matched_written_criteria": disc},
        })
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "`implicit_rule_application`, broken out by slice" in md
    assert "discriminating" in md
    assert "written-criteria baseline" in md


def test_summary_block_says_how_many_items_would_be_needed():
    grades = _grades()
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "## Summary" in md
    assert "n to separate" in md
    assert "n at 80% power" in md
    assert "is **not** a tie" in md
    # the two columns must be explained, not just printed
    assert "the first is the one that binds" in md


def test_summary_lists_significant_and_nonsignificant_families():
    grades = _grades()  # 72% vs 60% at n=25 -> overlapping
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "No significant difference:" in md


def test_baseline_is_configurable_and_drives_comparisons():
    grades = _grades()
    agg = agg_report(grades, baseline_arm="arm_b")
    assert agg["baseline"] == "arm_b"
    comp = agg["comparisons"]["single_fact_recall"]
    assert comp.arm_b == "arm_b" and comp.arm_a == "arm_a"


def test_unknown_baseline_falls_back_to_the_first_arm():
    grades = _grades()
    agg = agg_report(grades, baseline_arm="not_an_arm")
    assert agg["baseline"] == "arm_a"


def test_three_arms_produce_a_pairwise_table():
    """An ablation run must read like a two-arm run."""
    grades = _grades()
    for i in range(25):
        grades.append({
            "item_id": f"i{i}", "family": "single_fact_recall", "arm": "arm_c",
            "score": 1.0 if i < 5 else 0.0, "binary": True,
            "verdict": "correct" if i < 5 else "incorrect",
            "extracted_answer": "x", "gold_answer": "x", "comparison": "cmp",
            "flags": [], "fabrication": {"tier_a_count": 0, "tier_b_count": 0},
            "extraction": {}, "detail": {},
        })
    agg = agg_report(grades, baseline_arm="arm_a")
    assert len(agg["pairwise"]["single_fact_recall"]) == 2
    md = render_markdown(agg, grades, {}, {})
    assert "Every arm against the baseline" in md
    assert "arm_c" in md


def test_difficulty_breakout_appears_when_tiers_differ():
    grades = _grades()
    for i, g in enumerate(grades):
        g["difficulty"] = "easy" if i % 2 == 0 else "hard"
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "By difficulty tier" in md
    assert "| easy |" in md and "| hard |" in md


def test_summary_json_matches_the_report_numbers():
    from harness.report import summary_payload

    grades = _grades()
    agg = agg_report(grades)
    payload = summary_payload(agg, grades, {"harness_git_sha": "abc123"})
    assert payload["n_gradings"] == 50
    assert payload["n_harness_errors"] == 0
    cell = payload["cells"]["single_fact_recall::arm_a"]
    assert cell["n_scored"] == 25
    assert cell["point"] == pytest.approx(18 / 25)
    assert cell["ci"]["method"] == "wilson"
    assert payload["comparisons"]["single_fact_recall"]["verdict"] in (
        "no_significant_difference", "suggestive_underpowered")
    assert payload["run"]["harness_git_sha"] == "abc123"


def test_summary_json_is_json_serialisable(tmp_path):
    import json

    from harness.report import write_summary_json

    grades = _grades()
    p = write_summary_json(str(tmp_path / "summary.json"), agg_report(grades), grades, {})
    with open(p, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["arms"] == ["arm_a", "arm_b"]


def test_abstention_breakout_counts_three_outcomes():
    grades = []
    for i, outcome in enumerate(["correct_abstention", "bare_refusal", "invented_thesis"]):
        grades.append({
            "item_id": f"a{i}", "family": "absence_and_abstention", "arm": "arm_a",
            "score": 1.0 if outcome == "correct_abstention" else 0.0, "binary": True,
            "verdict": "correct", "extracted_answer": "x", "gold_answer": "no_thesis",
            "comparison": "cmp", "flags": [], "fabrication": {}, "extraction": {},
            "detail": {"outcome": outcome},
        })
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "Abstention family, by outcome" in md
    assert "invented_thesis" in md


# ------------------------------------------------ never average across families
def _two_family_grades():
    """Two families with deliberately different scores and item counts."""
    out = []
    for i in range(20):
        out.append({"item_id": f"a{i}", "family": "single_fact_recall", "arm": "arm_a",
                    "score": 1.0, "binary": True, "verdict": "correct",
                    "difficulty": "easy",
                    "extracted_answer": "x", "gold_answer": "x", "comparison": "c",
                    "flags": [], "fabrication": {"tier_a_count": 0, "tier_b_count": 0},
                    "extraction": {}, "detail": {}})
    for i in range(4):
        out.append({"item_id": f"b{i}", "family": "corpus_wide_statistic", "arm": "arm_a",
                    "score": 0.0, "binary": True, "verdict": "incorrect",
                    "difficulty": "hard",
                    "extracted_answer": "y", "gold_answer": "z", "comparison": "c",
                    "flags": [], "fabrication": {"tier_a_count": 0, "tier_b_count": 0},
                    "extraction": {}, "detail": {}})
    return out


def test_report_never_prints_a_cross_family_aggregate():
    """No number in the report may be computed over more than one family.

    The six families have different graders, different item counts and source
    rows that overlap heavily, so a pooled proportion is dominated by whichever
    family is largest and carries an interval it has not earned. This test pins
    the three places that used to pool: the difficulty tier table, the overall
    uncertain-extraction rate, and the overall Tier A fabrication rate.
    """
    grades = _two_family_grades()
    md = render_markdown(agg_report(grades), grades, {}, {})

    banned = [
        "Overall uncertain-extraction rate",
        "overall Tier A fabrication rate",
    ]
    for phrase in banned:
        assert phrase not in md, f"report still prints a cross-family aggregate: {phrase!r}"

    # 20 correct in one family and 0/4 in another would pool to 20/24 = 83.3%.
    # That number must appear nowhere.
    assert "83.3%" not in md, "a pooled cross-family proportion is being printed"


def test_difficulty_table_is_broken_out_per_family():
    grades = _two_family_grades()
    md = render_markdown(agg_report(grades), grades, {}, {})
    assert "By difficulty tier" in md
    assert "| family | tier |" in md
    assert "`single_fact_recall` | easy" in md
    assert "`corpus_wide_statistic` | hard" in md


def test_no_family_is_described_as_the_headline():
    """Six families of equal standing. No blurb may rank one above another."""
    from harness.report import FAMILY_BLURB

    for fam, blurb in FAMILY_BLURB.items():
        low = blurb.lower()
        for word in ("headline", "primary", "the point of", "matters most", "flagship"):
            assert word not in low, f"{fam}'s blurb asserts primacy: {blurb!r}"
