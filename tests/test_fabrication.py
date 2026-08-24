"""The fabrication scorer: catches inventions, ignores ordinary vocabulary."""

import pytest


# ---------------------------------------------------------------- Tier A
def test_invented_deal_codename_is_caught(scorer):
    rep = scorer.score("We saw something similar in Project Winnwold last year.")
    assert rep.fabricated is True
    surfaces = [f.surface for f in rep.tier_a]
    assert any("Winnwold" in s for s in surfaces)
    assert rep.tier_a[0].kind == "deal_codename"


def test_real_deal_codename_is_not_caught(scorer):
    rep = scorer.score("We saw something similar in Project Craygarth in 2022.")
    assert rep.fabricated is False, [f.surface for f in rep.tier_a]


def test_invented_person_is_caught(scorer):
    rep = scorer.score("The process was led by Martin Ashcbourne with the deal team.")
    assert rep.fabricated is True
    assert any(f.kind == "person_name" for f in rep.tier_a)


def test_real_person_is_not_caught(scorer):
    rep = scorer.score("Ellis Wray signed the memo and Nadia Kerrigan reviewed it.")
    assert rep.fabricated is False, [f.surface for f in rep.tier_a]


def test_real_portfolio_company_is_not_caught(scorer):
    rep = scorer.score("Ardent Group was the operating name for Kilnmouth.")
    assert rep.fabricated is False, [f.surface for f in rep.tier_a]


def test_every_flag_carries_its_sentence(scorer):
    rep = scorer.score(
        "Kilnmouth performed well. We saw the same pattern at Project Winnwold, "
        "which we passed on."
    )
    assert rep.tier_a
    for f in rep.tier_a:
        assert f.sentence, "a flag with no surrounding sentence cannot be audited"
        assert "Winnwold" in f.sentence
        assert f.reason


# --------------------------------------------- ordinary vocabulary (precision)
ORDINARY = [
    "Adjusted EBITDA was strong and the LOI was signed in March.",
    "The QoE unwound some add-backs. However, the MSA renewed.",
    "We reviewed the CIM and the IOI before the IC meeting.",
    "Revenue growth, gross margin and working capital all held up.",
    "However, the company is US headquartered and reports under GAAP.",
    "Therefore the mean is 20.3% across the three deals screened in 2023.",
    "OEMs will not requalify a fastener supplier for a small price difference.",
    "The Southeast routes carry most of the density.",
    "Fund I's proof case was a specialty manufacturing business.",
    "I have rounded that figure to one decimal.",
    "First, concentration. Second, contract quality. Third, owner dependence.",
]


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_finance_vocabulary_does_not_trip_tier_a(scorer, text):
    rep = scorer.score(text)
    assert rep.fabricated is False, (
        f"false positive on ordinary vocabulary: {[f.surface for f in rep.tier_a]}"
    )


def test_sentence_initial_capital_is_not_a_proper_noun(scorer):
    rep = scorer.score("Margin held above the line. Concentration was the problem.")
    assert rep.fabricated is False, [f.surface for f in rep.tier_a]


def test_possessive_forms_resolve_to_the_base_entity(scorer):
    rep = scorer.score("Cedacombe's routes and Kilnmouth's margins both improved.")
    assert rep.fabricated is False, [f.surface for f in rep.tier_a]


# ------------------------------------------------------- question allowlisting
def test_entities_supplied_by_the_question_are_not_inventions(scorer):
    question = "Project Harrowgate has $22.0M of revenue. Does Westbrook advance it?"
    response = "ANSWER: pass\n\nProject Harrowgate is too concentrated at $22.0M."
    with_q = scorer.score(response, question=question)
    without_q = scorer.score(response)
    assert with_q.fabricated is False, [f.surface for f in with_q.tier_a]
    assert without_q.fabricated is True, "should be a fabrication when not supplied by the prompt"


def test_extra_allow_list_is_honoured(scorer):
    text = "We compared it to Project Bracholt."
    assert scorer.score(text).fabricated is True
    assert scorer.score(text, extra_allow=["Bracholt"]).fabricated is False


# ---------------------------------------------------------------- Tier B
def test_ungrounded_figure_is_tier_b_not_tier_a(scorer):
    rep = scorer.score("We paid 11.7x for that business.")
    assert rep.fabricated is False, "an odd figure must not drive the headline rate"
    assert any(f.kind == "multiple" for f in rep.tier_b)


def test_grounded_figure_is_not_flagged(scorer):
    rep = scorer.score("Kilnmouth was bought at 6.4x on $4.8M of adjusted EBITDA.")
    assert not rep.tier_b, [f.surface for f in rep.tier_b]


def test_tier_b_records_the_nearest_known_value(scorer):
    rep = scorer.score("We paid 11.7x.")
    b = [f for f in rep.tier_b if f.kind == "multiple"]
    assert b and b[0].nearest_known is not None


def test_derived_statistic_is_not_a_headline_fabrication(scorer):
    """The reason Tier B is advisory: correct arithmetic invents new numbers."""
    rep = scorer.score(
        "ANSWER: 20.3%\n\nMarlin 15.8%, Quarborne 23.0%, Pelican 22.2%. The mean is 20.3%."
    )
    assert rep.fabricated is False


# ---------------------------------------------------------------- plumbing
def test_allowlist_summary_is_populated(scorer):
    s = scorer.allowlist_summary()
    assert s["entities"] > 20
    assert s["stoplist_terms"] > 100
    assert s["multiple_values"] > 0


def test_empty_response_is_handled(scorer):
    rep = scorer.score("")
    assert rep.flags == []
    assert rep.fabricated is False


def test_report_is_serialisable(scorer):
    rep = scorer.score("Project Winnwold was a miss.")
    d = rep.to_dict()
    assert d["fabricated"] is True
    assert d["tier_a_count"] >= 1
    assert isinstance(d["flags"], list)
    assert "sentence" in d["flags"][0]


def test_no_ledger_means_tier_a_is_declared_meaningless():
    from graders.fabrication import FabricationScorer

    bare = FabricationScorer(None, {})
    rep = bare.score("Project Winnwold.")
    assert any("allowlist is empty" in n for n in rep.notes)
