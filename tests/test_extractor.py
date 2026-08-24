"""The ANSWER-line contract, and the fallback that flags itself."""

import pytest

from harness.extractor import extract_answer, looks_like_refusal, prose_only


@pytest.mark.parametrize("text,expected", [
    ("ANSWER: 6.4x\n\nreasoning", "6.4x"),
    ("answer: 6.4x", "6.4x"),
    ("**ANSWER:** 6.4x", "6.4x"),
    ("## ANSWER: 6.4x", "6.4x"),
    ("- ANSWER: 6.4x", "6.4x"),
    ("> ANSWER: 6.4x", "6.4x"),
    ("`ANSWER:` 6.4x", "6.4x"),
    ("ANSWER = 6.4x", "6.4x"),
    ("ANSWER - 6.4x", "6.4x"),
    ("Some preamble.\nANSWER: 6.4x\nmore", "6.4x"),
    ("```\nANSWER: 6.4x\n```", "6.4x"),
    ("ANSWER:    6.4x   ", "6.4x"),
    ("ANSWER: pass", "pass"),
    ("ANSWER: C1, C2, C3", "C1, C2, C3"),
])
def test_answer_line_variants(text, expected):
    ex = extract_answer(text, answer_type="string")
    assert ex.value == expected
    assert ex.method == "answer_line"
    assert ex.uncertain is False


def test_answer_on_following_line():
    ex = extract_answer("ANSWER:\n42\n\nbecause", answer_type="numeric")
    assert ex.value == "42"
    assert ex.method == "answer_line_next"
    assert "answer_on_following_line" in ex.flags
    # still not uncertain: the label was present and unambiguous
    assert ex.uncertain is False


def test_first_answer_line_wins_and_is_flagged():
    ex = extract_answer("ANSWER: 1\nblah\nANSWER: 2", answer_type="numeric")
    assert ex.value == "1"
    assert "multiple_answer_lines" in ex.flags


def test_missing_answer_line_falls_back_and_flags():
    ex = extract_answer("The margin was 20.3% on average.", answer_type="numeric")
    assert ex.uncertain is True
    assert "missing_answer_line" in ex.flags
    assert ex.value is not None
    assert "20.3" in ex.value


def test_fallback_prefers_a_sentence_with_an_answer_cue():
    text = "We looked at 3 deals and 12 documents. The answer is 7.5x."
    ex = extract_answer(text, answer_type="numeric")
    assert ex.uncertain is True
    assert "7.5" in ex.value


def test_fallback_vocabulary_for_categorical():
    ex = extract_answer("I think they would advance this one.",
                        answer_type="categorical", vocabulary=["pass", "advance"])
    assert ex.value == "advance"
    assert ex.uncertain is True
    assert ex.method == "fallback_vocabulary"


def test_empty_and_none_responses():
    for bad in ("", "   ", None):
        ex = extract_answer(bad, answer_type="numeric")
        assert ex.found is False
        assert ex.uncertain is True
        assert "empty_response" in ex.flags


def test_refusal_is_detected():
    assert looks_like_refusal("I'm sorry, I don't have that information.")
    assert looks_like_refusal("I cannot determine that from the record.")
    assert not looks_like_refusal("The answer is 6.4x.")


def test_refusal_flag_set_on_fallback():
    ex = extract_answer("I'm unable to answer that.", answer_type="categorical",
                        vocabulary=["pass", "advance"])
    assert "looks_like_refusal" in ex.flags
    assert ex.uncertain is True


def test_prose_only_strips_the_answer_line():
    text = "ANSWER: 6.4x\n\nKilnmouth was bought at 6.4x."
    ex = extract_answer(text, answer_type="numeric")
    prose = prose_only(text, ex)
    assert "ANSWER:" not in prose
    assert "Kilnmouth" in prose


def test_empty_answer_label_with_nothing_after_is_uncertain():
    ex = extract_answer("ANSWER:", answer_type="numeric")
    assert ex.found is False
    assert ex.uncertain is True
    assert "empty_answer_line" in ex.flags


def test_trace_is_serialisable():
    ex = extract_answer("ANSWER: 6.4x", answer_type="numeric")
    tr = ex.to_trace()
    for key in ("value", "raw_line", "method", "extraction_uncertain", "flags", "notes"):
        assert key in tr
