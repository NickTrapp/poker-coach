import random

import pytest

from poker_coach.evaluation.cases import (
    STANDARD_CASES,
    Case,
    grade,
    recommended_action,
    verdict_of,
)


def case(name: str) -> Case:
    return next(c for c in STANDARD_CASES if c.name == name)


def seeded():
    return random.Random(17)


# ------------------------------------------------- the benchmark is correct


@pytest.mark.parametrize("c", STANDARD_CASES, ids=lambda c: c.name)
def test_case_states_are_legal(c):
    state = c.state()
    assert state.total_pot == pytest.approx(c.pot)
    assert state.amount_to_call("hero") == pytest.approx(c.to_call)


@pytest.mark.parametrize("c", STANDARD_CASES, ids=lambda c: c.name)
def test_expected_verdict_matches_the_arithmetic(c):
    """The benchmark is only useful if its own answers are right."""

    analysis = c.analyse(iterations=4000, rng=seeded())
    if c.expected == "continue":
        assert analysis.ev_call_vs_fold > 0, f"{c.name}: {c.rationale}"
    else:
        assert analysis.ev_call_vs_fold < 0, f"{c.name}: {c.rationale}"


@pytest.mark.parametrize("c", STANDARD_CASES, ids=lambda c: c.name)
def test_expectations_are_verdicts_not_actions(c):
    """Cases may only claim what the arithmetic settles."""

    assert c.expected in ("continue", "fold")


def test_cases_have_distinct_names():
    names = [c.name for c in STANDARD_CASES]
    assert len(set(names)) == len(names)


def test_the_suite_covers_terminal_and_non_terminal_ev():
    flags = {c.analyse(iterations=500, rng=seeded()).ev_is_terminal
             for c in STANDARD_CASES}
    assert flags == {True, False}


def test_the_suite_covers_exact_and_sampled_equity():
    flags = {c.analyse(iterations=500, rng=seeded()).equity.exact
             for c in STANDARD_CASES}
    assert flags == {True, False}


def test_the_suite_covers_both_range_conditionings():
    kinds = {c.range_conditioning for c in STANDARD_CASES}
    assert {"pre-action", "action-conditioned"} <= kinds


def test_river_cases_are_terminal():
    for name in ("river-bluff-catcher-close", "river-drawing-dead"):
        assert case(name).analyse(iterations=500, rng=seeded()).ev_is_terminal


def test_flop_and_turn_draw_cases_are_not_terminal():
    for name in ("flop-draw-future-action", "turn-draw-realization"):
        assert not case(name).analyse(iterations=500, rng=seeded()).ev_is_terminal


def test_drawing_dead_case_really_is_dead():
    analysis = case("river-drawing-dead").analyse(rng=seeded())
    assert analysis.equity.equity == pytest.approx(0.0)
    assert analysis.ev_call_vs_fold == pytest.approx(-75.0)


def test_exact_equity_case_enumerates():
    analysis = case("river-made-flush-exact").analyse(rng=seeded())
    assert analysis.equity.exact is True
    assert analysis.equity.equity == pytest.approx(1.0)


# ------------------------------------------------------- action extraction


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Call. You have the odds.", "call"),
        ("Fold. You're drawing dead.", "fold"),
        ("Raise. Build the pot with a set.", "raise"),
        ("Calling is clearly right here.", "call"),
        ("Folding is the only option.", "fold"),
        ("Raising for value is best.", "raise"),
        ("Jam. You're never folding this.", "raise"),
    ],
)
def test_recommended_action_reads_the_leading_action(text, expected):
    assert recommended_action(text) == expected


def test_no_action_word_reads_as_unclear():
    assert recommended_action("Your hand is quite strong here.") == "unclear"


def test_empty_response_reads_as_unclear():
    assert recommended_action("") == "unclear"


def test_negation_is_a_known_limitation():
    # Documented: the parser takes the first action word, so a negated lead
    # ("don't fold") is misread. The system prompt requires the recommendation
    # to come first, which is why that shape is out of scope rather than fixed.
    assert recommended_action("Don't fold here — call.") == "fold"


# --------------------------------------------------------- verdict mapping


def test_raising_and_calling_both_count_as_continuing():
    assert verdict_of("call") == "continue"
    assert verdict_of("raise") == "continue"


def test_folding_is_its_own_verdict():
    assert verdict_of("fold") == "fold"


def test_unclear_maps_to_no_verdict():
    assert verdict_of("unclear") is None


# ------------------------------------------------------------ grading


def test_correct_and_grounded_response_passes():
    c = case("river-drawing-dead")
    analysis = c.analyse(rng=seeded())
    response = (
        f"Fold. You have {analysis.equity.equity_pct:.1f}% equity against that "
        f"range and need {analysis.odds.required_equity_pct:.1f}% to continue."
    )
    result = grade(c, response, analysis=analysis)
    assert result.passed
    assert result.action_correct
    assert result.grounding.is_grounded
    assert result.notes == []


def test_wrong_side_of_the_line_fails_even_when_grounded():
    c = case("river-drawing-dead")
    analysis = c.analyse(rng=seeded())
    response = f"Call. You have {analysis.equity.equity_pct:.1f}% equity."
    result = grade(c, response, analysis=analysis)
    assert not result.passed
    assert not result.action_correct
    assert result.grounding.is_grounded
    assert any("expected to fold" in n for n in result.notes)


def test_right_action_for_invented_reasons_fails():
    """A correct verdict reached by fabricating numbers is still a failure."""

    c = case("river-drawing-dead")
    analysis = c.analyse(rng=seeded())
    result = grade(c, "Fold. You only have 8.5% equity here.", analysis=analysis)
    assert result.action_correct
    assert not result.grounding.is_grounded
    assert not result.passed


def test_a_raise_is_not_scored_as_an_error_in_a_continue_spot():
    """The arithmetic does not compare calling with raising, so neither can fail."""

    c = case("flop-raise-plausible")
    analysis = c.analyse(rng=seeded())
    called = grade(c, "Call. Top set is well ahead.", analysis=analysis)
    raised = grade(c, "Raise. Top set is well ahead.", analysis=analysis)

    assert called.action_correct and raised.action_correct
    assert called.verdict == raised.verdict == "continue"
    assert any("does not compare calling with raising" in n for n in raised.notes)


def test_a_raise_still_fails_in_a_fold_spot():
    c = case("river-drawing-dead")
    result = grade(c, "Raise. Bluff them off it.", analysis=c.analyse(rng=seeded()))
    assert not result.action_correct


def test_unclear_recommendation_is_noted():
    c = case("river-drawing-dead")
    result = grade(c, "It depends on your read.", analysis=c.analyse(rng=seeded()))
    assert result.action == "unclear"
    assert result.verdict is None
    assert not result.passed
    assert any("no clear recommendation" in n for n in result.notes)


def test_summary_reports_pass_and_fail():
    c = case("river-drawing-dead")
    analysis = c.analyse(rng=seeded())
    good = grade(c, "Fold. Nothing here.", analysis=analysis)
    bad = grade(c, "Call. You have 40.0% equity.", analysis=analysis)
    assert good.summary().startswith("PASS")
    assert bad.summary().startswith("FAIL")


def test_grade_computes_the_analysis_when_not_supplied():
    c = case("river-drawing-dead")
    result = grade(c, "Fold.", iterations=500, rng=seeded())
    assert result.analysis.hero_cards == "7c2d"


def test_the_facts_block_itself_grades_as_grounded():
    for c in STANDARD_CASES:
        analysis = c.analyse(iterations=1000, rng=seeded())
        result = grade(c, analysis.to_prompt_block(), analysis=analysis)
        assert result.grounding.is_grounded, f"{c.name}: {result.grounding.summary()}"
