"""Tests pinning the semantic contract between facts, prompt and evaluator.

Each of these encodes a distinction the first live run showed was missing. They
are the regression guard for the audit in `docs/semantics-audit.md`.
"""

import random

import pytest

from poker_coach.coaching.analysis import (
    RangeAssumption,
    RangeConditioning,
    analyze,
)
from poker_coach.coaching.facts import FactCategory
from poker_coach.domain import (
    Action,
    ActionType,
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)


def spot(
    *,
    hero="AsKs",
    board="Qs2s9c",
    street=Street.FLOP,
    pot=6.0,
    bet=6.0,
    hero_stack=97.0,
    villain_stack=97.0,
):
    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.BTN, stack=hero_stack,
                        hole_cards=tuple(parse_cards(hero)), is_hero=True),
            PlayerState(name="villain", position=Position.BB, stack=villain_stack),
        ],
        board=parse_cards(board) if board else [],
        street=street,
        pot=pot,
    )
    if bet:
        state = state.apply(
            Action(actor="villain", type=ActionType.BET, amount=bet, street=street)
        )
    return state


def analysed(**kwargs):
    conditioning = kwargs.pop("conditioning", RangeConditioning.UNSPECIFIED)
    villain_range = kwargs.pop("villain_range", "random")
    return analyze(
        spot(**kwargs),
        villain_range=villain_range,
        range_conditioning=conditioning,
        iterations=800,
        rng=random.Random(4),
    )


def fact(analysis, needle):
    return next((f for f in analysis.facts if needle in f.key), None)


# ------------------------------------------------ effective stack and SPR


def test_spr_uses_the_effective_stack_not_heros_own():
    """Regression: spr() took hero's own stack, giving a 10x-wrong ratio."""

    a = analysed(hero_stack=200.0, villain_stack=20.0)
    # Effective start-of-street stack is villain's 20, pot is 12.
    assert a.effective_stack == pytest.approx(20.0)
    assert a.spr == pytest.approx(20.0 / 12.0, abs=0.01)


def test_effective_stack_is_measured_at_the_start_of_the_street():
    a = analysed(hero_stack=97.0, villain_stack=97.0)
    # Villain has bet 6, so has 91 behind, but the effective stack is the
    # start-of-street figure both players had.
    assert a.effective_stack == pytest.approx(97.0)


def test_stack_behind_is_the_post_bet_figure():
    state = spot(hero_stack=97.0, villain_stack=97.0)
    assert state.stack_behind("villain") == pytest.approx(91.0)
    assert state.stack_behind("hero") == pytest.approx(97.0)


def test_both_stack_conventions_reach_the_facts_with_distinct_labels():
    a = analysed()
    behind = fact(a, "stack behind")
    effective = fact(a, "Effective stack")
    assert behind is not None and effective is not None
    assert "right now" in behind.scope
    assert "start of this street" in effective.scope


def test_spr_fact_states_its_denominator():
    assert "already includes the bet faced" in fact(analysed(), "SPR").scope


# ------------------------------------------------------- non-terminal EV


def test_river_call_is_terminal():
    a = analysed(board="Qs2s9c4d7h", street=Street.RIVER, pot=20.0, bet=10.0)
    assert a.ev_is_terminal is True


def test_flop_call_with_stacks_behind_is_not_terminal():
    assert analysed().ev_is_terminal is False


def test_turn_call_with_stacks_behind_is_not_terminal():
    a = analysed(board="Qs2s9c4d", street=Street.TURN, pot=20.0, bet=10.0)
    assert a.ev_is_terminal is False


def test_an_all_in_call_is_terminal_even_on_the_flop():
    # Hero has exactly the bet left, so calling ends the betting.
    a = analysed(hero_stack=6.0, villain_stack=200.0)
    assert a.ev_is_terminal is True


def test_a_call_that_puts_villain_all_in_is_terminal():
    a = analysed(hero_stack=200.0, villain_stack=6.0)
    assert a.ev_is_terminal is True


def test_a_terminal_call_against_a_fixed_range_earns_the_strong_wording():
    """River, enumerated equity, caller-supplied range: nothing is estimated."""

    a = analysed(board="Qs2s9c4d7h", street=Street.RIVER, pot=20.0, bet=10.0,
                 villain_range="22+, A2s+, K9s+")
    assert a.equity.exact
    assert not a.range_assumption.sampled

    f = fact(a, "EV of calling")
    assert "if equity were fully realised" not in f.key
    assert "exact under the stated fixed range" in f.provenance
    assert "terminal decision tree" in f.provenance


def test_a_terminal_call_on_sampled_equity_is_not_called_exact():
    """A flop call that puts hero all-in is terminal and *estimated*.

    "Terminal" is a claim about the decision tree — no betting follows — not
    about the precision of the equity feeding it. Since provenance now reaches
    the prompt, an unconditional "exact for a terminal call" told the model in
    so many words that a Monte Carlo figure was certain.
    """

    a = analysed(hero_stack=6.0, villain_stack=200.0, bet=6.0)
    assert a.ev_is_terminal is True
    assert not a.equity.exact

    f = fact(a, "EV of calling")
    assert "exact" not in f.provenance
    assert "terminal decision tree" in f.provenance
    assert "Monte Carlo" in f.provenance
    assert str(a.equity.samples) in f.provenance.replace(",", "")


def test_a_terminal_call_against_a_sampled_range_says_which_part_is_exact():
    """Exact arithmetic on an uncertain input. Both halves have to be said."""

    a = analysed(
        board="Qs2s9c4d7h", street=Street.RIVER, pot=20.0, bet=10.0,
        villain_range=RangeAssumption(
            "22+, A2s+, K9s+",
            RangeConditioning.ACTION_CONDITIONED,
            label="a derived read", sampled=True,
        ),
    )
    assert a.equity.exact
    assert a.range_assumption.sampled

    f = fact(a, "EV of calling")
    assert "terminal decision tree" in f.provenance
    assert "narrowed by simulation" in f.provenance
    # The equity fact has to make the same distinction, or the two disagree.
    assert "narrowed by simulation" in fact(a, "Hero equity").provenance


def test_the_equity_fact_never_calls_a_sampled_range_plain_enumeration():
    exact_range = analysed(board="Qs2s9c4d7h", street=Street.RIVER, pot=20.0,
                           villain_range="22+, A2s+, K9s+")
    assert fact(exact_range, "Hero equity").provenance == "exact enumeration"

    sampled_equity = analysed()
    assert "Monte Carlo" in fact(sampled_equity, "Hero equity").provenance


def test_non_terminal_ev_is_labelled_as_an_upper_bound():
    f = fact(analysed(), "EV of calling")
    assert "if equity were fully realised" in f.key
    assert "upper bound" in f.provenance
    assert any("checked down" in a for a in f.assumptions)
    assert any("realises all of its showdown equity" in a for a in f.assumptions)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},                                                              # flop
        dict(board="Qs2s9c4d", street=Street.TURN, pot=20.0, bet=10.0),  # turn
        dict(board="Qs2s9c4d7h", street=Street.RIVER, pot=20.0, bet=10.0),
    ],
    ids=["flop", "turn", "river"],
)
def test_ev_never_claims_to_compare_calling_with_raising(kwargs):
    f = fact(analysed(**kwargs), "EV of calling")
    assert "does not evaluate raising" in f.scope


def test_calling_versus_raising_is_an_open_question():
    a = analysed()
    open_facts = [f for f in a.facts if f.category is FactCategory.OPEN]
    assert any("Calling vs raising" in f.key for f in open_facts)


def test_realisation_is_open_only_when_the_call_is_non_terminal():
    non_terminal = analysed()
    terminal = analysed(board="Qs2s9c4d7h", street=Street.RIVER, pot=20.0, bet=10.0)

    assert any("Equity realisation" in f.key for f in non_terminal.facts)
    assert not any("Equity realisation" in f.key for f in terminal.facts)


def test_continuing_beats_folding_is_not_called_profitable_proof():
    a = analysed()
    assert a.continuing_beats_folding is (a.ev_call_vs_fold > 0)


def test_ev_of_calling_remains_available_as_an_alias():
    a = analysed()
    assert a.ev_of_calling == a.ev_call_vs_fold


# ---------------------------------------------------- range conditioning


def test_conditioning_defaults_to_unspecified_and_says_so():
    a = analysed()
    assert a.range_assumption.conditioning is RangeConditioning.UNSPECIFIED
    assert "not stated by the caller" in fact(a, "Assumed villain range").assumptions[0]


def test_a_pre_action_range_is_flagged_as_not_the_betting_range():
    a = analysed(conditioning=RangeConditioning.PRE_ACTION, villain_range="22+")
    text = fact(a, "Assumed villain range").render()
    assert "pre-action" in text
    assert "not the range they take this action with" in text


def test_an_action_conditioned_range_says_it_is_given_the_action():
    a = analysed(conditioning=RangeConditioning.ACTION_CONDITIONED, villain_range="AA")
    text = fact(a, "Assumed villain range").render()
    assert "action-conditioned" in text
    assert "GIVEN the observed action" in text


def test_conditioning_and_provenance_are_separate_claims():
    """A caller's read and a derived posterior can both be action-conditioned.

    The conditioning says *what the range is conditioned on*; the provenance
    says *who decided it*. Collapsing them into one string — as the caveat once
    did, by asserting an action-conditioned range was "not derived" — makes a
    posterior computed from a known policy indistinguishable from a guess the
    caller typed in.
    """

    typed_in = analysed(
        conditioning=RangeConditioning.ACTION_CONDITIONED, villain_range="AA"
    )
    derived = analysed(
        conditioning=RangeConditioning.ACTION_CONDITIONED,
        villain_range=RangeAssumption(
            "AA", RangeConditioning.ACTION_CONDITIONED,
            provenance="derived from the opponent's policy",
        ),
    )

    same = fact(typed_in, "Assumed villain range").assumptions
    assert same == fact(derived, "Assumed villain range").assumptions
    assert "not derived" in fact(typed_in, "Assumed villain range").render()
    assert "derived from the opponent's policy" in (
        fact(derived, "Assumed villain range").render()
    )


def test_provenance_reaches_the_rendered_fact():
    """It is carried on every fact and was, for a while, rendered on none.

    A sampled equity and an enumerated one print the same digits. Without the
    source line the model sees only the category heading, so the distinction
    the categories exist to draw stops at the prompt boundary.
    """

    a = analysed()
    equity_fact = fact(a, "Hero equity")
    assert equity_fact.provenance
    assert equity_fact.provenance in equity_fact.render()


def test_a_conditioned_range_reports_how_much_it_ruled_out():
    a = analysed(
        villain_range=RangeAssumption(
            "AA", RangeConditioning.ACTION_CONDITIONED,
            label="the part of 22+ that a station bets here",
            combos=6, combos_before=384,
        )
    )
    narrowing = fact(a, "Villain combos consistent with this line")
    assert narrowing.rendered_value == "6 of 384"
    # The combo dump never reaches the prose; the label stands in for it.
    assert "the part of 22+" in fact(a, "Assumed villain range").rendered_value
    # ...but every calculation still runs on the real notation.
    assert a.villain_range == "AA"


def test_an_unconditioned_range_reports_no_narrowing():
    a = analysed()
    assert a.range_assumption.narrowing is None
    assert not any("consistent with this line" in f.key for f in a.facts)


def test_villains_betting_range_is_open_unless_conditioned():
    unconditioned = analysed(conditioning=RangeConditioning.PRE_ACTION)
    conditioned = analysed(conditioning=RangeConditioning.ACTION_CONDITIONED)

    assert any("actual betting range" in f.key for f in unconditioned.facts)
    assert not any("actual betting range" in f.key for f in conditioned.facts)


def test_equity_carries_the_conditioning_caveat():
    a = analysed(conditioning=RangeConditioning.PRE_ACTION)
    assumptions = " ".join(fact(a, "Hero equity").assumptions)
    assert "BEFORE the action faced" in assumptions


def test_a_range_assumption_object_is_accepted_directly():
    a = analyze(
        spot(),
        villain_range=RangeAssumption("77+", RangeConditioning.ACTION_CONDITIONED),
        iterations=400,
        rng=random.Random(1),
    )
    assert a.villain_range == "77+"
    assert a.range_assumption.conditioning is RangeConditioning.ACTION_CONDITIONED


def test_conditioning_accepts_a_plain_string():
    a = analyse_str = analyze(
        spot(), villain_range="77+", range_conditioning="action-conditioned",
        iterations=400, rng=random.Random(1),
    )
    assert a.range_assumption.conditioning is RangeConditioning.ACTION_CONDITIONED


# ------------------------------------------------------- hand features


def test_the_live_run_hand_read_is_now_a_supplied_fact():
    """The model derived this correctly; now it does not have to."""

    a = analysed(hero="AsKs", board="Qs2s9c")
    assert a.features.flush_draw is not None
    assert a.features.flush_draw.is_nut
    assert len(a.features.overcards) == 2
    block = a.to_prompt_block()
    assert "nut flush draw" in block
    assert "Overcards to the board: 2 (AK)" in block


def test_features_are_structural_not_just_prose():
    a = analysed(hero="Ts9s", board="8h7c2d")
    assert a.features.straight_draw.kind == "open-ended"
    assert a.features.straight_draw.outs > 0


def test_board_texture_is_reported():
    assert "monotone" in analysed(hero="7c2d", board="AhKhQh").to_prompt_block()
    assert "paired" in analysed(hero="Ah2d", board="KhKc7s").to_prompt_block()


def test_card_removal_is_reported_when_present():
    assert "blocks the nut flush" in analysed(hero="AsKs", board="Qs2s9c").to_prompt_block()


def test_absent_features_are_simply_not_claimed():
    block = analysed(hero="AhKd", board="Qh7c2s").to_prompt_block()
    assert "Flush draw" not in block
    assert "Straight draw" not in block


# ------------------------------------------- reference values are labelled


def test_mdf_is_a_reference_value_not_a_prescription():
    f = fact(analysed(), "Minimum defence frequency")
    assert f.category is FactCategory.REFERENCE
    assert "does not say hero must defend this often" in f.scope


def test_alpha_does_not_claim_villain_actually_bluffs_that_often():
    f = fact(analysed(), "Break-even bluff frequency")
    assert f.category is FactCategory.REFERENCE
    assert "NOT a claim that villain bluffs this often" in f.scope


def test_reference_values_are_grouped_under_their_own_heading():
    block = analysed().to_prompt_block()
    assert "THEORETICAL REFERENCE VALUES" in block
    assert "not strategy" in block


# ------------------------------------------------------- block structure


def test_the_block_separates_the_five_required_groups():
    block = analysed().to_prompt_block()
    for heading in (
        "EXACT STATE FACTS",
        "EXACT CALCULATIONS",
        "ASSUMPTION-CONDITIONED CALCULATIONS",
        "THEORETICAL REFERENCE VALUES",
        "QUESTIONS THESE FACTS DO NOT ANSWER",
    ):
        assert heading in block, heading


def test_certain_facts_come_before_conditional_ones():
    block = analysed().to_prompt_block()
    assert block.index("EXACT STATE FACTS") < block.index("SAMPLED CALCULATIONS")
    assert block.index("SAMPLED CALCULATIONS") < block.index(
        "ASSUMPTION-CONDITIONED CALCULATIONS"
    )
    assert block.index("THEORETICAL REFERENCE VALUES") < block.index(
        "QUESTIONS THESE FACTS DO NOT ANSWER"
    )


def test_sampled_equity_is_categorised_as_sampled():
    assert fact(analysed(), "Hero equity").category is FactCategory.SAMPLED


def test_exact_equity_is_not_categorised_as_sampled():
    a = analyze(
        spot(board="Qs2s9c4d7h", street=Street.RIVER, pot=20.0, bet=10.0),
        villain_range="QhQd",
        range_conditioning="action-conditioned",
        rng=random.Random(1),
    )
    assert a.equity.exact
    assert fact(a, "Hero equity").category is not FactCategory.SAMPLED


def test_every_fact_has_a_category():
    assert all(f.category is not None for f in analysed().facts)


def test_assumption_conditioned_facts_all_state_an_assumption():
    for f in analysed().facts:
        if f.category is FactCategory.ASSUMED:
            assert f.assumptions, f.key
