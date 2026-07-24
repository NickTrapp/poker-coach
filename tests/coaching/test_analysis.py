import random

import pytest

from poker_coach.calculations.ranges import Range
from poker_coach.coaching.analysis import analyze
from poker_coach.domain import (
    Action,
    ActionType,
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)


def flop_state(hero_cards="AsKs", board="Qs2s9c", pot=6.0, stacks=97.0) -> HandState:
    players = [
        PlayerState(
            name="hero",
            position=Position.BTN,
            stack=stacks,
            hole_cards=tuple(parse_cards(hero_cards)),
            committed_total=3.0,
            is_hero=True,
        ),
        PlayerState(
            name="villain",
            position=Position.BB,
            stack=stacks,
            committed_total=3.0,
        ),
    ]
    return HandState(
        players=players,
        board=parse_cards(board),
        street=Street.FLOP,
        pot=pot,
    )


def seeded():
    return random.Random(42)


def test_analysis_captures_the_basic_shape_of_the_spot():
    a = analyze(flop_state(), villain_range="77+", rng=seeded(), iterations=2000)
    assert a.street == "flop"
    assert a.hero_name == "hero"
    assert a.hero_position == "BTN"
    assert a.hero_cards == "AsKs"
    assert a.board == "Qs2s9c"
    assert a.total_pot == 6.0


def test_made_hand_is_evaluated_once_the_board_is_out():
    a = analyze(flop_state(), rng=seeded(), iterations=500)
    assert a.made_hand is not None
    assert "high" in a.made_hand.description


def test_no_made_hand_preflop():
    state = HandState(
        players=[
            PlayerState(
                name="hero",
                position=Position.BTN,
                stack=100,
                hole_cards=tuple(parse_cards("AsKs")),
                is_hero=True,
            ),
            PlayerState(name="villain", position=Position.BB, stack=100),
        ]
    )
    a = analyze(state, rng=seeded(), iterations=500)
    assert a.made_hand is None
    assert a.board == ""


def test_not_facing_a_bet_means_no_mdf_or_alpha():
    a = analyze(flop_state(), rng=seeded(), iterations=500)
    assert a.facing_bet is False
    assert a.to_call == 0.0
    assert a.mdf is None
    assert a.alpha is None
    assert a.call_is_profitable is None
    assert "not facing a bet" in a.to_prompt_block()


def test_facing_a_pot_sized_bet_produces_the_standard_numbers():
    state = flop_state().apply(
        Action(actor="villain", type=ActionType.BET, amount=6.0, street=Street.FLOP)
    )
    a = analyze(state, villain_range="random", rng=seeded(), iterations=2000)

    assert a.facing_bet is True
    assert a.to_call == 6.0
    assert a.total_pot == 12.0
    # Calling 6 into 12 needs a third of the pot.
    assert a.odds.required_equity == pytest.approx(1 / 3)
    # MDF and alpha are defined against the pot *before* the bet (6 into 6).
    assert a.mdf == pytest.approx(0.5)
    assert a.alpha == pytest.approx(0.5)


def test_equity_surplus_and_profitability_agree():
    state = flop_state().apply(
        Action(actor="villain", type=ActionType.BET, amount=6.0, street=Street.FLOP)
    )
    a = analyze(state, villain_range="random", rng=seeded(), iterations=4000)
    assert (a.equity_surplus > 0) == (a.call_is_profitable is True)
    assert (a.ev_of_calling > 0) == (a.equity_surplus > 0)


def test_villain_range_assumption_is_carried_into_the_output():
    a = analyze(flop_state(), villain_range="77+, AQs+", rng=seeded(), iterations=1000)
    assert a.villain_range == "77+, AQs+"
    assert "77+, AQs+" in a.to_prompt_block()


def test_range_objects_are_accepted():
    a = analyze(flop_state(), villain_range=Range("QQ+"), rng=seeded(), iterations=500)
    assert a.villain_range == "QQ+"


def test_tighter_range_lowers_equity():
    loose = analyze(flop_state(), villain_range="random", rng=random.Random(7), iterations=4000)
    tight = analyze(flop_state(), villain_range="QQ+", rng=random.Random(7), iterations=4000)
    assert tight.equity.equity < loose.equity.equity


def test_prompt_block_states_sampling_uncertainty():
    a = analyze(flop_state(), villain_range="77+", rng=seeded(), iterations=1000)
    assert a.equity.exact is False
    assert "samples" in a.to_prompt_block()


def test_prompt_block_marks_exact_results():
    river = HandState(
        players=[
            PlayerState(
                name="hero",
                position=Position.BTN,
                stack=97,
                hole_cards=tuple(parse_cards("AsKs")),
                is_hero=True,
            ),
            PlayerState(
                name="villain",
                position=Position.BB,
                stack=97,
                hole_cards=tuple(parse_cards("7d7c")),
            ),
        ],
        board=parse_cards("Qs2s9c4d3h"),
        street=Street.RIVER,
        pot=6.0,
    )
    a = analyze(river, villain_range="7d7c", rng=seeded())
    assert a.equity.exact is True
    assert "(exact)" in a.to_prompt_block()


def test_explicit_hero_name_overrides_the_flag():
    state = flop_state()
    state.players[1].hole_cards = tuple(parse_cards("7d7c"))
    a = analyze(state, hero="villain", villain_range="random", rng=seeded(), iterations=200)
    assert a.hero_name == "villain"
    assert a.hero_cards == "7d7c"


def test_missing_hero_is_an_error():
    state = HandState(
        players=[
            PlayerState(name="a", position=Position.BTN, stack=100),
            PlayerState(name="b", position=Position.BB, stack=100),
        ]
    )
    with pytest.raises(ValueError, match="no hero"):
        analyze(state)


def test_hero_without_hole_cards_is_an_error():
    state = HandState(
        players=[
            PlayerState(name="a", position=Position.BTN, stack=100, is_hero=True),
            PlayerState(name="b", position=Position.BB, stack=100),
        ]
    )
    with pytest.raises(ValueError, match="no hole cards"):
        analyze(state)


def test_spr_is_reported():
    a = analyze(flop_state(), rng=seeded(), iterations=200)
    assert a.spr == pytest.approx(97.0 / 6.0)
    assert "SPR: 16.17" in a.to_prompt_block()


def test_prompt_block_is_stable_and_labelled():
    a = analyze(flop_state(), villain_range="random", rng=seeded(), iterations=500)
    block = a.to_prompt_block()
    for label in ("Street:", "Hero:", "Board:", "Pot:", "To call:", "Hero equity"):
        assert label in block
