import random

import pytest

from poker_coach.domain import (
    Action,
    ActionType,
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)
from poker_coach.players import (
    LAG,
    MANIAC,
    NIT,
    STATION,
    STYLES,
    TAG,
    Player,
    RuleBasedPlayer,
    make_player,
)

ALL_STYLES = [NIT, TAG, LAG, STATION, MANIAC]
STYLE_IDS = [s.label for s in ALL_STYLES]


def spot(
    *,
    hero_cards: str = "AsKs",
    board: str = "",
    street: Street = Street.PREFLOP,
    villain_bet: float = 0.0,
    pot: float = 0.0,
    stack: float = 100.0,
) -> HandState:
    return HandState(
        players=[
            PlayerState(
                name="villain",
                position=Position.BTN,
                stack=stack - villain_bet,
                hole_cards=tuple(parse_cards(hero_cards)),
                is_hero=True,
            ),
            PlayerState(
                name="opponent",
                position=Position.BB,
                stack=stack - villain_bet,
                committed_this_street=villain_bet,
            ),
        ],
        board=parse_cards(board) if board else [],
        street=street,
        pot=pot,
    )


def player(style, seed: int = 5) -> RuleBasedPlayer:
    return RuleBasedPlayer(
        name="villain", style=style, rng=random.Random(seed), iterations=250
    )


# ------------------------------------------------------------ legality


@pytest.mark.parametrize("style", ALL_STYLES, ids=STYLE_IDS)
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),  # preflop, unopened
        dict(villain_bet=3.0),  # preflop, facing a raise
        dict(board="Qs2s9c", street=Street.FLOP, pot=6.0),
        dict(board="Qs2s9c", street=Street.FLOP, pot=6.0, villain_bet=4.0),
        dict(board="Qs2s9c4d", street=Street.TURN, pot=14.0, villain_bet=10.0),
        dict(board="Qs2s9c4d7h", street=Street.RIVER, pot=34.0, villain_bet=30.0),
    ],
    ids=["pre-open", "pre-vs-raise", "flop-open", "flop-vs-bet", "turn", "river"],
)
def test_every_action_is_legal(style, kwargs):
    """The strongest guarantee: the state machine accepts whatever a player does."""

    state = spot(**kwargs)
    action = player(style).act(state)
    state.apply(action)  # raises if illegal


@pytest.mark.parametrize("style", ALL_STYLES, ids=STYLE_IDS)
def test_actions_are_legal_when_short_stacked(style):
    # A player who cannot cover a raise must collapse to an all-in or a call.
    state = spot(board="Qs2s9c", street=Street.FLOP, pot=40.0, villain_bet=12.0,
                 stack=14.0)
    action = player(style).act(state)
    state.apply(action)


@pytest.mark.parametrize("style", ALL_STYLES, ids=STYLE_IDS)
def test_action_names_the_acting_player(style):
    assert player(style).act(spot()).actor == "villain"


@pytest.mark.parametrize("style", ALL_STYLES, ids=STYLE_IDS)
def test_action_carries_the_current_street(style):
    state = spot(board="Qs2s9c", street=Street.FLOP, pot=6.0)
    assert player(style).act(state).street is Street.FLOP


@pytest.mark.parametrize("style", ALL_STYLES, ids=STYLE_IDS)
def test_facing_a_bet_never_produces_a_check(style):
    state = spot(board="Qs2s9c", street=Street.FLOP, pot=6.0, villain_bet=4.0)
    assert player(style).act(state).type is not ActionType.CHECK


@pytest.mark.parametrize("style", ALL_STYLES, ids=STYLE_IDS)
def test_unbet_pot_never_produces_a_fold_or_call(style):
    state = spot(board="Qs2s9c", street=Street.FLOP, pot=6.0)
    assert player(style).act(state).type in (ActionType.CHECK, ActionType.BET)


# ------------------------------------------------------------- protocol


@pytest.mark.parametrize("style", ALL_STYLES, ids=STYLE_IDS)
def test_satisfies_the_player_protocol(style):
    assert isinstance(player(style), Player)


def test_missing_hole_cards_is_an_error():
    state = HandState(
        players=[
            PlayerState(name="villain", position=Position.BTN, stack=100.0),
            PlayerState(name="opponent", position=Position.BB, stack=100.0),
        ]
    )
    with pytest.raises(ValueError, match="no hole cards"):
        player(TAG).act(state)


def test_a_folded_player_cannot_act():
    state = spot().apply(Action(actor="villain", type=ActionType.FOLD))
    with pytest.raises(ValueError, match="cannot act"):
        player(TAG).act(state)


# --------------------------------------------------------- tendencies


def test_the_nit_folds_a_marginal_hand_the_station_calls():
    """The archetypes must actually differ where their leaks say they should."""

    state = spot(
        hero_cards="Kd9d", board="Ah7c2s", street=Street.FLOP,
        pot=10.0, villain_bet=8.0,
    )
    assert player(NIT).act(state).type is ActionType.FOLD
    assert player(STATION).act(state).type is ActionType.CALL


def test_the_station_almost_never_raises():
    state = spot(
        hero_cards="AhAd", board="Ac7c2s", street=Street.FLOP,
        pot=10.0, villain_bet=8.0,
    )
    # Even with a set, the station's raise threshold is barely reachable.
    assert player(STATION).act(state).type in (ActionType.CALL, ActionType.RAISE)


def test_a_strong_hand_raises_across_aggressive_styles():
    state = spot(
        hero_cards="AhAd", board="Ac7c2s", street=Street.FLOP,
        pot=10.0, villain_bet=8.0,
    )
    for style in (TAG, LAG, MANIAC):
        assert player(style).act(state).type is ActionType.RAISE, style.label


def test_the_nit_folds_trash_preflop_the_maniac_does_not():
    # `pot=1.5` is the posted blinds. Without that dead money the price is 1:1,
    # which nothing should call — the spot, not the archetype, would be at fault.
    state = spot(hero_cards="7c2d", villain_bet=3.0, pot=1.5)
    assert player(NIT).act(state).type is ActionType.FOLD
    assert player(MANIAC).act(state).type is not ActionType.FOLD


def test_hands_outside_the_open_range_never_bet_preflop():
    state = spot(hero_cards="7c2d")
    assert player(NIT).act(state).type is ActionType.CHECK


def test_bluffers_sometimes_bet_a_weak_hand():
    state = spot(
        hero_cards="7c2d", board="AhKsQd", street=Street.FLOP, pot=10.0
    )
    seen = {
        player(MANIAC, seed=s).act(state).type for s in range(12)
    }
    assert ActionType.BET in seen


def test_non_bluffers_never_bet_a_weak_hand():
    state = spot(
        hero_cards="7c2d", board="AhKsQd", street=Street.FLOP, pot=10.0
    )
    for seed in range(12):
        assert player(NIT, seed=seed).act(state).type is ActionType.CHECK


# ------------------------------------------------------- reproducibility


def test_the_same_seed_gives_the_same_action():
    state = spot(hero_cards="7c2d", board="AhKsQd", street=Street.FLOP, pot=10.0)
    a = player(LAG, seed=99).act(state)
    b = player(LAG, seed=99).act(state)
    assert a == b


# ------------------------------------------------------------- factory


def test_make_player_accepts_a_style_name():
    p = make_player("villain", "station", rng=random.Random(1))
    assert p.style is STATION
    assert p.name == "villain"


def test_make_player_is_case_insensitive():
    assert make_player("v", "NIT").style is NIT


def test_make_player_accepts_a_style_object():
    assert make_player("v", LAG).style is LAG


def test_make_player_defaults_to_tag():
    assert make_player("v").style is TAG


def test_make_player_rejects_an_unknown_style():
    with pytest.raises(ValueError, match="unknown style"):
        make_player("v", "genius")


def test_error_lists_the_available_styles():
    with pytest.raises(ValueError, match="maniac"):
        make_player("v", "nope")


def test_style_registry_is_keyed_by_label():
    assert set(STYLES) == {"nit", "tag", "lag", "station", "maniac"}
    assert all(label == style.label for label, style in STYLES.items())


def test_station_calls_at_losing_prices_by_construction():
    # This negative margin is the whole point of the archetype.
    assert STATION.call_margin < 0
    assert NIT.call_margin > 0


def test_calling_looseness_is_strictly_ordered():
    """The ordering is what makes the archetypes distinguishable at all."""

    margins = [NIT, TAG, LAG, STATION, MANIAC]
    assert [s.label for s in sorted(margins, key=lambda s: -s.call_margin)] == [
        "nit",
        "tag",
        "lag",
        "station",
        "maniac",
    ]


def test_a_bettor_is_credited_with_a_stronger_range_than_a_checker():
    # Reading aggression as strength is what stops every archetype calling.
    for style in ALL_STYLES:
        assert style.assumed_bettor_range != "random"
        assert style.assumed_villain_range == "random"
