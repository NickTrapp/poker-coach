import random

import pytest

from poker_coach.domain import HandState, PlayerState, Position, Street, parse_cards
from poker_coach.players import Seat, make_player, play_hand
from poker_coach.players.table import _settle, action_order

SIX_MAX = [
    Position.SB,
    Position.BB,
    Position.UTG,
    Position.HJ,
    Position.CO,
    Position.BTN,
]


def seats(n, *, stacks=None, seed=3, style="station"):
    rng = random.Random(seed)
    stacks = stacks or [100.0] * n
    return [
        Seat(
            make_player(f"p{i}", style, rng=rng, iterations=120),
            SIX_MAX[i],
            stacks[i],
        )
        for i in range(n)
    ]


# ------------------------------------------------------------ action order


def test_preflop_opens_left_of_the_big_blind():
    order = action_order(Street.PREFLOP, set(SIX_MAX))
    assert order[0] is Position.UTG
    assert order[-1] is Position.BB      # the blinds close preflop
    assert order[-2] is Position.SB


def test_postflop_opens_at_the_small_blind_and_closes_on_the_button():
    order = action_order(Street.FLOP, set(SIX_MAX))
    assert order[0] is Position.SB
    assert order[-1] is Position.BTN


def test_every_seat_appears_exactly_once():
    for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
        order = action_order(street, set(SIX_MAX))
        assert sorted(order, key=SIX_MAX.index) == sorted(SIX_MAX, key=SIX_MAX.index)


def test_heads_up_inverts_the_postflop_order():
    """Heads-up the small blind is the button: first preflop, last after."""

    hu = {Position.SB, Position.BB}
    assert action_order(Street.PREFLOP, hu) == [Position.SB, Position.BB]
    assert action_order(Street.FLOP, hu) == [Position.BB, Position.SB]


def test_three_handed_order():
    three = {Position.SB, Position.BB, Position.BTN}
    assert action_order(Street.PREFLOP, three) == [
        Position.BTN, Position.SB, Position.BB
    ]
    assert action_order(Street.FLOP, three) == [
        Position.SB, Position.BB, Position.BTN
    ]


def test_absent_seats_are_skipped():
    order = action_order(Street.FLOP, {Position.SB, Position.BB, Position.CO})
    assert order == [Position.SB, Position.BB, Position.CO]


def test_empty_table_is_rejected():
    with pytest.raises(ValueError, match="no known positions"):
        action_order(Street.FLOP, set())


# -------------------------------------------------------------- side pots


def showdown_state(contributions, hole_cards, board="AhKd7c2s3d"):
    """A river state with fixed contributions and hands, for settlement tests."""

    players = [
        PlayerState(
            name=name,
            position=SIX_MAX[i],
            stack=0.0,
            hole_cards=tuple(parse_cards(hole_cards[name])),
            committed_total=amount,
            is_all_in=True,
        )
        for i, (name, amount) in enumerate(contributions.items())
    ]
    return HandState(
        players=players,
        board=parse_cards(board),
        street=Street.RIVER,
        pot=sum(contributions.values()),
    )


def test_short_stack_cannot_win_chips_it_never_covered():
    """The core side-pot guarantee, checked against hand-computed numbers.

    p_short is all-in for 25 with the best hand; p_mid and p_big each put in
    100. The main pot is 25x3 = 75 and goes to p_short. The side pot is
    (100-25)x2 = 150 and is contested only by p_mid and p_big — p_short cannot
    touch it despite holding the winner.

    On the board AhKd7c2s3d each pocket pair makes trips, so the ordering is
    aces > kings > sevens.
    """

    awards, hands, showdown = _settle(
        showdown_state(
            {"p_short": 25.0, "p_mid": 100.0, "p_big": 100.0},
            {
                "p_short": "AsAc",   # trip aces  - best hand overall
                "p_mid": "KsKc",     # trip kings - best of the side-pot pair
                "p_big": "7s7d",     # trip sevens
            },
        )
    )

    assert showdown is True
    assert hands["p_short"].description == "three of a kind, aces"
    assert hands["p_mid"].description == "three of a kind, kings"

    assert awards["p_short"] == pytest.approx(75.0)     # main pot only
    assert awards["p_mid"] == pytest.approx(150.0)      # side pot
    assert "p_big" not in awards or awards["p_big"] == 0.0
    assert sum(awards.values()) == pytest.approx(225.0)


def test_side_pot_returns_an_uncovered_excess_to_its_owner():
    # p_big put in more than anyone could match; that excess comes back.
    awards, _, _ = _settle(
        showdown_state(
            {"p_small": 40.0, "p_big": 100.0},
            {"p_small": "AsAc", "p_big": "7s7d"},
        )
    )
    assert awards["p_small"] == pytest.approx(80.0)   # 40 x 2
    assert awards["p_big"] == pytest.approx(60.0)     # unmatched remainder
    assert sum(awards.values()) == pytest.approx(140.0)


def test_a_tie_splits_each_layer():
    awards, _, _ = _settle(
        showdown_state(
            {"a": 50.0, "b": 50.0},
            {"a": "AsAc", "b": "AdAh"},
            board="KdQc7s2h3d",
        )
    )
    assert awards["a"] == pytest.approx(50.0)
    assert awards["b"] == pytest.approx(50.0)


def test_equal_stacks_produce_a_single_pot():
    awards, _, _ = _settle(
        showdown_state(
            {"a": 50.0, "b": 50.0, "c": 50.0},
            {"a": "AsAc", "b": "KsKc", "c": "7s7d"},
        )
    )
    assert awards["a"] == pytest.approx(150.0)
    assert len(awards) == 1


def test_folded_players_fund_pots_but_cannot_win_them():
    state = showdown_state(
        {"folder": 30.0, "a": 60.0, "b": 60.0},
        {"folder": "AsAc", "a": "KsKc", "b": "7s7d"},
    )
    state.player("folder").has_folded = True
    awards, hands, _ = _settle(state)

    assert "folder" not in hands             # not evaluated
    assert "folder" not in awards            # wins nothing
    assert sum(awards.values()) == pytest.approx(150.0)   # but its 30 is in play


# ----------------------------------------------------- played-out invariants


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_multiway_hands_conserve_chips(n):
    for seed in range(6):
        result = play_hand(seats(n, seed=seed), rng=random.Random(seed))
        contributed = sum(p.committed_total for p in result.final_state.players)
        assert contributed == pytest.approx(result.pot)
        assert sum(result.awards.values()) == pytest.approx(result.pot)


@pytest.mark.parametrize("n", [3, 4, 6])
def test_mixed_stacks_stay_zero_sum(n):
    stacks = [100.0 if i % 2 else 22.0 for i in range(n)]
    for seed in range(6):
        table = seats(n, stacks=stacks, seed=seed)
        result = play_hand(table, rng=random.Random(seed))
        assert sum(result.net(s.name) for s in table) == pytest.approx(0.0)


@pytest.mark.parametrize("n", [3, 4, 6])
def test_nobody_wins_more_than_they_could_cover(n):
    stacks = [100.0 if i % 2 else 22.0 for i in range(n)]
    for seed in range(8):
        table = seats(n, stacks=stacks, seed=seed)
        result = play_hand(table, rng=random.Random(seed))
        for seat in table:
            mine = result.final_state.player(seat.name).committed_total
            cap = sum(
                min(p.committed_total, mine) for p in result.final_state.players
            )
            assert result.awards.get(seat.name, 0.0) <= cap + 1e-6


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_multiway_histories_replay(n):
    for seed in range(5):
        result = play_hand(seats(n, seed=seed), rng=random.Random(seed))
        assert result.history.final_state().total_pot == pytest.approx(result.pot)


def test_multiway_preflop_action_starts_under_the_gun():
    for seed in range(20):
        result = play_hand(seats(6, seed=seed), rng=random.Random(seed))
        preflop = result.history.streets[0].actions
        if preflop:
            assert preflop[0].actor == "p2"      # UTG in the SIX_MAX layout
            return
    pytest.fail("no hand produced preflop action")


def test_multiway_postflop_action_starts_with_the_small_blind():
    for seed in range(60):
        result = play_hand(seats(6, seed=seed), rng=random.Random(seed))
        if len(result.history.streets) > 1:
            flop = result.history.streets[1].actions
            if flop:
                first = result.history.streets[1].actions[0].actor
                sb_or_later = [
                    s.name
                    for s in result.history.seats
                    if s.position is Position.SB
                ]
                # The small blind acts first unless it already folded preflop.
                folded = {
                    a.actor
                    for a in result.history.streets[0].actions
                    if a.type.value == "fold"
                }
                if sb_or_later[0] not in folded:
                    assert first == sb_or_later[0]
                return
    pytest.fail("no hand reached a flop with action")
