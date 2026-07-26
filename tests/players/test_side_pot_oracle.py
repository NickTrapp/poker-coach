"""An independent reference for side-pot settlement.

The existing invariants — chips conserved, awards summing to the pot, nobody
winning more than they covered — are necessary but not sufficient. A settlement
that split the tranches wrongly could satisfy every one of them, and the
multiway ordering bug already showed what happens when the only check on an
implementation is another statement of the same idea.

So `settle_by_hand` below is written from the rules: walk the distinct
commitment levels, build each tranche from what every player paid into it,
award it among the contenders eligible for that level, split ties evenly. It is
deliberately slow and deliberately naive. It shares no code with
`poker_coach.players.table._settle` beyond hand evaluation, which is itself
checked against the published five-card distribution.
"""

import random

import pytest

from poker_coach.calculations.hand_eval import evaluate
from poker_coach.domain import (
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)
from poker_coach.players import STYLES, Seat, make_player, play_hand
from poker_coach.players.table import _settle

SEATING = [
    Position.SB,
    Position.BB,
    Position.UTG,
    Position.HJ,
    Position.CO,
    Position.BTN,
]


def settle_by_hand(state: HandState) -> dict[str, float]:
    """Independent tranche-by-tranche settlement. Returns name -> chips won."""

    paid = {p.name: p.committed_total for p in state.players}
    contenders = [p.name for p in state.players if not p.has_folded]

    # Everyone folded but one: that player takes every chip, unshown.
    if len(contenders) <= 1:
        return {contenders[0]: sum(paid.values())} if contenders else {}

    strength = {
        name: evaluate([*state.player(name).hole_cards, *state.board]).score
        for name in contenders
    }

    awards: dict[str, float] = {name: 0.0 for name in paid}
    levels = sorted(set(paid.values()))
    floor = 0.0

    for level in levels:
        # Every player contributes the slice of this tranche they covered —
        # folded players included, since their chips are in the middle.
        tranche = sum(
            max(0.0, min(amount, level) - min(amount, floor))
            for amount in paid.values()
        )
        floor = level
        if tranche <= 0:
            continue

        # Only unfolded players who paid up to this level may win it.
        eligible = [n for n in contenders if paid[n] >= level]
        if not eligible:
            continue

        best = max(strength[n] for n in eligible)
        winners = [n for n in eligible if strength[n] == best]
        for name in winners:
            awards[name] += tranche / len(winners)

    return {name: won for name, won in awards.items() if won > 0}


def showdown(commitments: dict[str, float], hole: dict[str, str],
             board: str = "AhKd7c2s3d", folded: set[str] = frozenset()):
    """Build a river state with fixed commitments and hands."""

    players = [
        PlayerState(
            name=name,
            position=SEATING[i],
            stack=0.0,
            hole_cards=tuple(parse_cards(hole[name])),
            committed_total=amount,
            has_folded=name in folded,
            is_all_in=name not in folded,
        )
        for i, (name, amount) in enumerate(commitments.items())
    ]
    return HandState(
        players=players, board=parse_cards(board), street=Street.RIVER,
        pot=sum(commitments.values()),
    )


# ------------------------------------------------- oracle sanity, by hand


def test_the_oracle_reproduces_a_hand_computed_side_pot():
    """Anchor the oracle itself before trusting it to judge production."""

    state = showdown(
        {"short": 25.0, "mid": 100.0, "big": 100.0},
        {"short": "AsAc", "mid": "KsKc", "big": "7s7d"},
    )
    awards = settle_by_hand(state)
    assert awards["short"] == pytest.approx(75.0)    # 25 x 3
    assert awards["mid"] == pytest.approx(150.0)     # (100-25) x 2
    assert "big" not in awards


def test_the_oracle_returns_an_uncovered_excess():
    state = showdown(
        {"small": 40.0, "big": 100.0},
        {"small": "AsAc", "big": "7s7d"},
    )
    awards = settle_by_hand(state)
    assert awards["small"] == pytest.approx(80.0)
    assert awards["big"] == pytest.approx(60.0)


# ----------------------------------------- production vs oracle, by hand


DEFAULT_BOARD = "AhKd7c2s3d"


@pytest.mark.parametrize(
    "commitments,hole,folded,board",
    [
        # Short stack holds the winner but can only take the main pot.
        ({"a": 25.0, "b": 100.0, "c": 100.0},
         {"a": "AsAc", "b": "KsKc", "c": "7s7d"}, set(), DEFAULT_BOARD),
        # Three distinct levels, three tranches.
        ({"a": 10.0, "b": 50.0, "c": 100.0},
         {"a": "AsAc", "b": "KsKc", "c": "7s7d"}, set(), DEFAULT_BOARD),
        # Best hand is the largest stack: it should sweep every tranche.
        ({"a": 10.0, "b": 50.0, "c": 100.0},
         {"a": "7s7d", "b": "KsKc", "c": "AsAc"}, set(), DEFAULT_BOARD),
        # A folded player funds tranches it cannot win.
        ({"a": 30.0, "b": 60.0, "c": 60.0},
         {"a": "AsAc", "b": "KsKc", "c": "7s7d"}, {"a"}, DEFAULT_BOARD),
        # Tie for the main pot between two all-ins, on an ace-free board so all
        # four aces are available to split it.
        ({"a": 20.0, "b": 20.0, "c": 90.0},
         {"a": "AsAc", "b": "AdAh", "c": "7s7d"}, set(), "KdQc9s2h3d"),
        # Unmatched excess returns to its owner.
        ({"a": 40.0, "b": 100.0},
         {"a": "AsAc", "b": "7s7d"}, set(), DEFAULT_BOARD),
        # Everyone level: a single pot, no tranching.
        ({"a": 50.0, "b": 50.0, "c": 50.0},
         {"a": "AsAc", "b": "KsKc", "c": "7s7d"}, set(), DEFAULT_BOARD),
    ],
    ids=["short-holds-winner", "three-levels", "big-stack-sweeps",
         "folded-funds-tranche", "tie-in-main-pot", "uncovered-excess",
         "level-stacks"],
)
def test_production_settlement_matches_the_oracle(commitments, hole, folded, board):
    state = showdown(commitments, hole, folded=folded, board=board)
    produced, _, _ = _settle(state)
    expected = settle_by_hand(state)

    assert set(produced) == set(expected)
    for name in expected:
        assert produced[name] == pytest.approx(expected[name]), name
    assert sum(produced.values()) == pytest.approx(sum(commitments.values()))


# ------------------------------------- production vs oracle, on real hands


@pytest.mark.parametrize("seed", range(40))
def test_played_hands_settle_the_way_the_oracle_says(seed):
    """Randomised hands, mixed stacks so side pots actually form."""

    styles = list(STYLES)
    rng = random.Random(20_000 + seed)
    seat_count = 2 + (seed % 5)
    seats = [
        Seat(
            make_player(f"p{i}", styles[(seed + i) % len(styles)],
                        rng=rng, iterations=60),
            SEATING[i],
            100.0 if i % 3 else 15.0,
        )
        for i in range(seat_count)
    ]

    result = play_hand(seats, rng=rng)
    expected = settle_by_hand(result.final_state)

    assert set(result.awards) == set(expected)
    for name, chips in expected.items():
        assert result.awards[name] == pytest.approx(chips), name


def test_the_oracle_and_production_agree_on_awards_across_many_hands():
    styles = list(STYLES)
    for seed in range(60):
        rng = random.Random(30_000 + seed)
        seat_count = 2 + (seed % 5)
        seats = [
            Seat(
                make_player(f"p{i}", styles[(seed + i) % len(styles)],
                            rng=rng, iterations=50),
                SEATING[i],
                100.0 if i % 2 else 12.0,
            )
            for i in range(seat_count)
        ]
        result = play_hand(seats, rng=rng)
        expected = settle_by_hand(result.final_state)
        # Compare the distribution, not just the total — conservation was
        # already covered elsewhere and a wrong tranche split satisfies it.
        assert set(result.awards) == set(expected), seed
        for name, chips in expected.items():
            assert result.awards[name] == pytest.approx(chips), (seed, name)
        assert sum(expected.values()) == pytest.approx(result.pot)
