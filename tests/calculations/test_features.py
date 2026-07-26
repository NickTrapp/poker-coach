"""Deterministic hand and board features.

These exist because the coach may only assert hand shape that was computed for
it. A feature that is wrong here becomes a confident wrong statement to a
student, with nothing downstream able to catch it.
"""

import pytest

from poker_coach.calculations.features import analyse_features
from poker_coach.domain import parse_cards




# --------------------------------------------- a draw must be hero's own


def test_a_straight_on_the_board_is_not_heros_draw():
    """From a live session: 2d4d on JsTs7c9c was called a gutshot.

    An eight completes J-T-9-8-7 on the board. Every player has it, hero's
    cards contribute nothing, and calling it hero's draw is wrong twice.
    """

    features = analyse_features(parse_cards("2d4d"), parse_cards("JsTs7c9c"))
    assert features.straight_draw is None
    assert "straight draw" not in "; ".join(features.descriptions())


def test_a_turn_straight_draw_using_a_hole_card_is_still_reported():
    """Ts9s on 8h7c2d4s: a six or a jack makes hero's straight.

    Neither card makes one for the board alone (8-7-6-4-2 is not a straight),
    so the draw is genuinely hero's and must survive the board check.
    """

    features = analyse_features(parse_cards("Ts9s"), parse_cards("8h7c2d4s"))
    assert features.straight_draw is not None
    assert features.straight_draw.outs > 0


def test_a_board_that_outruns_heros_straight_is_not_counted():
    """6d5d on JsTs7c9c looks like a draw and is not one.

    The eight it "needs" makes J-T-9-8-7 for everyone, which outranks the
    9-high straight hero would have. Hero gains nothing.
    """

    assert analyse_features(
        parse_cards("6d5d"), parse_cards("JsTs7c9c")
    ).straight_draw is None


def test_flop_draws_are_unaffected_by_the_board_check():
    # Four community cards cannot make a straight alone, so every out counts.
    features = analyse_features(parse_cards("Ts9s"), parse_cards("8h7c2d"))
    assert features.straight_draw is not None
    assert features.straight_draw.kind == "open-ended"


# ------------------------------------------- no draws once the board is out


def test_no_flush_draw_on_a_complete_board():
    """Four to a flush on the river is not a draw — nothing is coming."""

    features = analyse_features(parse_cards("As2d"), parse_cards("Ks7s9s4c3h"))
    assert features.flush_draw is None
    assert "flush draw" not in "; ".join(features.descriptions())


def test_no_straight_draw_on_a_complete_board():
    features = analyse_features(parse_cards("Ts9s"), parse_cards("8h7c2d4dKh"))
    assert features.straight_draw is None


def test_a_made_flush_on_the_river_is_still_reported():
    features = analyse_features(parse_cards("AsKs"), parse_cards("Qs2s9s4c3h"))
    assert features.flush_draw is None          # not a draw
    assert "flush" in features.made_description  # but the made hand stands


@pytest.mark.parametrize(
    "hole,board",
    [
        ("AsKs", "Qs2s9c"),      # nut flush draw, flop
        ("Ts9s", "8h7c2d"),      # open-ended, flop
        ("Ts9s", "8h6c2d"),      # gutshot, flop
    ],
)
def test_genuine_flop_draws_survive(hole, board):
    features = analyse_features(parse_cards(hole), parse_cards(board))
    assert features.has_draw
