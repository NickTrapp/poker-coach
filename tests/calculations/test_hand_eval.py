import pytest

from poker_coach.calculations.hand_eval import (
    HandCategory,
    best_five,
    compare,
    evaluate,
)
from poker_coach.domain.cards import cards_to_str, parse_cards


def ev(text: str):
    return evaluate(parse_cards(text))


@pytest.mark.parametrize(
    "cards,category",
    [
        ("AsKsQsJsTs", HandCategory.STRAIGHT_FLUSH),
        ("9s8s7s6s5s", HandCategory.STRAIGHT_FLUSH),
        ("5s4s3s2sAs", HandCategory.STRAIGHT_FLUSH),  # steel wheel
        ("AsAhAdAc2s", HandCategory.FOUR_OF_A_KIND),
        ("AsAhAdKcKs", HandCategory.FULL_HOUSE),
        ("As9s7s5s3s", HandCategory.FLUSH),
        ("AsKhQdJcTs", HandCategory.STRAIGHT),
        ("5h4d3c2sAh", HandCategory.STRAIGHT),  # wheel
        ("AsAhAd5c2s", HandCategory.THREE_OF_A_KIND),
        ("AsAhKdKc2s", HandCategory.TWO_PAIR),
        ("AsAh9d5c2s", HandCategory.PAIR),
        ("AsQh9d5c2s", HandCategory.HIGH_CARD),
    ],
)
def test_five_card_categories(cards, category):
    assert ev(cards).category is category


def test_seven_cards_pick_the_best_five():
    # The board pairs twice, but five spades outrank the trips.
    hand = ev("AsKs" + "QsJs2s" + "2h2d")
    assert hand.category is HandCategory.FLUSH
    assert hand.ranks[:5] == (14, 13, 12, 11, 2)


def test_seven_cards_find_a_straight_flush_over_a_plain_flush():
    hand = ev("AsKs" + "QsJsTs" + "2h2d")
    assert hand.category is HandCategory.STRAIGHT_FLUSH


def test_wheel_straight_is_five_high_not_ace_high():
    hand = ev("5h4d3c2sAh")
    assert hand.category is HandCategory.STRAIGHT
    assert hand.ranks[0] == 5


def test_wheel_loses_to_six_high_straight():
    assert compare(parse_cards("6h5d4c3s2h"), parse_cards("5h4d3c2sAh")) == 1


def test_straight_flush_beats_quads():
    assert compare(parse_cards("9s8s7s6s5s"), parse_cards("AsAhAdAcKs")) == 1


def test_flush_beats_straight():
    assert compare(parse_cards("As9s7s5s3s"), parse_cards("AhKdQcJsTh")) == 1


def test_full_house_beats_flush():
    assert compare(parse_cards("AsAhAdKcKs"), parse_cards("Ks9s7s5s3s")) == 1


def test_higher_quads_win():
    assert compare(parse_cards("AsAhAdAc2s"), parse_cards("KsKhKdKc2s")) == 1


def test_full_house_compares_trips_before_pair():
    # 999-22 beats 888-AA
    assert compare(parse_cards("9s9h9dAcAs"), parse_cards("8s8h8dAcAh")) == 1


def test_two_pair_kicker_decides():
    better = parse_cards("AsAhKdKcQs")
    worse = parse_cards("AdAcKhKsJd")
    assert compare(better, worse) == 1


def test_identical_hand_values_tie_across_different_suits():
    assert compare(parse_cards("AsKhQdJcTs"), parse_cards("AhKsQcJdTh")) == 0


def test_seven_card_kickers_are_limited_to_five_cards():
    # The board makes a straight both players play; hole cards cannot improve
    # on it, so the extra two cards are ignored entirely.
    board = "AhKhQdJcTs"
    assert compare(parse_cards("3d4d" + board), parse_cards("5d6d" + board)) == 0


def test_low_hole_cards_still_count_as_kickers_on_a_high_card_board():
    board = "AhKhQd7c2s"
    assert compare(parse_cards("5d6d" + board), parse_cards("3d4d" + board)) == 1


def test_pair_uses_three_kickers():
    board = "AhKhQd7c2s"
    assert compare(parse_cards("AsJd" + board), parse_cards("AcTd" + board)) == 1


def test_best_five_returns_the_actual_cards():
    cards = parse_cards("AsKs" + "QsJsTs" + "2h3d")
    chosen = best_five(cards)
    assert len(chosen) == 5
    assert evaluate(chosen).category is HandCategory.STRAIGHT_FLUSH
    assert set(chosen) <= set(cards)


def test_best_five_of_exactly_five_is_identity():
    cards = parse_cards("AsKhQdJcTs")
    assert best_five(cards) == tuple(cards)


def test_score_ordering_matches_category_ordering():
    ladder = [
        "AsQh9d5c2s",  # high card
        "AsAh9d5c2s",  # pair
        "AsAhKdKc2s",  # two pair
        "AsAhAd5c2s",  # trips
        "AsKhQdJcTs",  # straight
        "As9s7s5s3s",  # flush
        "AsAhAdKcKs",  # full house
        "AsAhAdAc2s",  # quads
        "9s8s7s6s5s",  # straight flush
    ]
    scores = [ev(h).score for h in ladder]
    assert scores == sorted(scores)


def test_descriptions_read_naturally():
    assert ev("AsAhKdKc2s").description == "two pair, aces and kings"
    assert ev("AsKsQsJsTs").description == "royal flush"
    assert ev("9s9h9dAcAs").description == "full house, nines full of aces"
    assert ev("AsAh9d5c2s").description == "a pair of aces"
    assert ev("AsQh9d5c2s").description == "ace-high"


@pytest.mark.parametrize("count", [0, 4, 8])
def test_rejects_wrong_card_counts(count):
    cards = parse_cards("AsKsQsJsTs9s8s6s")[:count]
    with pytest.raises(ValueError):
        evaluate(cards)


def test_rejects_duplicate_cards():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(parse_cards("AsAsKdQcJh"))


def test_six_card_hands_are_supported():
    hand = ev("AsKsQsJsTs2h")
    assert hand.category is HandCategory.STRAIGHT_FLUSH


def test_flush_over_flush_uses_high_cards():
    board = "2s5s9s"
    assert compare(parse_cards("AsKs" + board + "7h8d"),
                   parse_cards("QsJs" + board + "7h8d")) == 1


def test_cards_to_str_helper_used_in_failure_messages():
    # Guards the display helper the coaching layer relies on.
    assert cards_to_str(best_five(parse_cards("AsKhQdJcTs2h3d"))) == "AsKhQdJcTs"
