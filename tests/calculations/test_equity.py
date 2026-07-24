import random

import pytest

from poker_coach.calculations.equity import (
    _normalize_opponent,
    equity,
    equity_vs_random,
)
from poker_coach.calculations.ranges import Range
from poker_coach.domain.cards import parse_cards


def seeded():
    return random.Random(1234)


def test_river_showdown_is_exact_and_decisive():
    # Hero has the nut flush; villain has top two pair. No cards to come.
    result = equity("AsKs", "AhKh", board="QsJs2s7d3c")
    assert result.exact is True
    assert result.equity == pytest.approx(1.0)
    assert result.samples == 1
    assert result.margin_of_error == 0.0


def test_river_tie_splits_the_pot():
    # Both play the board's straight; hole cards are irrelevant.
    result = equity("2c3d", "2h3s", board="AhKdQcJsTh")
    assert result.exact is True
    assert result.equity == pytest.approx(0.5)
    assert result.tie == pytest.approx(1.0)


def test_turn_spot_enumerates_all_44_rivers():
    result = equity("AsKs", "7h7d", board="Qs2s9c4d")
    assert result.exact is True
    assert result.samples == 44


def test_aces_versus_kings_preflop_is_about_82_percent():
    result = equity("AsAh", "KsKh", iterations=8000, rng=seeded())
    assert result.equity == pytest.approx(0.82, abs=0.02)


def test_coinflip_pair_versus_two_overcards():
    result = equity("7h7d", "AsKs", iterations=8000, rng=seeded())
    assert result.equity == pytest.approx(0.52, abs=0.03)


def test_flop_enumeration_is_exact_against_a_known_hand():
    # Both hands and the flop are known, leaving 45 unseen cards, so there are
    # C(45, 2) = 990 turn/river runouts — well under the exact limit.
    result = equity("AsKs", "7h7d", board="Qs2s9c")
    assert result.exact is True
    assert result.samples == 990


def test_dominated_hand_is_a_big_underdog():
    result = equity("AhQd", "AsKs", iterations=8000, rng=seeded())
    assert result.equity < 0.35


def test_equity_against_a_range_uses_sampling():
    result = equity("AsAh", Range("77+, AQs+"), iterations=4000, rng=seeded())
    assert result.exact is False
    assert 0.7 < result.equity < 0.95
    assert result.margin_of_error > 0.0


def test_range_notation_accepted_as_a_plain_string():
    result = equity("AsAh", "77+", iterations=2000, rng=seeded())
    assert result.exact is False
    assert result.equity > 0.7


def test_combo_string_is_preferred_over_range_parsing():
    # "KsKh" is a specific combo, not the class "KK" — so the opponent is
    # pinned and the spot enumerates exactly once the board is known.
    assert _normalize_opponent("KsKh") == tuple(parse_cards("KsKh"))
    assert _normalize_opponent("KK") == Range("KK")

    result = equity("AsAh", "KsKh", board="2c7d9h4s")
    assert result.exact is True


def test_multiway_equity_is_lower_than_heads_up():
    heads_up = equity("AsAh", ["KsKh"], iterations=4000, rng=seeded())
    three_way = equity("AsAh", ["KsKh", "QcQd"], iterations=4000, rng=seeded())
    assert three_way.equity < heads_up.equity


def test_win_tie_lose_sum_to_one():
    result = equity("AsKd", "7h7c", iterations=3000, rng=seeded())
    assert result.win + result.tie + result.lose == pytest.approx(1.0)


def test_results_are_reproducible_under_a_seed():
    a = equity("AsKd", Range("random"), iterations=1500, rng=random.Random(99))
    b = equity("AsKd", Range("random"), iterations=1500, rng=random.Random(99))
    assert a.equity == b.equity


def test_equity_vs_random_beats_half_for_a_premium_hand():
    result = equity_vs_random("AsAh", iterations=4000, rng=seeded())
    assert result.equity > 0.8


def test_equity_vs_multiple_random_opponents_drops():
    one = equity_vs_random("AsAh", opponents=1, iterations=3000, rng=seeded())
    three = equity_vs_random("AsAh", opponents=3, iterations=3000, rng=seeded())
    assert three.equity < one.equity


def test_margin_of_error_shrinks_with_more_samples():
    small = equity("AsKd", "7h7c", iterations=500, rng=seeded())
    large = equity("AsKd", "7h7c", iterations=8000, rng=seeded())
    assert large.margin_of_error < small.margin_of_error


def test_duplicate_cards_are_rejected():
    with pytest.raises(ValueError, match="duplicate cards"):
        equity("AsKs", "AsQd")
    with pytest.raises(ValueError, match="duplicate cards"):
        equity("AsKs", "QhJh", board="AsKd2c")


def test_blocked_range_raises_a_clear_error():
    # Hero holds two of the four aces, villain's range is AA only.
    with pytest.raises(ValueError, match="no combos after blockers"):
        equity("AsAh", Range("AA"), board="AdAc2s", iterations=100, rng=seeded())


def test_invalid_combo_length_is_rejected():
    with pytest.raises(ValueError, match="exactly 2 cards"):
        equity("AsKsQs", "7h7d")


def test_oversized_board_is_rejected():
    with pytest.raises(ValueError, match="cannot exceed 5"):
        equity("AsKs", "7h7d", board="2c3c4c5c6c7c")


def test_zero_iterations_is_rejected():
    with pytest.raises(ValueError, match="iterations must be positive"):
        equity("AsAh", Range("random"), iterations=0, rng=seeded())


def test_str_reports_exactness():
    exact = equity("AsKs", "AhKh", board="QsJs2s7d3c")
    assert "exact" in str(exact)
    sampled = equity("AsAh", Range("random"), iterations=1000, rng=seeded())
    assert "samples" in str(sampled)


@pytest.mark.parametrize(
    "hero,villain,board",
    [
        ("AsKs", "7h7d", "Qs2s9c4d"),
        ("AhQd", "AsKs", "Jc7h2d5s"),
        ("Ts9s", "AhKd", "8s7h2c3d"),
        ("2s2h", "AsKd", "Qc7h9d4s"),
    ],
)
def test_sampler_agrees_with_exact_enumeration(hero, villain, board):
    """The Monte Carlo path must be unbiased against the enumerated truth."""

    exact = equity(hero, villain, board)
    sampled = equity(
        hero, villain, board, iterations=40_000, rng=random.Random(5), exact_limit=0
    )
    assert exact.exact is True
    assert sampled.exact is False
    # Agreement to within the sampler's own stated confidence interval.
    assert abs(sampled.equity - exact.equity) < 2 * sampled.margin_of_error


def test_single_combo_range_collapses_to_an_exact_calculation():
    # "villain turned over 7d7c" is a known hand, even expressed as a range.
    result = equity("AsKs", Range("7d7c"), board="Qs2s9c4d")
    assert result.exact is True
    assert result.samples == 44
    assert result.equity == equity("AsKs", "7d7c", board="Qs2s9c4d").equity


def test_range_reduced_to_one_combo_by_blockers_also_collapses():
    # Hero holds one ace and the board shows another, so the only "AA" combo
    # still available is AdAc — enough to enumerate exactly.
    result = equity("AsKs", Range("AA"), board="Ah2c7d9h")
    assert result.exact is True
    assert result.samples == 44
    assert result.equity == equity("AsKs", "AdAc", board="Ah2c7d9h").equity


def test_multi_combo_range_still_samples():
    result = equity("AsKs", Range("77"), board="Qs2s9c4d", iterations=2000, rng=seeded())
    assert result.exact is False


def test_equity_pct_matches_equity():
    result = equity("AsKs", "AhKh", board="QsJs2s7d3c")
    assert result.equity_pct == pytest.approx(100 * result.equity)
