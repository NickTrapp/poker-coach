import pytest

from poker_coach.calculations.pot_odds import (
    implied_odds_breakeven,
    bet_as_pot_fraction,
    bluff_success_threshold,
    break_even_fold_equity,
    ev_bluff,
    ev_call,
    minimum_defence_frequency,
    outs_to_equity,
    pot_odds,
    required_equity,
    stack_to_pot_ratio,
)


def test_half_pot_bet_lays_three_to_one():
    # Pot was 100, villain bets 50 -> calling 50 into 150.
    odds = pot_odds(150, 50)
    assert odds.ratio == pytest.approx(3.0)
    assert odds.required_equity == pytest.approx(0.25)
    assert odds.required_equity_pct == pytest.approx(25.0)


def test_pot_sized_bet_needs_a_third():
    assert required_equity(200, 100) == pytest.approx(1 / 3)


def test_free_call_needs_no_equity():
    odds = pot_odds(100, 0)
    assert odds.required_equity == 0.0
    assert odds.ratio == float("inf")


def test_required_equity_of_an_empty_pot_is_zero():
    assert required_equity(0, 0) == 0.0


def test_negative_amounts_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        pot_odds(-1, 10)
    with pytest.raises(ValueError, match="non-negative"):
        pot_odds(10, -1)


def test_pot_odds_str_is_readable():
    assert "need 25.0%" in str(pot_odds(150, 50))


def test_ev_call_is_zero_at_the_break_even_point():
    assert ev_call(150, 50, 0.25) == pytest.approx(0.0)


def test_ev_call_is_positive_above_the_threshold():
    assert ev_call(150, 50, 0.40) > 0


def test_ev_call_is_negative_below_the_threshold():
    assert ev_call(150, 50, 0.10) < 0


def test_ev_call_with_certain_win_is_the_whole_pot():
    assert ev_call(150, 50, 1.0) == pytest.approx(150.0)


def test_ev_call_with_certain_loss_costs_the_call():
    assert ev_call(150, 50, 0.0) == pytest.approx(-50.0)


def test_ev_call_rejects_out_of_range_equity():
    with pytest.raises(ValueError, match="equity must be a fraction"):
        ev_call(100, 50, 1.5)


def test_mdf_for_a_pot_sized_bet_is_one_half():
    assert minimum_defence_frequency(100, 100) == pytest.approx(0.5)


def test_mdf_for_a_half_pot_bet_is_two_thirds():
    assert minimum_defence_frequency(100, 50) == pytest.approx(2 / 3)


def test_mdf_of_a_zero_bet_is_total_defence():
    assert minimum_defence_frequency(100, 0) == 1.0


def test_alpha_is_the_complement_of_mdf():
    for bet in (25, 50, 100, 200):
        assert bluff_success_threshold(100, bet) == pytest.approx(
            1 - minimum_defence_frequency(100, bet)
        )


def test_break_even_fold_equity_is_an_alias_for_alpha():
    assert break_even_fold_equity is bluff_success_threshold
    assert break_even_fold_equity(100, 100) == pytest.approx(0.5)


def test_pure_bluff_breaks_even_at_alpha():
    pot, bet = 100.0, 100.0
    alpha = bluff_success_threshold(pot, bet)
    assert ev_bluff(pot, bet, fold_equity=alpha) == pytest.approx(0.0)


def test_semi_bluff_beats_a_pure_bluff_at_the_same_fold_equity():
    pure = ev_bluff(100, 75, fold_equity=0.4)
    semi = ev_bluff(100, 75, fold_equity=0.4, equity_when_called=0.3)
    assert semi > pure


def test_bluff_that_always_works_wins_the_pot():
    assert ev_bluff(100, 75, fold_equity=1.0) == pytest.approx(100.0)


def test_bluff_that_never_works_loses_the_bet():
    assert ev_bluff(100, 75, fold_equity=0.0) == pytest.approx(-75.0)


def test_ev_bluff_validates_its_probabilities():
    with pytest.raises(ValueError, match="fold_equity"):
        ev_bluff(100, 50, fold_equity=1.2)
    with pytest.raises(ValueError, match="equity_when_called"):
        ev_bluff(100, 50, fold_equity=0.5, equity_when_called=-0.1)


def test_bet_as_pot_fraction():
    assert bet_as_pot_fraction(100, 75) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="pot must be positive"):
        bet_as_pot_fraction(0, 10)


def test_spr():
    assert stack_to_pot_ratio(100, 20) == pytest.approx(5.0)
    assert stack_to_pot_ratio(100, 0) == float("inf")


def test_flush_draw_one_card_equity():
    # 9 outs with 47 unseen cards.
    assert outs_to_equity(9, streets=1) == pytest.approx(9 / 47)


def test_flush_draw_two_cards_beats_the_rule_of_four():
    two_streets = outs_to_equity(9, streets=2)
    assert two_streets == pytest.approx(0.3497, abs=1e-3)
    assert two_streets < 2 * outs_to_equity(9, streets=1)


def test_gutshot_equity():
    assert outs_to_equity(4, streets=1) == pytest.approx(4 / 47)


def test_zero_outs_never_hits():
    assert outs_to_equity(0, streets=2) == 0.0


def test_every_card_is_an_out():
    assert outs_to_equity(47, streets=1) == pytest.approx(1.0)


def test_zero_streets_never_hits():
    assert outs_to_equity(9, streets=0) == 0.0


def test_turn_unseen_count_can_be_overridden():
    assert outs_to_equity(9, streets=1, unseen=46) == pytest.approx(9 / 46)


def test_outs_validation():
    with pytest.raises(ValueError, match="outs must be non-negative"):
        outs_to_equity(-1)
    with pytest.raises(ValueError, match="streets must be non-negative"):
        outs_to_equity(9, streets=-1)
    with pytest.raises(ValueError, match="cannot have"):
        outs_to_equity(50, unseen=47)


# ------------------------------------------------------------ implied odds


def test_implied_odds_are_zero_when_the_call_already_breaks_even():
    # 50% equity calling 50 into 150 is already profitable.
    assert implied_odds_breakeven(150, 50, 0.50) == 0.0


def test_implied_odds_break_even_matches_a_hand_computed_case():
    """From the first live benchmark: 18.15% equity, calling 30 into 40.

    The convention matters. Hero's own call is not winnings, so the answer is
    95.3, not the 65.3 you get by counting it — an error that would make a
    hopeless draw look playable.
    """

    needed = implied_odds_breakeven(40, 30, 0.1815)
    assert needed == pytest.approx(95.3, abs=0.1)


def test_the_call_is_not_counted_as_winnings():
    """Guards the convention: winning takes the pot, not the pot plus my call."""

    pot, call, equity = 40.0, 30.0, 0.1815
    needed = implied_odds_breakeven(pot, call, equity)
    wrong = ((1 - equity) * call) / equity - (pot + call)
    assert needed == pytest.approx(wrong + call, abs=1e-6)
    assert needed > wrong


def test_more_equity_needs_fewer_extra_chips():
    generous = implied_odds_breakeven(40, 30, 0.30)
    thin = implied_odds_breakeven(40, 30, 0.10)
    assert thin > generous > 0


def test_a_dead_hand_can_never_be_paid_off_enough():
    assert implied_odds_breakeven(40, 30, 0.0) == float("inf")


def test_the_break_even_figure_actually_breaks_even():
    pot, call, equity = 40.0, 30.0, 0.20
    extra = implied_odds_breakeven(pot, call, equity)
    # Winning the pot plus the extra when hitting, losing the call otherwise.
    ev = equity * (pot + extra) - (1 - equity) * call
    assert ev == pytest.approx(0.0, abs=1e-9)


def test_implied_odds_validates_equity():
    with pytest.raises(ValueError, match="equity must be a fraction"):
        implied_odds_breakeven(40, 30, 1.5)
