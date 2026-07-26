import random

import pytest

from poker_coach.coaching.analysis import analyze
from poker_coach.domain import (
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)
from poker_coach.evaluation.grounding import (
    SMALL_INTEGER_CEILING,
    check_grounding,
    extract_claims,
    grounded_values,
    mask_poker_notation,
)


def spot():
    """Hero facing a pot-sized bet of 6 into 12 on the flop."""

    state = HandState(
        players=[
            PlayerState(
                name="hero",
                position=Position.BTN,
                stack=97.0,
                hole_cards=tuple(parse_cards("AsKs")),
                is_hero=True,
            ),
            PlayerState(
                name="villain",
                position=Position.BB,
                stack=91.0,
                committed_this_street=6.0,
            ),
        ],
        board=parse_cards("Qs2s9c"),
        street=Street.FLOP,
        pot=6.0,
    )
    return analyze(
        state, villain_range="random", iterations=4000, rng=random.Random(11)
    )


# ---------------------------------------------------------------- masking


@pytest.mark.parametrize(
    "text",
    ["Qs2s9c", "AsKs", "7c2d", "Ah5d", "2h3d4c5s6h"],
)
def test_card_tokens_are_masked(text):
    assert not any(ch.isdigit() for ch in mask_poker_notation(text))


@pytest.mark.parametrize(
    "text",
    ["77+", "ATs+", "A5s-A2s", "T9s-76s", "99-66", "QQ", "AKo", "22+"],
)
def test_range_tokens_are_masked(text):
    assert not any(ch.isdigit() for ch in mask_poker_notation(text))


def test_masking_preserves_string_length():
    text = "On Qs2s9c against 77+ you have 56.0% equity"
    assert len(mask_poker_notation(text)) == len(text)


def test_masking_leaves_real_statistics_alone():
    masked = mask_poker_notation("You have 56.0% equity and 4.09 chips of EV")
    assert "56.0" in masked and "4.09" in masked


def test_board_digits_are_not_read_as_claims():
    claims = extract_claims("The board is Qs2s9c and the turn brought 7h.")
    assert claims == []


def test_range_digits_are_not_read_as_claims():
    claims = extract_claims("Villain's range here is 77+, ATs+, and A5s-A2s.")
    assert claims == []


# -------------------------------------------------------------- extraction


def test_percentages_are_extracted():
    claims = extract_claims("You hold 56.0% equity.")
    assert len(claims) == 1
    assert claims[0].kind == "percent"
    assert claims[0].value == pytest.approx(56.0)


def test_percent_with_space_is_extracted():
    assert extract_claims("about 33 % of the time")[0].value == pytest.approx(33.0)


def test_ratios_are_extracted():
    claims = extract_claims("You are getting 2.00:1 on the call.")
    assert claims[0].kind == "ratio"
    assert claims[0].value == pytest.approx(2.0)


def test_signed_numbers_are_extracted():
    claims = extract_claims("EV of calling is +4.09 chips.")
    assert claims[0].value == pytest.approx(4.09)


def test_decimals_are_extracted():
    assert extract_claims("SPR is 16.17 here.")[0].value == pytest.approx(16.17)


def test_bare_small_integers_are_ignored_by_design():
    claims = extract_claims("You have two pair and 9 outs going to the river.")
    assert claims == []


def test_large_integers_are_still_checked():
    claims = extract_claims("The pot is 12 and you must call 6000.")
    assert [c.value for c in claims] == [12.0, 6000.0]


def test_comma_grouped_numbers_read_as_one_value():
    claims = extract_claims("Based on 20,000 samples.")
    assert [c.value for c in claims] == [20000.0]


@pytest.mark.parametrize("text", ["33.3%", "73.2%", "56.0%", "22.5%"])
def test_percentages_whose_digits_look_like_ranks_survive_masking(text):
    # Regression: a naive rank pattern masked the "33" in "33.3%", leaving "3%".
    claims = extract_claims(f"You need {text} here.")
    assert len(claims) == 1
    assert claims[0].text == text


@pytest.mark.parametrize(
    "text",
    ["lower sets (`77`, `22`)", "pocket 22 is behind", "a set of 99 is ahead",
     "a pair of 88 wins"],
)
def test_a_marked_repeated_rank_reads_as_a_pocket_pair(text):
    """A live run flagged "lower sets (`77`, `22`)" as invented numbers."""

    assert extract_claims(text) == []


def test_holds_does_not_hide_a_percentage():
    """"holds" reads equally well before cards and before quantities, so it
    is not a marker. It used to be, and swallowed the figure."""

    claims = extract_claims("Villain holds 77% equity.")
    assert [c.text for c in claims] == ["77%"]


def test_holding_does_not_hide_a_chip_amount():
    claims = extract_claims("Villain is holding 77 chips.")
    assert [c.value for c in claims] == [77.0]


def test_a_percent_sign_vetoes_the_pocket_pair_reading_entirely():
    # Closes the class rather than the instance: no rank is written with a
    # percent sign, so the marker never wins against one.
    assert [c.text for c in extract_claims("pocket 77% of the time")] == ["77%"]
    assert [c.text for c in extract_claims("a set of 99% equity")] == ["99%"]


def test_a_repeated_rank_with_a_percent_sign_is_still_a_statistic():
    # The masking must not swallow "88% equity" — that is the exact shape of
    # an invented figure the checker exists to catch.
    claims = extract_claims("You have 88% equity.")
    assert [c.text for c in claims] == ["88%"]


@pytest.mark.parametrize(
    "text,value",
    [("The pot is 77 now.", 77.0), ("You must call 44 chips.", 44.0),
     ("The pot is 76 now.", 76.0)],
)
def test_an_unmarked_repeated_rank_is_still_an_amount(text, value):
    """Masking every bare 22-99 would blind the checker to eight common
    pot, stack and raise sizes — on a checker that already ignores 0-10."""

    assert [c.value for c in extract_claims(text)] == [value]


def test_known_notation_covers_an_unmarked_range_list():
    # A bare "99, 77, 22" carries no marker, so the generic patterns leave it —
    # but the spot knows its own range and masks it literally.
    text = "Against QQ, 99, 77, 22 you are well ahead."
    assert extract_claims(text) != []
    assert extract_claims(text, known=["QQ, 99, 77, 22"]) == []


def test_known_notation_masking_does_not_eat_real_numbers():
    # Regression: masking the range element "77" unguarded turned the equity
    # figure "77.40%" into a phantom "40%".
    claims = extract_claims("You have 77.40% equity.", known=["77", "QQ, 99, 77, 22"])
    assert [c.text for c in claims] == ["77.40%"]


def test_known_notation_masking_handles_a_trailing_period():
    assert extract_claims("Villain holds 77.", known=["77"]) == []


def test_known_notation_does_not_mask_inside_a_longer_number():
    claims = extract_claims("The pot is 1770 chips.", known=["77"])
    assert [c.value for c in claims] == [1770.0]


def test_small_integer_ceiling_is_the_boundary():
    below = extract_claims(f"count of {SMALL_INTEGER_CEILING}")
    above = extract_claims(f"count of {SMALL_INTEGER_CEILING + 1}")
    assert below == []
    assert len(above) == 1


def test_percent_and_number_are_not_double_counted():
    claims = extract_claims("You have 56.0% equity.")
    assert len(claims) == 1


def test_ratio_is_not_split_into_two_numbers():
    claims = extract_claims("Getting 2.00:1 here.")
    assert len(claims) == 1


def test_empty_text_yields_no_claims():
    assert extract_claims("") == []


# ------------------------------------------------------------- grounding


def test_facts_block_is_self_grounded():
    """The strongest check available: the facts must ground themselves."""

    analysis = spot()
    report = check_grounding(analysis.to_prompt_block(), analysis)
    assert report.is_grounded, report.summary()
    assert report.checked_count > 0


def test_quoting_the_equity_is_grounded():
    analysis = spot()
    text = f"Call. You have {analysis.equity.equity_pct:.1f}% equity."
    assert check_grounding(text, analysis).is_grounded


def test_quoting_the_price_is_grounded():
    analysis = spot()
    text = f"You need {analysis.odds.required_equity_pct:.1f}% to call."
    assert check_grounding(text, analysis).is_grounded


def test_quoting_the_pot_and_call_is_grounded():
    analysis = spot()
    text = f"The pot is {analysis.total_pot:g} and it is {analysis.to_call:g} to call."
    assert check_grounding(text, analysis).is_grounded


def test_quoting_ev_is_grounded():
    analysis = spot()
    assert check_grounding(f"EV is {analysis.ev_of_calling:+.2f} chips.", analysis)


def test_negated_ev_is_still_grounded():
    # "losing 4.09 chips" quotes the same figure with the sign flipped.
    analysis = spot()
    text = f"Folding here costs you {abs(analysis.ev_of_calling):.2f} chips."
    assert check_grounding(text, analysis).is_grounded


def test_rate_quoted_as_a_fraction_is_grounded():
    analysis = spot()
    text = f"Your equity is {analysis.equity.equity:.3f} against that range."
    assert check_grounding(text, analysis).is_grounded


def test_invented_equity_is_caught():
    analysis = spot()
    report = check_grounding("Call. You have 91.4% equity here.", analysis)
    assert not report.is_grounded
    assert "91.4%" in report.summary()


def test_invented_pot_odds_are_caught():
    analysis = spot()
    report = check_grounding("You only need 12.5% to make this call.", analysis)
    assert not report.is_grounded


def test_invented_chip_amount_is_caught():
    analysis = spot()
    report = check_grounding("Calling here nets you 87.30 chips.", analysis)
    assert not report.is_grounded


def test_mixed_response_separates_good_from_bad():
    analysis = spot()
    text = (
        f"Call. You have {analysis.equity.equity_pct:.1f}% equity "
        "and villain folds 73.2% of the time."
    )
    report = check_grounding(text, analysis)
    assert not report.is_grounded
    assert len(report.grounded) == 1
    assert [c.text for c in report.ungrounded] == ["73.2%"]


def test_rounding_within_tolerance_is_accepted():
    analysis = spot()
    rounded = round(analysis.equity.equity_pct)
    assert check_grounding(f"About {rounded}% equity.", analysis).is_grounded


def test_a_number_far_outside_tolerance_is_rejected():
    analysis = spot()
    far = analysis.equity.equity_pct + 12.0
    assert not check_grounding(f"About {far:.1f}% equity.", analysis).is_grounded


def test_response_with_no_numbers_is_trivially_grounded():
    report = check_grounding("Fold. Your hand is behind villain's range.", spot())
    assert report.is_grounded
    assert report.checked_count == 0
    assert "No numeric claims" in report.summary()


def test_report_is_falsy_when_ungrounded():
    assert not bool(check_grounding("You have 91.4% equity.", spot()))
    assert bool(check_grounding("Just fold.", spot()))


def test_summary_counts_are_accurate():
    analysis = spot()
    text = f"{analysis.equity.equity_pct:.1f}% equity, but villain bluffs 40.0%."
    report = check_grounding(text, analysis)
    assert report.checked_count == 2
    assert len(report.ungrounded) == 1


# --------------------------------------------------------- allowed values


def test_grounded_values_includes_the_headline_numbers():
    analysis = spot()
    percents, numbers = grounded_values(analysis)
    assert analysis.equity.equity_pct in percents
    assert analysis.odds.required_equity_pct in percents
    assert analysis.total_pot in numbers
    assert analysis.to_call in numbers


def test_grounded_values_includes_mdf_and_alpha_when_facing_a_bet():
    analysis = spot()
    percents, _ = grounded_values(analysis)
    assert analysis.mdf is not None
    assert 100.0 * analysis.mdf in percents
    assert 100.0 * analysis.alpha in percents


def test_grounded_values_handles_infinite_spr():
    # A preflop spot with no pot yet has an infinite SPR; it must not leak in.
    state = HandState(
        players=[
            PlayerState(
                name="hero",
                position=Position.BTN,
                stack=100.0,
                hole_cards=tuple(parse_cards("AsKs")),
                is_hero=True,
            ),
            PlayerState(name="villain", position=Position.BB, stack=100.0),
        ]
    )
    analysis = analyze(state, iterations=200, rng=random.Random(3))
    _, numbers = grounded_values(analysis)
    assert all(n != float("inf") for n in numbers)
