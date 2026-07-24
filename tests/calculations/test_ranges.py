import pytest

from poker_coach.calculations.ranges import (
    Range,
    canonical_class,
    class_combos,
    expand_classes,
)
from poker_coach.domain.cards import Rank, parse_cards


def test_canonical_class_orders_high_card_first():
    assert canonical_class(Rank.KING, Rank.ACE, True) == "AKs"
    assert canonical_class(Rank.SEVEN, Rank.SEVEN, None) == "77"


def test_pair_expands_to_six_combos():
    assert len(class_combos("77")) == 6


def test_suited_expands_to_four_combos():
    combos = class_combos("AKs")
    assert len(combos) == 4
    assert all(a.suit is b.suit for a, b in combos)


def test_offsuit_expands_to_twelve_combos():
    combos = class_combos("AKo")
    assert len(combos) == 12
    assert all(a.suit is not b.suit for a, b in combos)


def test_bare_class_covers_both_suited_and_offsuit():
    assert expand_classes("AK") == ["AKs", "AKo"]
    assert len(Range("AK").combos()) == 16


def test_pair_plus_expansion():
    assert expand_classes("QQ+") == ["AA", "KK", "QQ"]


def test_suited_plus_keeps_the_high_card_fixed():
    assert expand_classes("ATs+") == ["AKs", "AQs", "AJs", "ATs"]


def test_offsuit_plus_expansion():
    assert expand_classes("KTo+") == ["KQo", "KJo", "KTo"]


def test_pair_dash_range_is_inclusive():
    assert expand_classes("99-66") == ["99", "88", "77", "66"]
    assert expand_classes("66-99") == ["99", "88", "77", "66"]


def test_shared_high_card_dash_range():
    assert expand_classes("A5s-A2s") == ["A5s", "A4s", "A3s", "A2s"]


def test_gap_preserving_connector_range():
    assert expand_classes("T9s-76s") == ["T9s", "98s", "87s", "76s"]


def test_mismatched_suitedness_in_dash_range_is_rejected():
    with pytest.raises(ValueError, match="mismatched suitedness"):
        expand_classes("A5s-A2o")


def test_equal_gap_endpoints_expand_even_across_the_whole_ladder():
    # AKs and 32s are both one-gappers, so this is a valid connector range.
    assert expand_classes("AKs-32s")[0] == "AKs"


def test_unsupported_dash_range_is_rejected():
    # AQs is a two-gapper, 32s is a one-gapper: no consistent walk exists.
    with pytest.raises(ValueError, match="unsupported range"):
        expand_classes("AQs-32s")


def test_mixing_pairs_and_non_pairs_is_rejected():
    with pytest.raises(ValueError, match="cannot mix"):
        expand_classes("99-AKs")


def test_pairs_cannot_be_marked_suited():
    with pytest.raises(ValueError, match="cannot be suited"):
        expand_classes("77s")


def test_tokens_split_on_commas_and_whitespace():
    assert set(expand_classes("AA, KK QQ")) == {"AA", "KK", "QQ"}


def test_subtraction_removes_classes():
    assert expand_classes("22+, -55") == [
        c for c in expand_classes("22+") if c != "55"
    ]


def test_random_covers_every_starting_hand():
    assert len(Range("random").combos()) == 1326
    assert Range("any").percent_of_hands == pytest.approx(100.0)


def test_class_count_is_169():
    assert len(expand_classes("random")) == 169


def test_explicit_combo_notation():
    combos = Range("AsKd").combos()
    assert combos == [tuple(sorted(parse_cards("AsKd"), reverse=True))]


def test_explicit_combo_rejects_repeated_card():
    with pytest.raises(ValueError, match="same card twice"):
        Range("AsAs").combos()


def test_expand_classes_rejects_explicit_combos():
    with pytest.raises(ValueError, match="explicit combos"):
        expand_classes("AsKd")


def test_dead_cards_block_combos():
    full = Range("AA").combos()
    assert len(full) == 6
    blocked = Range("AA").combos(dead=parse_cards("As"))
    assert len(blocked) == 3
    assert all(parse_cards("As")[0] not in combo for combo in blocked)


def test_combos_are_deduplicated_across_overlapping_tokens():
    assert len(Range("77+, 88+").combos()) == len(Range("77+").combos())


def test_contains_checks_membership_by_class():
    rng = Range("77+, AQs+")
    assert parse_cards("AhAd") in rng
    assert parse_cards("AsKs") in rng
    assert parse_cards("AsKd") not in rng
    assert parse_cards("2s3d") not in rng


def test_contains_rejects_wrong_length():
    assert parse_cards("AsKdQh") not in Range("random")


def test_len_matches_combo_count():
    rng = Range("77+, AQs+")
    assert len(rng) == len(rng.combos()) == 8 * 6 + 2 * 4


def test_percent_of_hands():
    assert Range("AA").percent_of_hands == pytest.approx(100 * 6 / 1326)


def test_case_insensitive_notation():
    assert expand_classes("aks") == ["AKs"]


def test_unknown_token_raises():
    with pytest.raises(ValueError):
        expand_classes("XYZ")
