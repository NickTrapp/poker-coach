import pickle
import random

import pytest
from pydantic import BaseModel

from poker_coach.domain.cards import (
    FULL_DECK,
    Card,
    Deck,
    Rank,
    Suit,
    cards_to_str,
    parse_cards,
    remaining_deck,
)


def test_full_deck_has_52_distinct_cards():
    assert len(FULL_DECK) == 52
    assert len(set(FULL_DECK)) == 52


@pytest.mark.parametrize(
    "text,rank,suit",
    [
        ("As", Rank.ACE, Suit.SPADES),
        ("Td", Rank.TEN, Suit.DIAMONDS),
        ("2c", Rank.TWO, Suit.CLUBS),
        ("kh", Rank.KING, Suit.HEARTS),
    ],
)
def test_from_str_parses_rank_and_suit(text, rank, suit):
    card = Card.from_str(text)
    assert card.rank is rank
    assert card.suit is suit


def test_str_round_trips_and_normalises_case():
    assert str(Card.from_str("kh")) == "Kh"
    assert Card.from_str(str(Card(Rank.NINE, Suit.CLUBS))) == Card(Rank.NINE, Suit.CLUBS)


@pytest.mark.parametrize("bad", ["", "A", "Ass", "Xs", "Az", "10s"])
def test_from_str_rejects_malformed_input(bad):
    with pytest.raises((ValueError, TypeError)):
        Card.from_str(bad)


def test_cards_are_hashable_and_frozen():
    card = Card.from_str("As")
    assert card in {Card.from_str("As")}
    with pytest.raises(Exception):
        card.rank = Rank.KING


def test_cards_survive_pickling():
    card = Card.from_str("Qd")
    assert pickle.loads(pickle.dumps(card)) == card


def test_ordering_is_by_rank_then_suit():
    assert Card.from_str("2c") < Card.from_str("As")
    assert Card.from_str("Ac") < Card.from_str("As")


@pytest.mark.parametrize(
    "text",
    ["AsKd", "As Kd", "As,Kd", "As, Kd"],
)
def test_parse_cards_accepts_common_separators(text):
    assert parse_cards(text) == [Card.from_str("As"), Card.from_str("Kd")]


def test_parse_cards_rejects_odd_length_chunk():
    with pytest.raises(ValueError):
        parse_cards("AsK")


def test_cards_to_str_is_inverse_of_parse():
    assert cards_to_str(parse_cards("AsKdQh")) == "AsKdQh"


class _Model(BaseModel):
    hole: tuple[Card, Card]


def test_pydantic_validates_from_strings_and_serialises_back():
    model = _Model.model_validate({"hole": ["As", "Kd"]})
    assert model.hole[0] == Card.from_str("As")
    assert model.model_dump(mode="json") == {"hole": ["As", "Kd"]}


def test_pydantic_accepts_card_instances():
    model = _Model(hole=(Card.from_str("As"), Card.from_str("Kd")))
    assert model.hole[1] == Card.from_str("Kd")


def test_pydantic_rejects_bad_card_string():
    with pytest.raises(Exception):
        _Model.model_validate({"hole": ["As", "Zz"]})


def test_deck_starts_full_and_deals_from_the_top():
    deck = Deck()
    assert len(deck) == 52
    dealt = deck.deal(2)
    assert len(dealt) == 2
    assert len(deck) == 50
    assert dealt[0] not in deck


def test_deck_honours_removed_cards():
    dead = parse_cards("AsKd")
    deck = Deck(removed=dead)
    assert len(deck) == 50
    assert all(card not in deck for card in dead)


def test_deck_rejects_overdraw():
    deck = Deck()
    with pytest.raises(ValueError):
        deck.deal(53)


def test_deck_remove_rejects_missing_card():
    deck = Deck(removed=parse_cards("As"))
    with pytest.raises(ValueError):
        deck.remove(parse_cards("As"))


def test_sample_does_not_mutate_the_deck():
    deck = Deck(rng=random.Random(0))
    before = len(deck)
    sampled = deck.sample(5)
    assert len(sampled) == 5
    assert len(deck) == before


def test_shuffle_is_deterministic_under_a_seed():
    a = Deck(rng=random.Random(7)).shuffle().cards
    b = Deck(rng=random.Random(7)).shuffle().cards
    assert a == b


def test_remaining_deck_excludes_dead_cards():
    dead = parse_cards("AsKdQh")
    rest = remaining_deck(dead)
    assert len(rest) == 49
    assert not set(rest) & set(dead)
