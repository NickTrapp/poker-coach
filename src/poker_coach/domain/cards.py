"""Card primitives: ranks, suits, cards and decks.

`Card` is a lightweight frozen dataclass rather than a pydantic model so that it
stays cheap to hash and copy inside the Monte Carlo loops in
`poker_coach.calculations.equity`. It still participates in pydantic
validation/serialisation via `__get_pydantic_core_schema__`, so hand histories
round-trip as compact strings like `"As"`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "Suit",
    "Rank",
    "Card",
    "Deck",
    "parse_cards",
    "cards_to_str",
]


class Suit(str, Enum):
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def symbol(self) -> str:
        return _RANK_TO_SYMBOL[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> "Rank":
        try:
            return _SYMBOL_TO_RANK[symbol.upper()]
        except KeyError:
            raise ValueError(f"unknown rank symbol: {symbol!r}") from None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.symbol


_RANK_TO_SYMBOL: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}

_SYMBOL_TO_RANK: dict[str, Rank] = {v: k for k, v in _RANK_TO_SYMBOL.items()}


@dataclass(frozen=True, slots=True, order=False)
class Card:
    """A single playing card, e.g. ``Card(Rank.ACE, Suit.SPADES)`` -> ``"As"``."""

    rank: Rank
    suit: Suit

    @classmethod
    def from_str(cls, text: str) -> "Card":
        if isinstance(text, Card):  # tolerate re-validation
            return text
        if not isinstance(text, str):
            raise TypeError(f"card must be a string, got {type(text).__name__}")
        cleaned = text.strip()
        if len(cleaned) != 2:
            raise ValueError(f"card must be 2 characters like 'As', got {text!r}")
        rank = Rank.from_symbol(cleaned[0])
        try:
            suit = Suit(cleaned[1].lower())
        except ValueError:
            raise ValueError(f"unknown suit in card: {text!r}") from None
        return cls(rank, suit)

    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.value}"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Card('{self}')"

    def __lt__(self, other: "Card") -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return (self.rank, self.suit.value) < (other.rank, other.suit.value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        from_str = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.from_str),
            ]
        )
        return core_schema.json_or_python_schema(
            json_schema=from_str,
            python_schema=core_schema.union_schema(
                [core_schema.is_instance_schema(cls), from_str]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


FULL_DECK: tuple[Card, ...] = tuple(
    Card(rank, suit) for suit in Suit for rank in Rank
)


def parse_cards(text: str | Iterable[str]) -> list[Card]:
    """Parse cards from ``"AsKd"``, ``"As Kd"``, ``"As,Kd"`` or an iterable."""

    if not isinstance(text, str):
        return [Card.from_str(str(item)) for item in text]

    compact = text.replace(",", " ").split()
    tokens: list[str] = []
    for chunk in compact:
        if len(chunk) % 2 != 0:
            raise ValueError(f"cannot split {chunk!r} into 2-character cards")
        tokens.extend(chunk[i : i + 2] for i in range(0, len(chunk), 2))
    return [Card.from_str(token) for token in tokens]


def cards_to_str(cards: Iterable[Card]) -> str:
    """Render cards back to the compact ``"AsKd"`` form."""

    return "".join(str(card) for card in cards)


class Deck:
    """A mutable 52-card deck with optional removed (dead) cards."""

    def __init__(
        self,
        *,
        removed: Iterable[Card] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._rng = rng or random.Random()
        dead = set(removed or ())
        unknown = dead - set(FULL_DECK)
        if unknown:
            raise ValueError(f"removed cards are not real cards: {sorted(unknown)}")
        self._cards: list[Card] = [c for c in FULL_DECK if c not in dead]

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards)

    def __contains__(self, card: object) -> bool:
        return card in self._cards

    @property
    def cards(self) -> tuple[Card, ...]:
        return tuple(self._cards)

    def shuffle(self) -> "Deck":
        self._rng.shuffle(self._cards)
        return self

    def remove(self, cards: Iterable[Card]) -> "Deck":
        for card in cards:
            try:
                self._cards.remove(card)
            except ValueError:
                raise ValueError(f"{card} is not in the deck") from None
        return self

    def deal(self, count: int = 1) -> list[Card]:
        if count > len(self._cards):
            raise ValueError(
                f"cannot deal {count} cards, only {len(self._cards)} remain"
            )
        dealt = self._cards[:count]
        del self._cards[:count]
        return dealt

    def sample(self, count: int) -> list[Card]:
        """Draw ``count`` cards at random *without* mutating the deck."""

        if count > len(self._cards):
            raise ValueError(
                f"cannot sample {count} cards, only {len(self._cards)} remain"
            )
        return self._rng.sample(self._cards, count)


def remaining_deck(dead: Sequence[Card]) -> list[Card]:
    """Fast helper for hot loops: the deck minus ``dead``, as a plain list."""

    blocked = set(dead)
    return [card for card in FULL_DECK if card not in blocked]
