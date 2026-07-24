"""Five-to-seven card hand evaluation.

The evaluator packs a hand into a single comparable integer:

    score = category * 15**5 + k0 * 15**4 + k1 * 15**3 + k2 * 15**2 + k3 * 15 + k4

where ``k0..k4`` are the five significant ranks in descending importance
(kickers included). Higher is better, and two hands tie exactly when their
scores are equal — which is what pot-splitting in
:mod:`poker_coach.calculations.equity` relies on.

This is a straightforward counting evaluator rather than a lookup table. It
handles ~1M seven-card hands/second, which is enough for the Monte Carlo sample
sizes the coach uses, and it stays readable enough to audit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Iterable, Sequence

from ..domain.cards import Card, Rank

__all__ = ["HandCategory", "HandRank", "evaluate", "best_five", "compare"]

_BASE = 15


class HandCategory(IntEnum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9

    @property
    def label(self) -> str:
        return _CATEGORY_LABELS[self]


_CATEGORY_LABELS: dict[HandCategory, str] = {
    HandCategory.HIGH_CARD: "high card",
    HandCategory.PAIR: "one pair",
    HandCategory.TWO_PAIR: "two pair",
    HandCategory.THREE_OF_A_KIND: "three of a kind",
    HandCategory.STRAIGHT: "straight",
    HandCategory.FLUSH: "flush",
    HandCategory.FULL_HOUSE: "full house",
    HandCategory.FOUR_OF_A_KIND: "four of a kind",
    HandCategory.STRAIGHT_FLUSH: "straight flush",
}


@dataclass(frozen=True, slots=True, order=False)
class HandRank:
    """The strength of a made hand, comparable via ``<`` / ``==``."""

    category: HandCategory
    ranks: tuple[int, ...]
    score: int

    def __lt__(self, other: "HandRank") -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.score < other.score

    def __le__(self, other: "HandRank") -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.score <= other.score

    def __gt__(self, other: "HandRank") -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.score > other.score

    def __ge__(self, other: "HandRank") -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.score >= other.score

    @property
    def description(self) -> str:
        """Human-readable summary, e.g. ``"two pair, aces and eights"``."""

        names = [_RANK_NAMES[r] for r in self.ranks]
        cat = self.category

        if cat is HandCategory.STRAIGHT_FLUSH:
            if self.ranks[0] == Rank.ACE:
                return "royal flush"
            return f"straight flush, {names[0]}-high"
        if cat is HandCategory.FOUR_OF_A_KIND:
            return f"four of a kind, {_plural(self.ranks[0])}"
        if cat is HandCategory.FULL_HOUSE:
            return f"full house, {_plural(self.ranks[0])} full of {_plural(self.ranks[1])}"
        if cat is HandCategory.FLUSH:
            return f"flush, {names[0]}-high"
        if cat is HandCategory.STRAIGHT:
            return f"straight, {names[0]}-high"
        if cat is HandCategory.THREE_OF_A_KIND:
            return f"three of a kind, {_plural(self.ranks[0])}"
        if cat is HandCategory.TWO_PAIR:
            return f"two pair, {_plural(self.ranks[0])} and {_plural(self.ranks[1])}"
        if cat is HandCategory.PAIR:
            return f"a pair of {_plural(self.ranks[0])}"
        return f"{names[0]}-high"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.description


_RANK_NAMES: dict[int, str] = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "jack",
    12: "queen",
    13: "king",
    14: "ace",
}

_RANK_PLURALS: dict[int, str] = {
    2: "twos",
    3: "threes",
    4: "fours",
    5: "fives",
    6: "sixes",
    7: "sevens",
    8: "eights",
    9: "nines",
    10: "tens",
    11: "jacks",
    12: "queens",
    13: "kings",
    14: "aces",
}


def _plural(rank: int) -> str:
    return _RANK_PLURALS[rank]


def _pack(category: HandCategory, ranks: Sequence[int]) -> int:
    score = int(category)
    for i in range(5):
        score = score * _BASE + (ranks[i] if i < len(ranks) else 0)
    return score


def _straight_high(ranks: Iterable[int]) -> int | None:
    """Highest card of the best straight in ``ranks``, or None.

    Handles the wheel by treating an ace as a 1 as well as a 14.
    """

    present = set(ranks)
    if Rank.ACE in present:
        present.add(1)
    for high in range(14, 4, -1):
        if all(high - offset in present for offset in range(5)):
            return high
    return None


def evaluate(cards: Sequence[Card]) -> HandRank:
    """Evaluate the best five-card hand from 5, 6 or 7 cards."""

    if not 5 <= len(cards) <= 7:
        raise ValueError(f"need between 5 and 7 cards, got {len(cards)}")
    if len(set(cards)) != len(cards):
        raise ValueError("duplicate cards passed to evaluate()")

    ranks = [int(card.rank) for card in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(card.suit for card in cards)

    flush_suit = next((s for s, n in suit_counts.items() if n >= 5), None)

    if flush_suit is not None:
        flush_ranks = sorted(
            (int(c.rank) for c in cards if c.suit is flush_suit), reverse=True
        )
        sf_high = _straight_high(flush_ranks)
        if sf_high is not None:
            return _make(HandCategory.STRAIGHT_FLUSH, [sf_high])
        return _make(HandCategory.FLUSH, flush_ranks[:5])

    # Group ranks by how many times they appear, best count first then
    # highest rank first, so `groups[0]` is always the most significant group.
    groups = sorted(rank_counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top_rank, top_count = groups[0]

    if top_count == 4:
        kicker = max(r for r in rank_counts if r != top_rank)
        return _make(HandCategory.FOUR_OF_A_KIND, [top_rank, kicker])

    if top_count == 3:
        pair_rank = next((r for r, n in groups[1:] if n >= 2), None)
        if pair_rank is not None:
            return _make(HandCategory.FULL_HOUSE, [top_rank, pair_rank])

    straight = _straight_high(ranks)
    if straight is not None:
        return _make(HandCategory.STRAIGHT, [straight])

    if top_count == 3:
        kickers = sorted((r for r in rank_counts if r != top_rank), reverse=True)[:2]
        return _make(HandCategory.THREE_OF_A_KIND, [top_rank, *kickers])

    pairs = sorted((r for r, n in rank_counts.items() if n == 2), reverse=True)
    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        kicker = max(r for r in rank_counts if r not in (high, low))
        return _make(HandCategory.TWO_PAIR, [high, low, kicker])

    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted((r for r in rank_counts if r != pair), reverse=True)[:3]
        return _make(HandCategory.PAIR, [pair, *kickers])

    return _make(HandCategory.HIGH_CARD, sorted(ranks, reverse=True)[:5])


def _make(category: HandCategory, ranks: Sequence[int]) -> HandRank:
    packed = tuple(ranks)
    return HandRank(category=category, ranks=packed, score=_pack(category, packed))


def best_five(cards: Sequence[Card]) -> tuple[Card, ...]:
    """The specific five cards making up the best hand.

    Slower than :func:`evaluate` (it enumerates the 21 five-card subsets of a
    seven-card hand), so it is meant for explanation and display, not hot loops.
    """

    if len(cards) == 5:
        return tuple(cards)
    return max(combinations(cards, 5), key=lambda combo: evaluate(combo).score)


def compare(left: Sequence[Card], right: Sequence[Card]) -> int:
    """Return 1 if ``left`` wins, -1 if ``right`` wins, 0 on a tie."""

    a, b = evaluate(left).score, evaluate(right).score
    return (a > b) - (a < b)
