"""Deterministic hand and board features.

The first live run exposed the gap this fills: the model correctly identified a
"nut flush draw with two overcards", but that reading came from the model, not
from the facts. It happened to be right; a wrong hand-read would have been
indistinguishable. Anything the coach may assert about the *shape* of a hand
needs to be computed here so there is something to check against.

Scope is deliberately narrow — draws, overcards, and board texture. This is not
a strategic board classifier and makes no claim about which textures favour
whom.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from ..domain.cards import Card, Rank, Suit, remaining_deck
from .hand_eval import HandCategory, evaluate

__all__ = [
    "DrawKind",
    "FlushDraw",
    "StraightDraw",
    "BoardTexture",
    "HandFeatures",
    "analyse_features",
    "board_texture",
]


class DrawKind(str):
    """Marker type kept simple; the concrete kinds are the dataclasses below."""


@dataclass(frozen=True, slots=True)
class FlushDraw:
    """A four-to-a-flush holding, with whether it is the nut draw."""

    suit: Suit
    cards_held: int
    is_nut: bool
    #: True when the hero holds both of the suited cards making the draw.
    uses_both_hole_cards: bool

    @property
    def description(self) -> str:
        prefix = "nut flush draw" if self.is_nut else "flush draw"
        return f"{prefix} ({self.suit.value})"


@dataclass(frozen=True, slots=True)
class StraightDraw:
    """An open-ended, gutshot, or double-gutshot straight draw."""

    kind: str  # "open-ended" | "gutshot" | "double-gutshot"
    outs: int

    @property
    def description(self) -> str:
        return f"{self.kind} straight draw ({self.outs} outs)"


@dataclass(frozen=True, slots=True)
class BoardTexture:
    """Structural properties of the community cards."""

    suitedness: str  # "rainbow" | "two-tone" | "monotone" | "n/a"
    is_paired: bool
    is_connected: bool
    high_card: Rank | None

    @property
    def description(self) -> str:
        if self.high_card is None:
            return "no board yet"
        parts = [self.suitedness]
        if self.is_paired:
            parts.append("paired")
        if self.is_connected:
            parts.append("connected")
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class HandFeatures:
    """Everything deterministic that can be said about hero's holding."""

    made_category: HandCategory | None
    made_description: str | None
    flush_draw: FlushDraw | None
    straight_draw: StraightDraw | None
    overcards: tuple[Rank, ...]
    texture: BoardTexture
    #: Cards of villain's possible holdings removed by hero's own cards.
    blocks_nut_flush: bool

    @property
    def has_draw(self) -> bool:
        return self.flush_draw is not None or self.straight_draw is not None

    def descriptions(self) -> list[str]:
        """Each feature as a short phrase, for rendering into the facts."""

        out: list[str] = []
        if self.made_description:
            out.append(self.made_description)
        if self.flush_draw:
            out.append(self.flush_draw.description)
        if self.straight_draw:
            out.append(self.straight_draw.description)
        if self.overcards:
            names = "".join(r.symbol for r in self.overcards)
            plural = "s" if len(self.overcards) > 1 else ""
            out.append(f"{len(self.overcards)} overcard{plural} ({names})")
        return out


def board_texture(board: Sequence[Card]) -> BoardTexture:
    """Suitedness, pairing and connectedness of the community cards."""

    if not board:
        return BoardTexture(
            suitedness="n/a", is_paired=False, is_connected=False, high_card=None
        )

    suit_counts = Counter(card.suit for card in board)
    top = max(suit_counts.values())
    if top >= 3 and len(board) >= 3:
        suitedness = "monotone" if top == len(board) else "two-tone"
    elif top == 2:
        suitedness = "two-tone"
    else:
        suitedness = "rainbow"

    ranks = sorted({int(c.rank) for c in board})
    connected = any(
        b - a <= 2 for a, b in zip(ranks, ranks[1:])
    ) if len(ranks) > 1 else False

    return BoardTexture(
        suitedness=suitedness,
        is_paired=len(set(c.rank for c in board)) < len(board),
        is_connected=connected,
        high_card=max(card.rank for card in board),
    )


def _plays_the_hand(
    hole: Sequence[Card], board: Sequence[Card], out: Card
) -> bool:
    """True when ``out`` gives hero something the community cards alone do not.

    A draw belongs to hero only if hero's cards are part of it. With a board of
    JsTs7c9c an eight completes J-T-9-8-7 *on the board* — every player holds
    that straight, so calling it hero's "gutshot" is wrong twice over: it is
    not a draw and it is not hero's.

    Fewer than five community cards cannot make a hand on their own, so
    everything counts at that point.
    """

    community = [*board, out]
    if len(community) < 5:
        return True
    return evaluate([*hole, *community]).score > evaluate(community).score


def _flush_draw(hole: Sequence[Card], board: Sequence[Card]) -> FlushDraw | None:
    # A draw needs a card to come. On a complete board there is nothing left to
    # draw to, however many of a suit hero happens to hold.
    if len(board) >= 5:
        return None

    cards = [*hole, *board]
    counts = Counter(card.suit for card in cards)

    for suit, count in counts.items():
        if count != 4:  # exactly four means a draw; five is a made flush
            continue

        held = [c for c in hole if c.suit is suit]
        if not held:
            continue  # the board alone is four-suited; hero has no draw

        # The nut draw is the one holding the highest suited card still unseen.
        seen = {c for c in cards if c.suit is suit}
        higher_available = [
            Card(rank, suit)
            for rank in Rank
            if Card(rank, suit) not in seen
        ]
        best_outstanding = max((c.rank for c in higher_available), default=None)
        my_best = max(c.rank for c in held)
        is_nut = best_outstanding is None or my_best > best_outstanding

        return FlushDraw(
            suit=suit,
            cards_held=len(held),
            is_nut=is_nut,
            uses_both_hole_cards=len(held) == 2,
        )
    return None


def _straight_draw(hole: Sequence[Card], board: Sequence[Card]) -> StraightDraw | None:
    """Classify a straight draw by counting the cards that would complete one.

    Only meaningful while cards are still to come: on a complete board there is
    no draw, and adding a hypothetical card would exceed what the evaluator
    accepts.
    """

    if len(board) >= 5:
        return None

    cards = [*hole, *board]
    if len(cards) >= 5 and evaluate(cards).category >= HandCategory.STRAIGHT:
        return None  # already made a straight or better

    outs = []
    for card in remaining_deck(sorted(set(cards), key=str)):
        category = evaluate([*cards, card]).category
        if not (HandCategory.STRAIGHT <= category < HandCategory.FLUSH):
            continue
        if not _plays_the_hand(hole, board, card):
            continue  # the board makes that straight without hero
        outs.append(card)
    if not outs:
        return None

    distinct = {out.rank for out in outs}
    count = len(outs)
    if len(distinct) >= 2:
        kind = "open-ended" if len(distinct) == 2 else "double-gutshot"
    else:
        kind = "gutshot"
    return StraightDraw(kind=kind, outs=count)


def analyse_features(
    hole: Sequence[Card], board: Sequence[Card]
) -> HandFeatures:
    """Compute every deterministic feature of ``hole`` against ``board``."""

    if len(hole) != 2:
        raise ValueError(f"a hold'em hand has 2 hole cards, got {len(hole)}")

    texture = board_texture(board)
    made_category = made_description = None
    if len(board) >= 3:
        rank = evaluate([*hole, *board])
        made_category = rank.category
        made_description = rank.description

    flush_draw = _flush_draw(hole, board) if board else None
    straight_draw = _straight_draw(hole, board) if len(board) >= 3 else None

    overcards: tuple[Rank, ...] = ()
    if board:
        top = max(card.rank for card in board)
        overcards = tuple(
            sorted((c.rank for c in hole if c.rank > top), reverse=True)
        )

    # Holding the ace of a flush-relevant suit denies villain the nut flush.
    blocks_nut = False
    if board:
        suit_counts = Counter(card.suit for card in board)
        for suit, count in suit_counts.items():
            if count >= 2 and any(
                c.suit is suit and c.rank is Rank.ACE for c in hole
            ):
                blocks_nut = True

    return HandFeatures(
        made_category=made_category,
        made_description=made_description,
        flush_draw=flush_draw,
        straight_draw=straight_draw,
        overcards=overcards,
        texture=texture,
        blocks_nut_flush=blocks_nut,
    )
