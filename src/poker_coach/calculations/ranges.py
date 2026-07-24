"""Hand-range notation parsing and combo expansion.

Supports the notation players actually write:

===================  =========================================================
``AA``, ``AKs``      a single class (pair / suited / offsuit)
``AK``               both ``AKs`` and ``AKo``
``77+``              that pair and every higher pair
``ATs+``             ``ATs`` through ``AKs`` (high card fixed)
``99-66``            an inclusive pair range
``A5s-A2s``          an inclusive range sharing a high card
``T9s-76s``          an inclusive gap-preserving connector range
``AsKd``             one explicit combo
``random`` / ``any`` every hand
===================  =========================================================

Tokens are separated by commas and/or whitespace. A leading ``-`` on a token
subtracts it from the range, so ``"22+, -55"`` is every pair except fives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from ..domain.cards import Card, Rank, Suit

__all__ = ["Range", "expand_classes", "class_combos", "canonical_class"]

_RANK_ORDER: tuple[Rank, ...] = tuple(sorted(Rank, reverse=True))
_RANK_SYMBOLS = "AKQJT98765432"
_SUIT_CHARS = "cdhs"

_CLASS_RE = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so]?)$", re.IGNORECASE)
_COMBO_RE = re.compile(
    rf"^([2-9TJQKA])([{_SUIT_CHARS}])([2-9TJQKA])([{_SUIT_CHARS}])$",
    re.IGNORECASE,
)


def canonical_class(high: Rank, low: Rank, suited: bool | None) -> str:
    """Build the canonical class string, e.g. ``"AKs"`` or ``"77"``."""

    if high < low:
        high, low = low, high
    if high == low:
        return f"{high.symbol}{low.symbol}"
    if suited is None:
        raise ValueError("non-pair classes must specify suited/offsuit")
    return f"{high.symbol}{low.symbol}{'s' if suited else 'o'}"


def _parse_class(token: str) -> tuple[Rank, Rank, bool | None]:
    match = _CLASS_RE.match(token)
    if not match:
        raise ValueError(f"not a hand class: {token!r}")
    r1 = Rank.from_symbol(match.group(1))
    r2 = Rank.from_symbol(match.group(2))
    flag = match.group(3).lower()

    high, low = (r1, r2) if r1 >= r2 else (r2, r1)
    if high == low:
        if flag:
            raise ValueError(f"pairs cannot be suited or offsuit: {token!r}")
        return high, low, None
    if not flag:
        return high, low, None  # caller expands to both s and o
    return high, low, flag == "s"


def _classes_for(high: Rank, low: Rank, suited: bool | None) -> list[str]:
    if high == low:
        return [canonical_class(high, low, None)]
    if suited is None:
        return [
            canonical_class(high, low, True),
            canonical_class(high, low, False),
        ]
    return [canonical_class(high, low, suited)]


def _expand_plus(high: Rank, low: Rank, suited: bool | None) -> list[str]:
    out: list[str] = []
    if high == low:
        for rank in Rank:
            if rank >= high:
                out.extend(_classes_for(rank, rank, None))
    else:
        for rank in Rank:
            if low <= rank < high:
                out.extend(_classes_for(high, rank, suited))
    return out


def _expand_dash(left: str, right: str) -> list[str]:
    h1, l1, s1 = _parse_class(left)
    h2, l2, s2 = _parse_class(right)

    # Check the pair/non-pair mix first: a pair token carries no suitedness, so
    # the suitedness check below would otherwise mask the real problem.
    if (h1 == l1) != (h2 == l2):
        raise ValueError(f"cannot mix pairs and non-pairs: {left}-{right}")

    if s1 != s2:
        raise ValueError(f"mismatched suitedness in range {left}-{right}")

    if h1 == l1 and h2 == l2:  # pair range, e.g. 99-66
        lo, hi = sorted((h1, h2))
        return [c for r in Rank if lo <= r <= hi for c in _classes_for(r, r, None)]

    if h1 == h2:  # shared high card, e.g. A5s-A2s
        lo, hi = sorted((l1, l2))
        return [c for r in Rank if lo <= r <= hi for c in _classes_for(h1, r, s1)]

    if h1 - l1 == h2 - l2:  # gap-preserving, e.g. T9s-76s
        gap = h1 - l1
        lo, hi = sorted((h1, h2))
        return [
            c
            for r in Rank
            if lo <= r <= hi
            for c in _classes_for(r, Rank(r - gap), s1)
        ]

    raise ValueError(
        f"unsupported range {left}-{right}: endpoints must share a high card, "
        "share a gap, or both be pairs"
    )


def _all_classes() -> list[str]:
    out: list[str] = []
    for i, high_sym in enumerate(_RANK_SYMBOLS):
        for low_sym in _RANK_SYMBOLS[i:]:
            high = Rank.from_symbol(high_sym)
            low = Rank.from_symbol(low_sym)
            out.extend(_classes_for(high, low, None))
    return out


def _tokenize(notation: str) -> Iterator[str]:
    for raw in notation.replace(",", " ").split():
        token = raw.strip()
        if token:
            yield token


def expand_classes(notation: str) -> list[str]:
    """Expand range notation into a sorted list of canonical class strings.

    Explicit combos (``"AsKd"``) are *not* returned here — use
    :meth:`Range.combos`, which keeps them at combo granularity.
    """

    include, exclude, combos = _parse_notation(notation)
    if combos:
        raise ValueError(
            "notation contains explicit combos; use Range.combos() instead"
        )
    return _sorted_classes(include - exclude)


def _parse_notation(
    notation: str,
) -> tuple[set[str], set[str], set[tuple[Card, Card]]]:
    include: set[str] = set()
    exclude: set[str] = set()
    explicit: set[tuple[Card, Card]] = set()

    for token in _tokenize(notation):
        negate = token.startswith("-") and not _looks_like_dash_range(token)
        body = token[1:] if negate else token
        target = exclude if negate else include

        lowered = body.lower()
        if lowered in ("random", "any", "*"):
            target.update(_all_classes())
            continue

        combo = _COMBO_RE.match(body)
        if combo:
            card_a = Card.from_str(body[:2])
            card_b = Card.from_str(body[2:])
            if card_a == card_b:
                raise ValueError(f"combo uses the same card twice: {body!r}")
            if negate:
                raise ValueError("negating explicit combos is not supported")
            explicit.add(tuple(sorted((card_a, card_b), reverse=True)))  # type: ignore[arg-type]
            continue

        if body.endswith("+"):
            high, low, suited = _parse_class(body[:-1])
            target.update(_expand_plus(high, low, suited))
            continue

        if "-" in body:
            left, _, right = body.partition("-")
            target.update(_expand_dash(left, right))
            continue

        high, low, suited = _parse_class(body)
        target.update(_classes_for(high, low, suited))

    return include, exclude, explicit


def _looks_like_dash_range(token: str) -> bool:
    """Distinguish ``"-55"`` (subtract) from a stray leading dash on a range."""

    return bool(re.match(r"^-[2-9TJQKA]{2}[so]?-", token, re.IGNORECASE))


def _class_sort_key(cls: str) -> tuple[int, int, int]:
    high = Rank.from_symbol(cls[0])
    low = Rank.from_symbol(cls[1])
    suited_rank = 0 if len(cls) == 2 else (1 if cls[2] == "s" else 2)
    return (-int(high), -int(low), suited_rank)


def _sorted_classes(classes: Iterable[str]) -> list[str]:
    return sorted(set(classes), key=_class_sort_key)


def class_combos(cls: str) -> list[tuple[Card, Card]]:
    """All specific two-card combos belonging to a canonical class."""

    high = Rank.from_symbol(cls[0])
    low = Rank.from_symbol(cls[1])
    suits = list(Suit)

    if high == low:
        return [
            (Card(high, s1), Card(low, s2))
            for i, s1 in enumerate(suits)
            for s2 in suits[i + 1 :]
        ]

    if len(cls) != 3:
        raise ValueError(f"non-pair class must end in 's' or 'o': {cls!r}")

    if cls[2] == "s":
        return [(Card(high, s), Card(low, s)) for s in suits]
    return [
        (Card(high, s1), Card(low, s2))
        for s1 in suits
        for s2 in suits
        if s1 is not s2
    ]


@dataclass(frozen=True, slots=True)
class Range:
    """A parsed hand range.

    >>> r = Range("77+, AQs+")
    >>> len(r.combos())
    56
    """

    notation: str

    @property
    def classes(self) -> list[str]:
        include, exclude, _ = _parse_notation(self.notation)
        return _sorted_classes(include - exclude)

    def combos(self, dead: Sequence[Card] = ()) -> list[tuple[Card, Card]]:
        """Every combo in the range, minus any using a blocked card."""

        include, exclude, explicit = _parse_notation(self.notation)
        blocked = set(dead)

        out: list[tuple[Card, Card]] = []
        for cls in _sorted_classes(include - exclude):
            out.extend(class_combos(cls))
        out.extend(explicit)

        seen: set[tuple[Card, Card]] = set()
        result: list[tuple[Card, Card]] = []
        for combo in out:
            key = tuple(sorted(combo, reverse=True))
            if key in seen:
                continue
            if blocked & set(combo):
                continue
            seen.add(key)  # type: ignore[arg-type]
            result.append(combo)
        return result

    def __contains__(self, combo: Sequence[Card]) -> bool:
        if len(combo) != 2:
            return False
        a, b = combo
        high, low = (a, b) if a.rank >= b.rank else (b, a)
        suited = None if high.rank == low.rank else (a.suit is b.suit)
        cls = canonical_class(high.rank, low.rank, suited)
        if cls in set(self.classes):
            return True
        _, _, explicit = _parse_notation(self.notation)
        return tuple(sorted(combo, reverse=True)) in explicit

    def __len__(self) -> int:
        return len(self.combos())

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.notation

    @property
    def percent_of_hands(self) -> float:
        """Share of the 1326 possible starting hands this range covers."""

        return 100.0 * len(self.combos()) / 1326.0
