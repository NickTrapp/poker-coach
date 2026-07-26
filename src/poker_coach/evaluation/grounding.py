"""Check that a coaching response only quotes numbers it was given.

The whole architecture rests on one rule: the model receives exact figures in
the FACTS block and must quote them rather than recompute or invent them. This
module is what actually verifies that. It extracts numeric claims from a
response and checks each against the values the model was handed.

Scope, stated plainly
---------------------
Poker text is full of digits that are not claims — card names (``Qs2s9c``),
range notation (``77+``, ``A5s-A2s``), and prose counts ("two pair", "3 outs").
Flagging those would drown the real signal, so:

* Card and range tokens are **masked out** before extraction.
* Percentages, decimals, ratios, and signed chip amounts **are** checked.
* Bare small integers (0-10) are **ignored by design** — they are far more
  often prose than quantities. This is a deliberate blind spot, not an
  oversight: a model that invents "you have 7 outs" will not be caught here.

So a clean report means "no fabricated statistics", not "every word is true".

**Values are checked, not meanings — and this is weaker than it sounds.** Every
percentage is checked against one pool of supplied percentages and every other
number against one pool of numbers. A claim passes when its *magnitude* appears
somewhere in the facts, regardless of what the sentence says the number means.
Demonstrated: with hero's true equity at 71.80% and MDF at 50%, the sentence
"Hero has 50% equity here" **passes**, because 50 is in the percentage pool.

So the honest guarantee is narrower than "no fabricated statistics": it is
*every number quoted appears somewhere in the facts*. Attaching it to the wrong
quantity is not caught. Allowing sign flips widens this slightly further.

Closing it properly means fact-addressed generation — give each fact an id and
a unit, have the model cite `[hero_equity_pct]`, and check the citation rather
than the digits. That is a change to how responses are produced, not a patch to
this module, which is why it is written down here rather than bolted on.

**Non-numeric claims are entirely unchecked, and this matters more than it
sounds.** In the first live run against a real model, the response passed with
all six numeric claims traceable — while also asserting the hero held a "nut
flush draw with two overcards". That was correct, and it was nowhere in the
facts: the block said only "Hero's made hand: ace-high". The model read the
board itself. Had it said "open-ended straight draw" instead, this checker
would still have reported a clean pass.

Hand-reading claims, board-texture claims, and range assertions are exactly the
kind of thing a coach can be confidently wrong about, and none of them are
numbers. Verifying them needs a different mechanism — most likely surfacing the
draw and texture facts in `SpotAnalysis` so there is something to check
*against*, rather than trying to parse prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..coaching.analysis import SpotAnalysis

__all__ = [
    "NumericClaim",
    "GroundingReport",
    "extract_claims",
    "grounded_values",
    "check_grounding",
    "SMALL_INTEGER_CEILING",
]

#: Bare integers at or below this are treated as prose, not statistics.
SMALL_INTEGER_CEILING = 10

#: Percentages are compared in percentage points.
PERCENT_TOLERANCE = 0.55

#: Plain numbers are compared with a relative and an absolute floor.
RELATIVE_TOLERANCE = 0.02
ABSOLUTE_TOLERANCE = 0.06

# Masked before number extraction.
#
# Every pattern demands a *distinguishing marker* — a suit, a face rank, a
# `+`/`s`/`o` suffix, or a dash range. A naive `[2-9TJQKA]{2}` would also match
# the digits inside real statistics ("33.3%" -> masks "33", leaving "3%"), which
# is precisely the kind of silent corruption this module exists to prevent.
#
# The deliberate consequence: a bare digit pair with no marker ("77" written on
# its own, meaning pocket sevens) is read as the number 77, not as notation.
# That form is rare in prose, and erring toward checking a number is safer than
# erring toward blanking one.
_RANK = "[2-9TJQKA]"
_FACE = "[TJQKA]"
_EDGE = r"(?<![\w.])"  # not mid-word and not inside a decimal
_END = r"(?![\w.])"

_MASK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Dash ranges: T9s-76s, A5s-A2s, 99-66
    re.compile(rf"{_EDGE}{_RANK}{_RANK}[so]?-{_RANK}{_RANK}[so]?{_END}"),
    # Plus ranges: 77+, ATs+, 22+
    re.compile(rf"{_EDGE}{_RANK}{_RANK}[so]?\+{_END}"),
    # Suited/offsuit classes: AKs, A5s, 87o
    re.compile(rf"{_EDGE}{_RANK}{_RANK}[so]{_END}"),
    # Bare classes containing a face rank: QQ, AK, AQ, T9
    re.compile(rf"{_EDGE}(?:{_FACE}{_RANK}|{_RANK}{_FACE}){_END}"),
    # Numeric pocket pairs — but only where the text marks them as notation.
    # Treating every standalone 22, 33 ... 99 as a rank blinded the checker to
    # eight values that occur naturally as pots, stacks and raise sizes, on a
    # checker that already ignores 0-10. So a signal is required:
    #   `77`                      inline code, how the live failure was written
    #   pocket 77 / set of 77 / pair of 77
    # "The pot is 77 now" and "call 44 chips" stay checkable.
    #
    # "holds" and "holding" are deliberately absent: they read equally well
    # before cards and before quantities, so "villain holds 77% equity" lost
    # its percentage. A trailing percent sign vetoes the whole rule for the
    # same reason — no rank is ever written with one.
    re.compile(r"`([2-9])\1`"),
    re.compile(
        rf"(?i:(?<=pocket )|(?<=set of )|(?<=sets of )|(?<=pair of ))"
        rf"([2-9])\1(?![\w.]|\s*%)"
    ),
    # Explicit cards, possibly concatenated: AsKs, Qs2s9c
    re.compile(rf"{_EDGE}(?:{_RANK}[cdhs])+{_END}"),
)

_PERCENT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?\s*%")
_RATIO_RE = re.compile(r"\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?")
# Comma-grouped form first, so "4,000" reads as 4000 rather than 4 and 000.
_NUMBER_RE = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class NumericClaim:
    """A number the response asserts, with how it was written."""

    text: str
    value: float
    kind: str  # "percent" | "ratio" | "number"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.text} ({self.kind})"


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """The verdict on one response."""

    claims: list[NumericClaim]
    grounded: list[NumericClaim] = field(default_factory=list)
    ungrounded: list[NumericClaim] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True when no checked number was invented."""

        return not self.ungrounded

    @property
    def checked_count(self) -> int:
        return len(self.grounded) + len(self.ungrounded)

    def summary(self) -> str:
        if not self.checked_count:
            return "No numeric claims to check."
        if self.is_grounded:
            return f"All {self.checked_count} numeric claim(s) trace to the facts."
        listed = ", ".join(c.text for c in self.ungrounded)
        return (
            f"{len(self.ungrounded)} of {self.checked_count} numeric claim(s) "
            f"are not in the facts: {listed}"
        )

    def __bool__(self) -> bool:
        return self.is_grounded

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.summary()


def _blank(match: re.Match[str]) -> str:
    return " " * len(match.group(0))


def mask_poker_notation(text: str, known: Iterable[str] = ()) -> str:
    """Blank out card and range tokens so their digits aren't read as claims.

    ``known`` are notation strings taken verbatim from the spot — the board,
    the hero's cards, the assumed villain range. Masking those literally is
    far more reliable than pattern matching, and it is what rescues ranges
    made of bare digit pairs (``"QQ, 99, 77, 22"``) that the generic patterns
    deliberately leave alone.
    """

    masked = text
    for token in sorted((t for t in known if t), key=len, reverse=True):
        # Guarded on both sides: an unguarded literal mask of the range element
        # "77" would eat the leading digits of the equity figure "77.40%" and
        # leave a phantom "40%" behind. Only a following digit (or a decimal
        # point introducing one) blocks the mask, so "villain has 77." works.
        masked = re.sub(rf"{_EDGE}{re.escape(token)}(?!\.?\d)", _blank, masked)
    for pattern in _MASK_PATTERNS:
        masked = pattern.sub(_blank, masked)
    return masked


def extract_claims(text: str, known: Iterable[str] = ()) -> list[NumericClaim]:
    """Pull every checkable numeric claim out of a response."""

    masked = mask_poker_notation(text, known)
    claims: list[NumericClaim] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < e and s < end for s, e in consumed)

    for match in _PERCENT_RE.finditer(masked):
        raw = match.group(0)
        claims.append(
            NumericClaim(
                text=raw.strip(),
                value=float(raw.replace("%", "").strip()),
                kind="percent",
            )
        )
        consumed.append(match.span())

    for match in _RATIO_RE.finditer(masked):
        if overlaps(*match.span()):
            continue
        raw = match.group(0)
        left, right = (part.strip() for part in raw.split(":"))
        denominator = float(right)
        claims.append(
            NumericClaim(
                text=raw.strip(),
                value=float(left) / denominator if denominator else float("inf"),
                kind="ratio",
            )
        )
        consumed.append(match.span())

    for match in _NUMBER_RE.finditer(masked):
        if overlaps(*match.span()):
            continue
        raw = match.group(0)
        value = float(raw.replace(",", ""))
        # Bare small integers are prose far more often than statistics.
        if raw.lstrip("+-").isdigit() and abs(value) <= SMALL_INTEGER_CEILING:
            continue
        claims.append(NumericClaim(text=raw, value=value, kind="number"))
        consumed.append(match.span())

    return claims


def grounded_values(analysis: SpotAnalysis) -> tuple[set[float], set[float]]:
    """The percentages and plain numbers the model was actually given.

    Returns ``(percents, numbers)``. Both the fraction and the percentage form
    of each rate are allowed, since either is a faithful quote.
    """

    equity = analysis.equity
    odds = analysis.odds

    percents: set[float] = {
        equity.equity_pct,
        100.0 * equity.win,
        100.0 * equity.tie,
        100.0 * equity.lose,
        100.0 * equity.margin_of_error,
        odds.required_equity_pct,
    }
    numbers: set[float] = {
        analysis.total_pot,
        analysis.to_call,
        analysis.pot_after_call,
        analysis.effective_stack,
        analysis.hero_stack_behind,
        analysis.ev_call_vs_fold,
        equity.equity,
        odds.required_equity,
        float(equity.samples),
        # The facts block renders the margin of error as a bare number
        # ("56.04% ±0.69"), so the percentage-point form counts as a number too.
        equity.margin_of_error,
        100.0 * equity.margin_of_error,
    }

    # Deterministic hand features are quotable facts too — an out-count the
    # coach cites must match the one it was given.
    features = analysis.features
    if features.straight_draw is not None:
        numbers.add(float(features.straight_draw.outs))
    if features.overcards:
        numbers.add(float(len(features.overcards)))

    if analysis.spr != float("inf"):
        numbers.add(analysis.spr)
    if odds.ratio != float("inf"):
        numbers.add(odds.ratio)
    if analysis.implied_odds_needed not in (None, float("inf")):
        numbers.add(analysis.implied_odds_needed)
    if analysis.mdf is not None:
        percents.add(100.0 * analysis.mdf)
        numbers.add(analysis.mdf)
    if analysis.alpha is not None:
        percents.add(100.0 * analysis.alpha)
        numbers.add(analysis.alpha)

    # A quoted figure may be negated ("losing 4.09 chips") — allow the sign flip.
    numbers |= {-n for n in numbers}
    return percents, numbers


def _known_notation(analysis: SpotAnalysis) -> list[str]:
    """Notation strings the spot itself supplied, longest first when masking."""

    tokens = [analysis.villain_range, analysis.board, analysis.hero_cards]
    # A range is a comma-separated list; mask the whole line and each element,
    # so a response that cites only part of it is still handled.
    tokens += [part.strip() for part in analysis.villain_range.split(",")]
    return [t for t in tokens if t]


def _matches(value: float, allowed: Iterable[float], *, percent: bool) -> bool:
    for candidate in allowed:
        if percent:
            if abs(value - candidate) <= PERCENT_TOLERANCE:
                return True
        else:
            tolerance = max(
                ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * abs(candidate)
            )
            if abs(value - candidate) <= tolerance:
                return True
    return False


def check_grounding(response: str, analysis: SpotAnalysis) -> GroundingReport:
    """Verify every checkable number in ``response`` came from ``analysis``."""

    percents, numbers = grounded_values(analysis)
    # The spot's own notation is masked literally — the villain-range line in
    # particular is dense with digit pairs that are ranks, not quantities.
    known = _known_notation(analysis)
    claims = extract_claims(response, known)

    grounded: list[NumericClaim] = []
    ungrounded: list[NumericClaim] = []

    for claim in claims:
        if claim.kind == "percent":
            ok = _matches(claim.value, percents, percent=True)
        elif claim.kind == "ratio":
            ok = _matches(claim.value, numbers, percent=False)
        else:
            # A plain number may be quoting a chip figure or a rate written as
            # a fraction, so it is checked against both pools.
            ok = _matches(claim.value, numbers, percent=False) or _matches(
                claim.value * 100.0, percents, percent=True
            )
        (grounded if ok else ungrounded).append(claim)

    return GroundingReport(claims=claims, grounded=grounded, ungrounded=ungrounded)
