"""A small benchmark of spots whose correct action is settled by arithmetic.

Each case pins a hero hand, a board, a price, and a villain range chosen so the
EV of calling is decisively positive or negative — not a close judgement call.
That makes the expected verdict checkable without a human, which is the only
way this suite stays useful as a regression gate.

What the arithmetic actually establishes
---------------------------------------
A positive call-versus-fold figure shows that **continuing beats folding**. It
does not show that calling is the uniquely correct action — nothing computed in
this package compares calling with raising. So a case's expectation is
``"continue"`` or ``"fold"``, and a coach that recommends raising in a
``"continue"`` spot has not made an error by this benchmark's lights.

Grading is two-part and both halves must pass:

1. **Recommendation** — is the response on the correct side of the
   continue/fold line the arithmetic settles?
2. **Grounding** — did it get there by quoting the supplied numbers, rather
   than inventing its own? A right answer for fabricated reasons still fails.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from ..coaching.analysis import SpotAnalysis, analyze
from ..domain.cards import parse_cards
from ..domain.enums import Position, Street
from ..domain.state import HandState, PlayerState
from .grounding import GroundingReport, check_grounding

__all__ = [
    "Case",
    "CaseResult",
    "STANDARD_CASES",
    "grade",
    "recommended_action",
]

Action = Literal["call", "raise", "fold", "unclear"]

#: What the supplied arithmetic can actually settle.
Verdict = Literal["continue", "fold"]

_CALL_WORDS = ("call", "calling")
_RAISE_WORDS = ("raise", "raising", "shove", "jam")
_FOLD_WORDS = ("fold", "folding")

#: Actions that count as continuing rather than folding.
CONTINUING_ACTIONS = frozenset({"call", "raise"})


@dataclass(frozen=True, slots=True)
class Case:
    """One benchmark spot with its settled answer."""

    name: str
    hero_cards: str
    board: str
    pot: float
    to_call: float
    villain_range: str
    #: What the arithmetic settles — "continue" or "fold", never "call",
    #: because nothing here evaluates raising.
    expected: Verdict
    rationale: str
    hero_stack: float = 100.0
    range_conditioning: str = "unspecified"
    description: str = ""

    def state(self) -> HandState:
        """Build the hand state the coach will be asked about."""

        street = {0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}[
            len(parse_cards(self.board)) if self.board else 0
        ]
        players = [
            PlayerState(
                name="hero",
                position=Position.BTN,
                stack=self.hero_stack,
                hole_cards=tuple(parse_cards(self.hero_cards)),
                is_hero=True,
            ),
            PlayerState(
                name="villain",
                position=Position.BB,
                stack=self.hero_stack,
                committed_this_street=self.to_call,
            ),
        ]
        return HandState(
            players=players,
            board=parse_cards(self.board) if self.board else [],
            street=street,
            # `pot` is the dead money from prior streets; villain's live bet is
            # tracked separately, so subtract it to reach the stated total.
            pot=self.pot - self.to_call,
        )

    def analyse(
        self, *, iterations: int = 6000, rng: random.Random | None = None
    ) -> SpotAnalysis:
        return analyze(
            self.state(),
            villain_range=self.villain_range,
            range_conditioning=self.range_conditioning,
            iterations=iterations,
            rng=rng or random.Random(17),
        )


@dataclass(frozen=True, slots=True)
class CaseResult:
    """How one response scored on one case."""

    case: Case
    analysis: SpotAnalysis
    response: str
    action: Action
    grounding: GroundingReport
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict | None:
        """Which side of the continue/fold line the response landed on."""

        return verdict_of(self.action)

    @property
    def action_correct(self) -> bool:
        """Whether the response is on the side the arithmetic supports.

        A raise and a call both count as continuing. This benchmark cannot
        distinguish them, and pretending otherwise would score a defensible
        raise as an error.
        """

        return self.verdict == self.case.expected

    @property
    def passed(self) -> bool:
        """Both halves must hold: right side of the line, reached honestly."""

        return self.action_correct and self.grounding.is_grounded

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        parts = [
            f"{verdict} {self.case.name}",
            f"expected {self.case.expected}, read {self.action} "
            f"({self.verdict or 'unclear'})",
            self.grounding.summary(),
        ]
        return " | ".join(parts)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.summary()


def _first_index(text: str, words: tuple[str, ...]) -> int:
    hits = [text.find(w) for w in words if text.find(w) != -1]
    return min(hits) if hits else -1


def recommended_action(response: str) -> Action:
    """Read the recommended action out of a coaching response.

    Deliberately simple: it takes whichever action word appears first, since the
    system prompt requires the recommendation to lead. A response that buries or
    hedges its recommendation reads as ``"unclear"`` — a real finding about the
    response, not a parser failure.

    ``"raise"`` is recognised as its own action rather than being folded into
    ``"call"``, because the two are not interchangeable and the benchmark must
    not silently score a raise as a call.
    """

    lowered = response.lower()
    found = {
        "call": _first_index(lowered, _CALL_WORDS),
        "raise": _first_index(lowered, _RAISE_WORDS),
        "fold": _first_index(lowered, _FOLD_WORDS),
    }
    present = {action: i for action, i in found.items() if i != -1}
    if not present:
        return "unclear"
    return min(present, key=present.get)  # type: ignore[return-value]


def verdict_of(action: Action) -> Verdict | None:
    """Which side of the continue/fold line an action falls on."""

    if action in CONTINUING_ACTIONS:
        return "continue"
    if action == "fold":
        return "fold"
    return None


def grade(
    case: Case,
    response: str,
    *,
    analysis: SpotAnalysis | None = None,
    iterations: int = 6000,
    rng: random.Random | None = None,
) -> CaseResult:
    """Score one response against one case."""

    spot = analysis or case.analyse(iterations=iterations, rng=rng)
    action = recommended_action(response)
    grounding = check_grounding(response, spot)

    notes: list[str] = []
    if action == "unclear":
        notes.append("no clear recommendation found in the response")
    if not grounding.is_grounded:
        notes.append(grounding.summary())
    if action != "unclear" and verdict_of(action) != case.expected:
        notes.append(f"expected to {case.expected} because {case.rationale}")
    if action == "raise" and case.expected == "continue":
        notes.append(
            "recommended raising; scored as continuing, since the supplied "
            "arithmetic does not compare calling with raising"
        )

    return CaseResult(
        case=case,
        analysis=spot,
        response=response,
        action=action,
        grounding=grounding,
        notes=notes,
    )


#: Spots chosen so the arithmetic, not taste, decides the continue/fold line.
#: Between them they cover the semantics this package has to get right:
#: terminal vs non-terminal EV, sampled vs exact equity, and pre-action vs
#: action-conditioned ranges.
STANDARD_CASES: tuple[Case, ...] = (
    Case(
        name="river-bluff-catcher-close",
        description="River bluff-catcher where the price is close to the edge.",
        hero_cards="Ah5d",
        board="Ac8h3d5s2c",
        pot=60.0,
        to_call=45.0,
        villain_range="AK, AQ, AJ",
        range_conditioning="action-conditioned",
        expected="continue",
        rationale="hero has two pair against a range of one-pair hands, and the "
                  "river call is terminal so the EV figure is exact",
    ),
    Case(
        name="river-drawing-dead",
        description="Terminal river call with zero equity.",
        hero_cards="7c2d",
        board="AhKhQd9s3c",
        pot=100.0,
        to_call=75.0,
        villain_range="AA, KK, QQ, AK",
        range_conditioning="action-conditioned",
        expected="fold",
        rationale="hero has seven-high and is drawing dead on a complete board",
    ),
    Case(
        name="river-made-flush-exact",
        description="Exact-equity river spot: villain's range is a single combo.",
        hero_cards="AsKs",
        board="Qs7s2h9s3d",
        pot=100.0,
        to_call=50.0,
        villain_range="QhQd",
        range_conditioning="action-conditioned",
        expected="continue",
        rationale="hero holds the nut flush against a set; enumeration is exact",
    ),
    Case(
        name="flop-draw-future-action",
        description="Non-all-in flop draw — future betting is not modelled.",
        hero_cards="AsKs",
        board="Qs2s9c",
        pot=12.0,
        to_call=6.0,
        villain_range="22+, ATs+, KQs, AJo+",
        range_conditioning="pre-action",
        expected="continue",
        rationale="the nut flush draw with two overcards clears the price even "
                  "before any implied odds",
        hero_stack=97.0,
    ),
    Case(
        name="turn-draw-realization",
        description="Turn draw where realization matters: one card to come.",
        hero_cards="Ts9s",
        board="8s7c2d4h",
        pot=40.0,
        to_call=30.0,
        villain_range="88+, 87s, 77, 44",
        range_conditioning="action-conditioned",
        expected="fold",
        rationale="one card to come against a made range; the draw does not "
                  "price in without implied odds the facts do not supply",
        hero_stack=80.0,
    ),
    Case(
        name="flop-raise-plausible",
        description="Raising is a plausible alternative to calling.",
        hero_cards="AhAd",
        board="Ac7h2s",
        pot=20.0,
        to_call=15.0,
        villain_range="77, 22, 76s, AK, AQ",
        range_conditioning="action-conditioned",
        expected="continue",
        rationale="top set on a dry board; either calling or raising continues, "
                  "and the arithmetic does not choose between them",
        hero_stack=150.0,
    ),
    Case(
        name="narrow-action-conditioned-range",
        description="Deliberately narrow range conditioned on the observed bet.",
        hero_cards="KdKc",
        board="AhQs4d8c2h",
        pot=80.0,
        to_call=60.0,
        villain_range="AA, AQ, A8s",
        range_conditioning="action-conditioned",
        expected="fold",
        rationale="against a range that always has an ace, an underpair is "
                  "beaten by every combo",
    ),
    Case(
        name="broad-pre-action-range-misleading",
        description="A broad pre-action range flatters hero; the facts must say so.",
        hero_cards="Js Td".replace(" ", ""),
        board="Ah7c2d",
        pot=20.0,
        to_call=15.0,
        villain_range="random",
        range_conditioning="pre-action",
        expected="fold",
        rationale="jack-high has no equity share worth the price once villain's "
                  "betting range is considered, and 'random' is not that range",
        hero_stack=85.0,
    ),
    Case(
        name="dry-board-made-hand",
        description="Dry-board made-hand decision with sampled equity.",
        hero_cards="QhQd",
        board="Qc7h2s",
        pot=24.0,
        to_call=18.0,
        villain_range="22+, A2s+, K9s+, QTs+, ATo+, KQo",
        range_conditioning="pre-action",
        expected="continue",
        rationale="top set on a dry board is ahead of the entire range",
        hero_stack=120.0,
    ),
)
