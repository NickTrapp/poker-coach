"""Street-by-street review of a hand the student already played.

Where `analysis.py` answers "what should I do here", this answers "what did I
do, and was it right". It replays a `HandHistory` through the same state
machine live play uses, computes a `SpotAnalysis` at each of the hero's
decision points, and pairs each one with the action actually taken.

The same grounding rule applies: the model receives exact numbers and the
action taken, and judges the action against those numbers. It never recomputes
them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..calculations.ranges import Range
from ..domain.history import DecisionPoint, HandHistory
from ..domain.state import HandState
from .analysis import DEFAULT_VILLAIN_RANGE, SpotAnalysis, analyze

__all__ = ["ReviewedDecision", "HandReviewFacts", "build_review_facts"]


@dataclass(frozen=True, slots=True)
class ReviewedDecision:
    """One hero decision: the facts as they stood, and what the hero did."""

    index: int
    street: str
    analysis: SpotAnalysis
    action_taken: str
    action_type: str
    amount: float

    @property
    def was_a_call(self) -> bool:
        return self.action_type == "call"

    @property
    def call_ev_verdict(self) -> str | None:
        """A blunt, purely arithmetic verdict on a call. None if not a call.

        This is deliberately mechanical: it states only what the EV number says,
        leaving the judgement of *why* to the model.
        """

        if not self.was_a_call:
            return None
        ev = self.analysis.ev_of_calling
        if abs(ev) < 1e-9:
            return "break-even by pot odds"
        return (
            f"{'+' if ev > 0 else ''}{ev:.2f} chips by pot odds "
            f"({'profitable' if ev > 0 else 'losing'})"
        )

    def to_prompt_block(self) -> str:
        lines = [
            f"DECISION {self.index} ({self.street})",
            self.analysis.to_prompt_block(),
            f"Action taken: {self.action_taken}",
        ]
        verdict = self.call_ev_verdict
        if verdict:
            lines.append(f"Arithmetic verdict on the call: {verdict}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class HandReviewFacts:
    """Every hero decision in a hand, with the facts behind each one."""

    hero: str
    decisions: list[ReviewedDecision]
    final_state: HandState

    def to_prompt_block(self) -> str:
        if not self.decisions:
            return f"Hero ({self.hero}) faced no decisions in this hand."
        header = f"Hero: {self.hero} — {len(self.decisions)} decision(s) to review."
        blocks = [decision.to_prompt_block() for decision in self.decisions]
        return "\n\n".join([header, *blocks])

    def __len__(self) -> int:
        return len(self.decisions)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.to_prompt_block()


def _describe(point: DecisionPoint) -> str:
    action = point.action
    if action.amount:
        return f"{action.type.value} to {action.amount:g}"
    return action.type.value


def build_review_facts(
    history: HandHistory,
    *,
    villain_range: str | Range = DEFAULT_VILLAIN_RANGE,
    iterations: int = 10_000,
    rng: random.Random | None = None,
) -> HandReviewFacts:
    """Replay ``history`` and analyse every decision the hero faced.

    Decision points where the hero holds no cards are skipped rather than
    raising: a history with folded-out hole cards is still worth partially
    reviewing.
    """

    rng_obj = rng or random.Random()
    decisions: list[ReviewedDecision] = []
    index = 0

    for point in history.replay():
        if not point.is_hero:
            continue

        hero_seat = point.state.player(point.action.actor)
        if hero_seat.hole_cards is None:
            continue

        index += 1
        analysis = analyze(
            point.state,
            hero=point.action.actor,
            villain_range=villain_range,
            iterations=iterations,
            rng=rng_obj,
        )
        decisions.append(
            ReviewedDecision(
                index=index,
                street=point.street.value,
                analysis=analysis,
                action_taken=_describe(point),
                action_type=point.action.type.value,
                amount=point.action.amount,
            )
        )

    return HandReviewFacts(
        hero=history.hero,
        decisions=decisions,
        final_state=history.final_state(),
    )
