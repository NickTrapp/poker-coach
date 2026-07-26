"""Feedback split by what actually justifies it.

The system can say some things with certainty, some things directionally, and
some things not at all. Rendering them as one block of prose invites a student
to read all of it with the same confidence — which is the failure this project
has been guarding against on the model's side and would reintroduce on the
interface side.

So a critique has three visibly separate parts:

``verified``
    Calculations over supplied facts, generated deterministically with no model
    involved. "Deterministic" is not "exact": when equity was sampled, every
    figure derived from it inherits that sampling error, and the heading says
    so. A terminal call has an exact decision *tree* — no betting follows — but
    its EV is still an estimate if the equity feeding it was.
``interpretation``
    The model's reading — exploitative advice, hand-reading, what to watch for.
    Useful, not proven, and explicitly conditioned on an assumed opponent range
    the caller configured rather than the system inferred.
``unresolved``
    Questions the supplied calculations do not answer, taken straight from the
    OPEN facts. Listed so the student can see the edge of what was established.

No binary correct/incorrect grade is offered except where the arithmetic is
exact — a terminal call against a fixed range — because nothing here computes a
best action. See `docs/semantics-audit.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..coaching.analysis import SpotAnalysis
from ..coaching.facts import FactCategory
from ..domain.enums import ActionType
from ..evaluation.grounding import GroundingReport

__all__ = ["Feedback", "verified_lines", "unresolved_lines", "build_feedback"]


@dataclass(frozen=True, slots=True)
class Feedback:
    """One critique of one decision, with its parts kept apart."""

    action_taken: str
    verified: list[str]
    unresolved: list[str]
    #: Whether the equity behind the verified figures was enumerated rather
    #: than sampled. Drives the heading, which must not claim exactness the
    #: numbers underneath do not have.
    equity_exact: bool = True
    interpretation: str | None = None
    grounding: GroundingReport | None = None
    repair_attempted: bool = False
    #: True when the model's text was withheld and only arithmetic is shown.
    fell_back: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def has_interpretation(self) -> bool:
        return bool(self.interpretation and self.interpretation.strip())

    def render(self) -> str:
        blocks: list[str] = []

        if self.verified:
            qualifier = (
                "Deterministically computed under the stated assumptions."
                if self.equity_exact
                else "Deterministically computed from a sampled equity "
                     "estimate; respect the stated margin of error."
            )
            blocks.append(
                "VERIFIED CALCULATIONS\n"
                f"({qualifier})\n"
                + "\n".join(f"  {line}" for line in self.verified)
            )

        if self.has_interpretation:
            blocks.append(
                "EXPLOITATIVE INTERPRETATION\n"
                "(The coach's reading. Not proven, and only as good as the "
                "assumed range.)\n"
                + "\n".join(f"  {line}" for line in
                            self.interpretation.strip().splitlines())
            )
        elif self.fell_back:
            blocks.append(
                "EXPLOITATIVE INTERPRETATION\n"
                "  (withheld — the coach cited numbers it was not given, so "
                "only the arithmetic above is shown)"
            )

        if self.unresolved:
            blocks.append(
                "UNRESOLVED\n"
                "(Nothing above settles these.)\n"
                + "\n".join(f"  {line}" for line in self.unresolved)
            )

        if self.warnings:
            blocks.append(
                "WARNINGS\n" + "\n".join(f"  {w}" for w in self.warnings)
            )

        return "\n\n".join(blocks)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.render()


def verified_lines(
    analysis: SpotAnalysis, action_taken: str,
    action_type: ActionType | None = None,
) -> list[str]:
    """Statements the arithmetic supports, with no model involved.

    Framed around the action actually taken. Telling someone who just raised
    that "calling beats folding by 0.31 chips" is a non-sequitur — it answers a
    question they did not ask, about an action they did not take, and reads as
    though it were a verdict on their raise.

    Deliberately narrow. Each line says what was compared and under what
    assumption; none claims the action was optimal.
    """

    lines: list[str] = []
    equity = analysis.equity

    provenance = (
        "exact enumeration" if equity.exact
        else f"{equity.samples:,} samples, ±{100 * equity.margin_of_error:.2f}"
    )
    lines.append(
        f"Showdown equity vs the assumed range: {equity.equity_pct:.2f}% "
        f"({provenance})."
    )

    if analysis.facing_bet:
        lines.append(
            f"Price: {analysis.to_call:g} to call into {analysis.total_pot:g}, "
            f"so {analysis.odds.required_equity_pct:.1f}% equity breaks even."
        )

        ev = analysis.ev_call_vs_fold
        better = "beats" if ev > 0 else "loses to"
        # "terminal decision tree", not "exact": the tree ends here, but the
        # figure is only as exact as the equity feeding it.
        exactness = (
            "terminal decision tree — no betting follows"
            if analysis.ev_is_terminal
            else "an upper bound; it assumes the hand checks down and hero "
                 "realises all of its equity"
        )
        comparison = (
            f"Calling {better} folding by {abs(ev):.2f} chips ({exactness})."
        )

        if action_type is ActionType.RAISE:
            lines.append(
                f"You raised. The only comparison computed here is calling "
                f"against folding, which does not evaluate your raise. "
                f"For reference: {comparison[0].lower()}{comparison[1:]}"
            )
        elif action_type is ActionType.FOLD:
            kept = "gave up" if ev > 0 else "avoided a loss of"
            lines.append(
                f"You folded, so you {kept} {abs(ev):.2f} chips relative to "
                f"calling ({exactness})."
            )
        else:
            lines.append(f"You called. {comparison}")

        if analysis.implied_odds_needed:
            lines.append(
                f"To break even on implied odds hero must win a further "
                f"{analysis.implied_odds_needed:.1f} chips on later streets, "
                f"against {analysis.hero_stack_behind:g} behind."
            )
    else:
        lines.append("Hero is not facing a bet; no price to compare against.")

    return lines


def unresolved_lines(analysis: SpotAnalysis) -> list[str]:
    """The OPEN facts, verbatim — the stated edge of what was computed."""

    return [
        f"{fact.key}: {fact.rendered_value}"
        for fact in analysis.facts
        if fact.category is FactCategory.OPEN
    ]


def build_feedback(
    analysis: SpotAnalysis,
    action_taken: str,
    *,
    action_type: ActionType | None = None,
    interpretation: str | None = None,
    grounding: GroundingReport | None = None,
    repair_attempted: bool = False,
    fell_back: bool = False,
    warnings: list[str] | None = None,
) -> Feedback:
    return Feedback(
        action_taken=action_taken,
        verified=verified_lines(analysis, action_taken, action_type),
        unresolved=unresolved_lines(analysis),
        equity_exact=analysis.equity.exact,
        interpretation=interpretation,
        grounding=grounding,
        repair_attempted=repair_attempted,
        fell_back=fell_back,
        warnings=list(warnings or []),
    )
