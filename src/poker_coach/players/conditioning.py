"""Update an opponent's range from the action it just took.

    P(H | A, S)  ∝  P(A | H, S) · P(H | S)

The likelihood is the unusual part. Normally `P(A | H, S)` is the thing nobody
has, and asking a model to invent it is guesswork dressed as inference. Here the
opponent *is* an explicit policy, so the likelihood can be **computed**: give
the policy a hypothetical combo, ask what it would do, and read off how often it
does that. The archetypes stop being merely opponents and become inspectable
generative models of a range.

Three honest limits, all stated rather than smoothed over:

**A posterior derived from a policy is a fact about the policy.** It says what
*this* configured station would hold, not what a person would. That is still a
large improvement on a fixed pre-action range, which is wrong the moment the
opponent does anything.

**The likelihood is exact where the policy is deterministic** — folding,
calling, raising and value betting all are. The single stochastic branch is
bluffing an unbet pot, whose probability is a declared parameter, so it is read
off rather than sampled.

**The equity read behind the policy is sampled**, so combos sitting within
sampling error of a threshold could plausibly have gone either way and this
resolves them one way. Raising `iterations` narrows that band; nothing removes
it. Combos near the boundary are the least trustworthy part of the posterior.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..calculations.equity import Combo, equity_grid
from ..calculations.ranges import Range
from ..domain.action import Action
from ..domain.cards import Card, cards_to_str
from ..domain.enums import ActionType, Street
from ..domain.state import HandState
from .base import RuleBasedPlayer

__all__ = ["ConditionedRange", "ConditioningStep", "condition_range"]

#: Iterations for the per-combo equity grid. Lower than a coaching-grade equity
#: figure on purpose: this ranks combos against a threshold rather than
#: reporting a number, and the whole grid is recomputed on every action.
DEFAULT_GRID_ITERATIONS = 300


@dataclass(frozen=True, slots=True)
class ConditioningStep:
    """One observed action, and what conditioning on it left standing."""

    observed: ActionType
    street: Street
    considered: int
    survivors: int

    def describe(self) -> str:
        return (
            f"{self.observed.value} ({self.street.value}): "
            f"{self.survivors} of {self.considered} combos"
        )

    def phrase(self) -> str:
        """The action alone, with no counts — safe to embed in prose.

        Prompt prose is checked for numbers it was never given, so anything
        that ends up in a range *description* has to be countless. The counts
        live in a fact of their own, where they are grounded values.
        """

        where = (
            "preflop" if self.street is Street.PREFLOP
            else f"on the {self.street.value}"
        )
        return f"{self.observed.value} {where}"


@dataclass(frozen=True, slots=True)
class ConditionedRange:
    """A posterior over an opponent's holdings, with its derivation attached."""

    prior_notation: str
    weights: dict[Combo, float]
    observed: ActionType
    street: Street
    #: Combos in the prior that were possible at all, before conditioning.
    considered: int
    #: Every action folded in so far, oldest first. The last entry is this one.
    steps: tuple[ConditioningStep, ...] = ()

    @property
    def combos(self) -> list[Combo]:
        """Combos the policy could have taken this action with."""

        return [combo for combo, weight in self.weights.items() if weight > 0]

    @property
    def survivors(self) -> int:
        return len(self.combos)

    @property
    def is_certain(self) -> bool:
        """True when every surviving combo takes the action every time.

        False means some survivors only sometimes act this way — a bluffing
        branch — and the flat range below understates how much of the posterior
        those hands should carry.
        """

        return all(weight >= 1.0 for weight in self.weights.values() if weight > 0)

    def notation(self) -> str:
        """The posterior as explicit combos, for handing to `analyze`."""

        return ", ".join(cards_to_str(combo) for combo in self.combos)

    def to_range(self) -> Range:
        return Range(self.notation())

    def describe(self) -> str:
        if not self.steps:
            return "no actions observed"
        chain = "; ".join(step.describe() for step in self.steps)
        line = f"conditioned on {chain}"
        if not self.is_certain:
            line += " — weighted equally, though some only bluff here sometimes"
        return line

    def line(self) -> str:
        """The observed line in words, with no counts. See `Step.phrase`."""

        return ", then ".join(step.phrase() for step in self.steps)


def condition_range(
    prior: Range | str | ConditionedRange,
    player: RuleBasedPlayer,
    state: HandState,
    observed: Action,
    *,
    iterations: int = DEFAULT_GRID_ITERATIONS,
    rng: random.Random | None = None,
) -> ConditionedRange:
    """Narrow ``prior`` to the combos that would take ``observed`` in ``state``.

    ``state`` must be the position *before* the action, since that is what the
    policy would have been looking at.

    Passing an earlier :class:`ConditionedRange` chains the inference, which is
    what a hand actually calls for — each action narrows what the last one left:

        P(H | A₁, A₂) ∝ P(A₂ | H, S₂) · P(A₁ | H, S₁) · P(H)

    so the weights multiply. Chaining through :meth:`to_range` instead would
    flatten them and quietly discard the fact that a bluffing branch holds some
    combos at less than full weight. It is also *cheaper* than re-deriving from
    the original prior, because each step starts from a smaller candidate set.
    """

    rng = rng or random.Random()

    dead: list[Card] = list(state.board)
    for other in state.players:
        # Hero's own cards block the opponent; the opponent's do not block
        # themselves, since those are exactly what is being inferred.
        if other.name != observed.actor and other.hole_cards:
            dead.extend(other.hole_cards)
    blocked = set(dead)

    if isinstance(prior, ConditionedRange):
        prior_notation = prior.prior_notation
        earlier = prior.steps
        prior_weights = prior.weights
        # The board has grown since the last step; drop what it now blocks.
        candidates = [c for c in prior.combos if not blocked & set(c)]
    else:
        prior_range = prior if isinstance(prior, Range) else Range(prior)
        prior_notation = str(prior_range)
        earlier = ()
        prior_weights = {}
        candidates = prior_range.combos(dead=tuple(dead))

    if not candidates:
        step = ConditioningStep(observed.type, state.street, 0, 0)
        return ConditionedRange(
            prior_notation=prior_notation, weights={},
            observed=observed.type, street=state.street, considered=0,
            steps=(*earlier, step),
        )

    facing_bet = state.amount_to_call(observed.actor) > 0
    assumed = (
        player.style.assumed_bettor_range
        if facing_bet
        else player.style.assumed_villain_range
    )
    grid = equity_grid(
        candidates, Range(assumed), state.board,
        iterations=iterations, rng=rng,
    )

    weights: dict[Combo, float] = {}
    for combo in candidates:
        # Ask the policy itself, with this combo in the seat.
        hypothetical = state.model_copy(deep=True)
        hypothetical.player(observed.actor).hole_cards = combo
        likelihood = player.action_probability(
            hypothetical, grid[combo], observed.type
        )
        weights[combo] = prior_weights.get(combo, 1.0) * likelihood

    survivors = sum(1 for w in weights.values() if w > 0)
    step = ConditioningStep(
        observed.type, state.street, len(candidates), survivors
    )
    return ConditionedRange(
        prior_notation=prior_notation,
        weights=weights,
        observed=observed.type,
        street=state.street,
        considered=len(candidates),
        steps=(*earlier, step),
    )
