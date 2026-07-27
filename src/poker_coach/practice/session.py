"""Play hands against an archetype and get a critique of each decision.

The hand itself runs through `players.table.play_hand` unchanged. A human is
just another `Player` — something with a name and an `act(state)` — so the
rules engine, action order, side pots and strict legality all apply exactly as
they do to a simulated opponent. Nothing here re-implements a betting round,
which is the mistake that produced the multiway ordering bug.

What the range assumption is, and is not
----------------------------------------
Choosing a "station" tells the *simulator* how the opponent behaves. Turning
that into a read — what it holds *given* the action it just took — normally
needs a strategy model this package does not have, so the range was pure
configuration: the caller stated it, the facts labelled it, and nothing was
inferred from play.

Here that model does exist. The opponent is not a person, it is an explicit
policy, so `players.conditioning` can ask it directly which holdings take the
observed line and narrow the range accordingly. Three things about that:

- **It is a fact about the policy, not about a person.** A real opponent is not
  this station. The facts say so, in the provenance line, every time.
- **The prior is "random", not the archetype's assumed range.** The dealer hands
  the opponent any two cards, and the archetype's own preflop range gate is part
  of the policy — so conditioning *recovers* that range instead of assuming it,
  and cannot disagree with it. (It once did: the station's configured range and
  its `open_range` differed by ``96s+``.)
- **Sampling still bounds it.** The grid behind the likelihood is Monte Carlo,
  so a combo sitting within sampling error of a threshold can be kept or
  dropped either way — including, occasionally, the hand the opponent actually
  holds.

Set ``condition_on_action=False`` for the old behaviour: a fixed configured
range, labelled pre-action.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from ..coaching.analysis import (
    RangeAssumption,
    RangeConditioning,
    SpotAnalysis,
    analyze,
)
from ..domain.action import Action
from ..domain.betting import LegalAction
from ..domain.enums import Position
from ..domain.state import HandState
from ..models.base import LanguageModel
from ..players import STYLES, Seat, make_player, play_hand
from ..players.base import RuleBasedPlayer
from ..players.conditioning import ConditionedRange, condition_range
from ..players.table import HandResult
from .critique import CritiqueResult, critique
from .log import DecisionRecord

__all__ = [
    "PracticeConfig",
    "HeroTurn",
    "PracticeSession",
    "CallbackPlayer",
    "ObservedOpponent",
]

#: What the coach assumes the opponent holds, per archetype. Configuration,
#: not inference — see the module docstring.
DEFAULT_RANGES: dict[str, str] = {
    "nit": "88+, ATs+, KQs, AQo+",
    "tag": "22+, A2s+, K9s+, QTs+, JTs, ATo+, KQo",
    "lag": "22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, A7o+, K9o+, QTo+",
    "station": "22+, A2s+, K2s+, Q6s+, J7s+, T7s+, A2o+, K8o+, Q9o+, JTo",
    "maniac": "random",
}


@dataclass
class CallbackPlayer:
    """A `Player` whose action comes from a callable — a person, or a script."""

    name: str
    decide: Callable[[HandState, list[LegalAction]], Action]

    def act(self, state: HandState) -> Action:
        return self.decide(state, state.legal_actions(self.name))


@dataclass
class ObservedOpponent:
    """Wraps an opponent so the session sees exactly what its policy saw.

    Conditioning needs the state *before* each action. Reconstructing it by
    replaying the history would be a second implementation of the runner, and
    the worst bug in this project came from two copies of one rule agreeing
    with each other. The runner already hands the policy that state; this keeps
    a copy of it as it goes past.
    """

    player: RuleBasedPlayer
    seen: list[tuple[HandState, Action]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.player.name

    def act(self, state: HandState) -> Action:
        action = self.player.act(state)
        self.seen.append((state.model_copy(deep=True), action))
        return action

    def forget(self) -> None:
        """Start a new hand. Reads never carry across hands."""

        self.seen.clear()


@dataclass
class _RangeTracker:
    """Folds each observed opponent action into a running posterior.

    Incremental on purpose: every action is conditioned exactly once, and each
    step starts from what the last one left, so the work shrinks as the hand
    goes on instead of being redone from the top at every hero decision.
    """

    config: PracticeConfig
    opponent: ObservedOpponent
    rng: random.Random
    posterior: ConditionedRange | None = None
    consumed: int = 0
    #: Set when conditioning had to stop, and why. Never silently ignored.
    stopped: str | None = None

    def update(self) -> None:
        for state, action in self.opponent.seen[self.consumed:]:
            self.consumed += 1
            if self.stopped is not None:
                continue
            narrowed = condition_range(
                self.posterior if self.posterior is not None
                else self.config.conditioning_prior,
                self.opponent.player,
                state,
                action,
                iterations=self.config.conditioning_iterations,
                rng=self.rng,
            )
            if not narrowed.combos:
                # No holding in what is left takes this action. Real, and
                # expected near a threshold: the opponent decided from its own
                # equity sample and the grid drew another. Keep the last good
                # posterior and say conditioning stopped — an empty range has
                # no equity, and silently reverting would hide the disagreement.
                self.stopped = (
                    f"conditioning stopped at the {state.street.value} "
                    f"{action.type.value}: no holding left in the range takes "
                    "that action"
                )
                continue
            self.posterior = narrowed

    def assumption(self) -> RangeAssumption:
        self.update()
        posterior = self.posterior
        if posterior is None or not posterior.steps:
            # Either the opponent has not acted — in which case a pre-action
            # range is exactly right — or the very first action emptied the
            # range. Falling back to the prior is correct in both cases; saying
            # which one it was is what stops the second from looking like the
            # first.
            return RangeAssumption(
                notation=self.config.conditioning_prior,
                conditioning=RangeConditioning.PRE_ACTION,
                provenance=(
                    f"no read: {self.stopped}" if self.stopped is not None
                    else "the opponent has not acted yet, so nothing has "
                         "been inferred from play"
                ),
            )

        style = self.config.opponent_style
        provenance = (
            f"derived by asking the simulated {style} which holdings take this "
            "line; true of that policy, not of a person, and the equity behind "
            "it is sampled"
        )
        if self.stopped is not None:
            provenance += f" — {self.stopped}"

        return RangeAssumption(
            notation=posterior.notation(),
            conditioning=RangeConditioning.ACTION_CONDITIONED,
            # No digits: this lands in prompt prose, and the counts belong in
            # the fact below where they are grounded values.
            label=f"the hands a {style} takes this line with "
                  f"({posterior.line()})",
            observed_line=posterior.line(),
            # The support goes in `notation` for display and blockers; the
            # probabilities go here, and this is what every figure is computed
            # against. Handing over only the support would keep the "∝" of
            # `P(H|A,S) ∝ P(A|H,S)·P(H|S)` and discard the distribution.
            weighted=posterior.to_weighted(),
            provenance=provenance,
            combos=posterior.survivors,
            combos_before=posterior.steps[0].considered,
            # The likelihood behind every step came off a Monte Carlo grid.
            sampled=True,
        )


@dataclass(frozen=True, slots=True)
class HeroTurn:
    """Everything the student is shown before choosing."""

    state: HandState
    legal: list[LegalAction]
    analysis: SpotAnalysis


@dataclass
class PracticeConfig:
    """How a practice session is set up. Every field is a stated assumption."""

    opponent_style: str = "station"
    hero_name: str = "you"
    opponent_name: str = "villain"
    stack: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0
    #: None means "use the archetype's configured range".
    villain_range: str | None = None
    range_conditioning: RangeConditioning = RangeConditioning.PRE_ACTION
    iterations: int = 6000
    opponent_iterations: int = 400
    #: Narrow the opponent's range using the actions it actually took, by
    #: asking its own policy which holdings take them. See the module docstring.
    condition_on_action: bool = True
    #: Iterations for the per-combo grid behind the likelihood. Deliberately
    #: below `iterations`: this ranks combos against a threshold rather than
    #: reporting a figure, and it runs once per opponent action.
    conditioning_iterations: int = 300

    @property
    def conditioning_prior(self) -> str:
        """Where conditioning starts from.

        "random" unless the caller named a range, because the dealer hands the
        opponent any two cards. Starting from the archetype's assumed range
        would assert preflop a filter the policy applies for itself, and could
        contradict it.
        """

        return self.villain_range or "random"

    def range_for(self) -> RangeAssumption:
        """The unconditioned range: what the coach assumes absent any read."""

        notation = self.villain_range or DEFAULT_RANGES.get(
            self.opponent_style, "random"
        )
        return RangeAssumption(
            notation,
            self.range_conditioning,
            provenance="fixed practice configuration, not inferred from play",
        )

    def describe(self) -> str:
        if self.condition_on_action:
            return "\n".join(
                [
                    f"Opponent          : {self.opponent_style}",
                    f"Range assumption  : starts at "
                    f"{self.conditioning_prior}, narrowed by each action",
                    "Conditioning      : "
                    f"{RangeConditioning.ACTION_CONDITIONED.value}",
                    f"Source            : the simulated {self.opponent_style}'s "
                    "own policy — true of it, not of a person",
                    f"Stacks            : {self.stack:g} each, blinds "
                    f"{self.small_blind:g}/{self.big_blind:g}",
                ]
            )
        assumption = self.range_for()
        return "\n".join(
            [
                f"Opponent          : {self.opponent_style}",
                f"Range assumption  : {assumption.notation}",
                f"Conditioning      : {assumption.conditioning.value}",
                f"Source            : {assumption.provenance}",
                f"Stacks            : {self.stack:g} each, blinds "
                f"{self.small_blind:g}/{self.big_blind:g}",
            ]
        )


@dataclass
class PracticeSession:
    """Deals hands, routes hero decisions to a callback, and critiques them."""

    config: PracticeConfig = field(default_factory=PracticeConfig)
    model: LanguageModel | None = None
    rng: random.Random = field(default_factory=random.Random)
    #: Called when it is the student's turn; returns their chosen action.
    on_turn: Callable[[HeroTurn], Action] | None = None
    #: Called with each critique, plus the turn it refers to.
    on_feedback: Callable[[HeroTurn, CritiqueResult], None] | None = None
    #: Called with a fully-populated record for every hero decision.
    on_record: Callable[[DecisionRecord], None] | None = None
    coach: bool = True

    hands_played: int = 0
    records: list[DecisionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.config.opponent_style not in STYLES:
            known = ", ".join(sorted(STYLES))
            raise ValueError(
                f"unknown opponent {self.config.opponent_style!r}; "
                f"choose one of: {known}"
            )

    # ------------------------------------------------------------------ play

    def play_hand(self, seed: int | None = None) -> HandResult:
        """Deal and play one hand, pausing at every hero decision."""

        hand_seed = self.rng.randrange(2**31) if seed is None else seed
        hand_rng = random.Random(hand_seed)
        self.hands_played += 1
        decision_index = 0

        opponent = ObservedOpponent(
            make_player(
                self.config.opponent_name, self.config.opponent_style,
                rng=hand_rng, iterations=self.config.opponent_iterations,
            )
        )
        # A fresh tracker per hand: a read is about this hand's actions only.
        tracker = _RangeTracker(self.config, opponent, hand_rng)

        def decide(state: HandState, legal: list[LegalAction]) -> Action:
            nonlocal decision_index
            decision_index += 1

            assumption = (
                tracker.assumption() if self.config.condition_on_action
                else self.config.range_for()
            )
            analysis = analyze(
                state,
                hero=self.config.hero_name,
                villain_range=assumption,
                iterations=self.config.iterations,
                rng=hand_rng,
            )
            turn = HeroTurn(state=state, legal=legal, analysis=analysis)

            if self.on_turn is None:
                raise ValueError("PracticeSession needs an on_turn callback")
            action = self.on_turn(turn)

            result = critique(
                analysis,
                _describe(action),
                opponent=self.config.opponent_style,
                action_type=action.type,
                model=self.model if self.coach else None,
            )

            record = DecisionRecord.build(
                hand_number=self.hands_played,
                decision_number=decision_index,
                seed=hand_seed,
                config=self.config,
                state=state,
                legal=legal,
                action=action,
                analysis=analysis,
                critique=result,
            )
            self.records.append(record)
            if self.on_record is not None:
                self.on_record(record)
            if self.on_feedback is not None:
                self.on_feedback(turn, result)

            return action

        # Alternate the button so the student sees both seats.
        hero_is_sb = self.hands_played % 2 == 1
        hero_seat = Position.SB if hero_is_sb else Position.BB
        villain_seat = Position.BB if hero_is_sb else Position.SB

        seats = [
            Seat(CallbackPlayer(self.config.hero_name, decide),
                 hero_seat, self.config.stack),
            Seat(opponent, villain_seat, self.config.stack),
        ]

        return play_hand(
            seats,
            rng=hand_rng,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            hero=self.config.hero_name,
        )


def _describe(action: Action) -> str:
    if action.amount:
        return f"{action.type.value} to {action.amount:g}"
    return action.type.value
