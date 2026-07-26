"""Play hands against an archetype and get a critique of each decision.

The hand itself runs through `players.table.play_hand` unchanged. A human is
just another `Player` — something with a name and an `act(state)` — so the
rules engine, action order, side pots and strict legality all apply exactly as
they do to a simulated opponent. Nothing here re-implements a betting round,
which is the mistake that produced the multiway ordering bug.

What the range assumption is, and is not
----------------------------------------
Choosing a "station" tells the *simulator* how the opponent behaves. It does
not tell the *coach* what that opponent holds having taken the action it just
took — deriving that needs a strategy model this package does not have. So the
range is configuration: the caller states it, the facts label it, and the
critique says where it came from. Anything else would be inventing a read and
presenting it as a fact.
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
from ..players.table import HandResult
from .critique import CritiqueResult, critique
from .log import DecisionRecord

__all__ = ["PracticeConfig", "HeroTurn", "PracticeSession", "CallbackPlayer"]

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

    def range_for(self) -> RangeAssumption:
        notation = self.villain_range or DEFAULT_RANGES.get(
            self.opponent_style, "random"
        )
        return RangeAssumption(notation, self.range_conditioning)

    def describe(self) -> str:
        assumption = self.range_for()
        return "\n".join(
            [
                f"Opponent          : {self.opponent_style}",
                f"Range assumption  : {assumption.notation}",
                f"Conditioning      : {assumption.conditioning.value}",
                "Source            : fixed practice configuration, "
                "not inferred from play",
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

        def decide(state: HandState, legal: list[LegalAction]) -> Action:
            nonlocal decision_index
            decision_index += 1

            analysis = analyze(
                state,
                hero=self.config.hero_name,
                villain_range=self.config.range_for(),
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
            Seat(
                make_player(
                    self.config.opponent_name, self.config.opponent_style,
                    rng=hand_rng, iterations=self.config.opponent_iterations,
                ),
                villain_seat, self.config.stack,
            ),
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
