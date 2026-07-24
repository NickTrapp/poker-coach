"""Recorded hands, and replaying them back into `HandState`.

A `HandHistory` is the durable, serialisable form of a hand — the thing you
save to disk, load from `examples/hands/`, or receive from a tracker. Replaying
it re-derives every intermediate `HandState` through the same state machine
that governs live play, so a reviewed hand and a played hand cannot drift apart.

Blinds and antes are *not* recorded as actions: they are posted automatically by
:meth:`HandHistory.initial_state` from each seat's position, so a history file
contains only the decisions a player actually made.
"""

from __future__ import annotations

from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .action import Action
from .cards import Card
from .enums import Position, Street
from .state import HandState, PlayerState

__all__ = ["SeatRecord", "StreetRecord", "DecisionPoint", "HandHistory"]


class SeatRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    position: Position
    starting_stack: float = Field(gt=0.0)
    hole_cards: tuple[Card, Card] | None = None


class StreetRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    street: Street
    cards: list[Card] = Field(
        default_factory=list,
        description="Cards dealt at the *start* of this street (empty preflop).",
    )
    actions: list[Action] = Field(default_factory=list)


class DecisionPoint(BaseModel):
    """One action, plus the state as it stood immediately before it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: HandState
    action: Action
    is_hero: bool

    @property
    def street(self) -> Street:
        return self.state.street

    def __str__(self) -> str:  # pragma: no cover - display helper
        marker = "*" if self.is_hero else " "
        return f"{marker} [{self.street.value}] {self.action}"


class HandHistory(BaseModel):
    """A complete recorded hand."""

    model_config = ConfigDict(validate_assignment=True)

    seats: list[SeatRecord]
    hero: str
    small_blind: float = Field(default=0.5, gt=0.0)
    big_blind: float = Field(default=1.0, gt=0.0)
    ante: float = Field(default=0.0, ge=0.0)
    streets: list[StreetRecord] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "HandHistory":
        if not self.seats:
            raise ValueError("a hand needs at least one seat")

        names = {seat.name for seat in self.seats}
        if len(names) != len(self.seats):
            raise ValueError("duplicate seat names")
        if self.hero not in names:
            raise ValueError(f"hero {self.hero!r} is not seated in this hand")

        for record in self.streets:
            for action in record.actions:
                if action.actor not in names:
                    raise ValueError(
                        f"action by unknown player {action.actor!r} "
                        f"on the {record.street.value}"
                    )

        if self.streets:
            order = list(Street)
            expected = order.index(Street.PREFLOP)
            for record in self.streets:
                if order.index(record.street) != expected:
                    raise ValueError(
                        f"streets must run in order from preflop; got "
                        f"{record.street.value} where "
                        f"{order[expected].value} was expected"
                    )
                expected += 1

            if self.streets[0].cards:
                raise ValueError("no cards are dealt at the start of preflop")

        return self

    # ------------------------------------------------------------------ replay

    def initial_state(self) -> HandState:
        """The state after blinds and antes are posted, before any decision."""

        players: list[PlayerState] = []
        dead_money = 0.0

        for seat in self.seats:
            stack = seat.starting_stack
            posted_ante = 0.0

            if self.ante:
                posted_ante = min(self.ante, stack)
                stack -= posted_ante
                dead_money += posted_ante

            if seat.position is Position.SB:
                blind = min(self.small_blind, stack)
            elif seat.position is Position.BB:
                blind = min(self.big_blind, stack)
            else:
                blind = 0.0
            stack -= blind

            players.append(
                PlayerState(
                    name=seat.name,
                    position=seat.position,
                    stack=stack,
                    hole_cards=seat.hole_cards,
                    # Antes are dead money, not a live bet, so they count toward
                    # the total contributed but not toward the amount to call.
                    committed_this_street=blind,
                    committed_total=blind + posted_ante,
                    is_all_in=stack <= 0,
                    is_hero=seat.name == self.hero,
                )
            )

        return HandState(
            players=players,
            board=[],
            street=Street.PREFLOP,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            ante=self.ante,
            pot=dead_money,
        )

    def replay(self) -> Iterator[DecisionPoint]:
        """Walk the hand, yielding each action with the state preceding it."""

        state = self.initial_state()

        for index, record in enumerate(self.streets):
            if index > 0:
                state = state.advance_street(record.cards)

            for action in record.actions:
                # The recorded street is authoritative; stamp it so a decision
                # point is self-describing even when read out of context.
                stamped = action.model_copy(update={"street": record.street})
                yield DecisionPoint(
                    state=state,
                    action=stamped,
                    is_hero=action.actor == self.hero,
                )
                state = state.apply(stamped)

    def final_state(self) -> HandState:
        """Replay to the end and return the resulting state."""

        state = self.initial_state()
        for index, record in enumerate(self.streets):
            if index > 0:
                state = state.advance_street(record.cards)
            for action in record.actions:
                state = state.apply(
                    action.model_copy(update={"street": record.street})
                )
        return state

    def hero_decisions(self) -> list[DecisionPoint]:
        """Only the points where the hero had a decision to make."""

        return [point for point in self.replay() if point.is_hero]

    @property
    def board(self) -> list[Card]:
        return [card for record in self.streets for card in record.cards]

    # -------------------------------------------------------------- load/save

    @classmethod
    def from_json_file(cls, path: str) -> "HandHistory":
        from pathlib import Path

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_json_file(self, path: str, *, indent: int = 2) -> None:
        from pathlib import Path

        Path(path).write_text(
            self.model_dump_json(indent=indent, exclude_none=True) + "\n",
            encoding="utf-8",
        )

    def __str__(self) -> str:  # pragma: no cover - display helper
        return "\n".join(str(point) for point in self.replay())
