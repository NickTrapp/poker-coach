"""Enumerations shared across the domain model."""

from __future__ import annotations

from enum import Enum

__all__ = ["Street", "Position", "ActionType", "TABLE_POSITIONS"]


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"

    @property
    def board_size(self) -> int:
        """Number of board cards visible on this street."""

        return _STREET_BOARD_SIZE[self]

    def next(self) -> "Street":
        order = list(Street)
        idx = order.index(self)
        if idx == len(order) - 1:
            return self
        return order[idx + 1]


_STREET_BOARD_SIZE: dict[Street, int] = {
    Street.PREFLOP: 0,
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
    Street.SHOWDOWN: 5,
}


class Position(str, Enum):
    """Seat positions, ordered from earliest to latest preflop action.

    The full-ring names are a superset of the 6-max ones; a 6-max table uses
    ``UTG, HJ, CO, BTN, SB, BB``.
    """

    UTG = "UTG"
    UTG1 = "UTG+1"
    MP = "MP"
    LJ = "LJ"
    HJ = "HJ"
    CO = "CO"
    BTN = "BTN"
    SB = "SB"
    BB = "BB"

    @property
    def is_blind(self) -> bool:
        return self in (Position.SB, Position.BB)

    @property
    def is_in_position_postflop(self) -> bool:
        """True for seats that act last postflop (button, absent blinds)."""

        return self is Position.BTN


#: Preflop action order (blinds act last preflop, first postflop).
TABLE_POSITIONS: tuple[Position, ...] = (
    Position.UTG,
    Position.UTG1,
    Position.MP,
    Position.LJ,
    Position.HJ,
    Position.CO,
    Position.BTN,
    Position.SB,
    Position.BB,
)


class ActionType(str, Enum):
    POST_BLIND = "post_blind"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"

    @property
    def is_aggressive(self) -> bool:
        return self in (ActionType.BET, ActionType.RAISE)

    @property
    def is_voluntary(self) -> bool:
        return self is not ActionType.POST_BLIND

    @property
    def puts_money_in(self) -> bool:
        return self in (
            ActionType.POST_BLIND,
            ActionType.CALL,
            ActionType.BET,
            ActionType.RAISE,
        )
