"""Betting actions.

Amount convention
-----------------
``Action.amount`` is always a **"to" amount**: the total number of chips the
actor will have committed *on the current street* once the action resolves.
This matches how poker is spoken ("raise to 30") and means a single field
covers blinds, calls, bets and raises without per-type special cases. ``FOLD``
and ``CHECK`` carry an amount of ``0``.

The chips actually leaving a stack are derived at apply time as
``amount - player.committed_this_street`` (see
:meth:`poker_coach.domain.state.HandState.apply`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ActionType, Street

__all__ = ["Action"]


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    actor: str = Field(description="Name of the player taking the action.")
    type: ActionType
    street: Street = Street.PREFLOP
    amount: float = Field(
        default=0.0,
        ge=0.0,
        description="Total street commitment after the action ('raise to' semantics).",
    )

    @model_validator(mode="after")
    def _check_amount(self) -> "Action":
        if self.type in (ActionType.FOLD, ActionType.CHECK) and self.amount != 0:
            raise ValueError(f"{self.type.value} cannot carry an amount")
        if self.type.puts_money_in and self.amount <= 0:
            raise ValueError(f"{self.type.value} requires a positive amount")
        return self

    def __str__(self) -> str:
        if self.amount:
            return f"{self.actor} {self.type.value} {self.amount:g}"
        return f"{self.actor} {self.type.value}"
