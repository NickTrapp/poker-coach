"""Betting-round rules: action order, minimum raises, and round completion.

`HandState.apply` previously validated only stack limits, checks facing bets,
call amounts, and that a raise exceeded the current bet. An external review
found four things it accepted: a player acting twice in a row, an action tagged
with the wrong street, a raise to 3.5 over a bet of 3, and advancing the street
with an unmatched bet outstanding. Each replayed silently.

This module is the single source of truth for what is legal. It is derived
state — nothing here is stored on `HandState`, so it cannot fall out of sync.

Two tiers of enforcement, deliberately
--------------------------------------
Some rules follow from the state alone and are always enforced: the street tag,
the minimum raise, and whether a betting round has actually closed.

Turn order and raise-reopening need the street's *action history* to evaluate —
who has acted, and who has acted since the last full raise. A `HandState` built
directly in a test or a fixture has no history and no way to acquire one, so
those rules are enforced only under ``strict=True``, which replay and the table
runner both pass. The split is explicit rather than inferred, because a
guarantee that silently switches itself off is worse than one with a stated
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .action import Action
from .enums import ActionType, Position, Street

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .state import HandState

__all__ = [
    "LegalAction",
    "BettingRound",
    "action_order",
    "RING",
]

#: Physical seating order, clockwise from the small blind.
RING: tuple[Position, ...] = (
    Position.SB,
    Position.BB,
    Position.UTG,
    Position.UTG1,
    Position.MP,
    Position.LJ,
    Position.HJ,
    Position.CO,
    Position.BTN,
)

_EPS = 1e-9


def action_order(street: Street, positions: set[Position]) -> list[Position]:
    """Seats in the order they act on ``street``.

    Preflop opens to the left of the big blind and the blinds close. Later
    streets open at the small blind and the button closes. Heads-up inverts the
    postflop order, because there the small blind *is* the button.
    """

    ring = [p for p in RING if p in positions]
    if not ring:
        raise ValueError("no known positions at the table")

    if street is Street.PREFLOP:
        if Position.BB not in positions:
            raise ValueError("preflop action order needs a big blind")
        pivot = ring.index(Position.BB)
        return ring[pivot + 1 :] + ring[: pivot + 1]

    # Postflop the button closes the action, so rotate the ring to end on it.
    if Position.BTN in positions:
        button = Position.BTN
    elif len(ring) == 2:
        # Heads-up without an explicit button seat: the small blind holds it,
        # so it acts last from the flop on.
        button = ring[0]
    else:
        # No button seated at a fuller table — the latest seat stands in.
        button = ring[-1]
    pivot = ring.index(button)
    return ring[pivot + 1 :] + ring[: pivot + 1]


@dataclass(frozen=True, slots=True)
class LegalAction:
    """One action a player may legally take, with its permitted sizing.

    ``min_amount`` and ``max_amount`` are "to" amounts, matching
    :attr:`poker_coach.domain.action.Action.amount`. They are equal for folds,
    checks and calls, which have no sizing freedom.
    """

    type: ActionType
    min_amount: float = 0.0
    max_amount: float = 0.0

    def permits(self, amount: float) -> bool:
        return self.min_amount - _EPS <= amount <= self.max_amount + _EPS

    def __str__(self) -> str:  # pragma: no cover - display helper
        if self.min_amount == self.max_amount:
            return self.type.value if not self.min_amount else (
                f"{self.type.value} to {self.min_amount:g}"
            )
        return f"{self.type.value} to {self.min_amount:g}-{self.max_amount:g}"


@dataclass(frozen=True, slots=True)
class BettingRound:
    """The current street's betting state, derived from a `HandState`."""

    street: Street
    current_bet: float
    big_blind: float
    #: Largest full raise *increment* seen this street; the floor for the next
    #: legal raise. Starts at the big blind, which is also the minimum bet.
    last_full_raise: float
    #: Street commitment each player had when they last acted. Absent means
    #: they have not acted on this street.
    committed_when_acted: dict[str, float] = field(default_factory=dict)
    #: Names in the order they act, filtered to those still able to act.
    order: tuple[str, ...] = ()
    #: True when the history was available to derive turn-dependent facts.
    has_history: bool = False

    # ------------------------------------------------------------ derivation

    @classmethod
    def from_state(cls, state: "HandState") -> "BettingRound":
        street_actions = [a for a in state.actions if a.street is state.street]

        positions = {p.position for p in state.players}
        by_position = {p.position: p.name for p in state.players}
        try:
            ordered = [by_position[p] for p in action_order(state.street, positions)]
        except ValueError:
            # Unconventional seating (fixtures often use BTN/BB heads-up).
            # Turn order is simply unavailable; other rules still apply.
            ordered = [p.name for p in state.players]

        committed_when_acted: dict[str, float] = {}
        last_full_raise = state.big_blind
        previous_level = 0.0 if state.street is not Street.PREFLOP else state.big_blind

        for action in street_actions:
            if action.type.is_aggressive:
                increment = action.amount - previous_level
                if increment >= last_full_raise - _EPS:
                    last_full_raise = increment
                previous_level = action.amount
            committed_when_acted[action.actor] = action.amount

        return cls(
            street=state.street,
            current_bet=state.current_bet,
            big_blind=state.big_blind,
            last_full_raise=last_full_raise,
            committed_when_acted=committed_when_acted,
            order=tuple(name for name in ordered if state.player(name).can_act),
            has_history=bool(street_actions),
        )

    # ------------------------------------------------------------- accessors

    def min_raise_to(self, committed: float, stack: float) -> float:
        """Smallest legal raise-to amount, capped by an all-in."""

        wanted = self.current_bet + self.last_full_raise
        return min(wanted, committed + stack)

    def raising_is_reopened_for(self, name: str, committed: float) -> bool:
        """Whether ``name`` may re-raise, or is limited to calling.

        An all-in that is *less* than a full raise does not reopen the betting
        for players who have already acted against the previous level.
        """

        if name not in self.committed_when_acted:
            return True  # has not acted this street
        since = self.current_bet - self.committed_when_acted[name]
        return since >= self.last_full_raise - _EPS

    def next_actor(self, state: "HandState") -> str | None:
        """Whose turn it is, or None if the round is closed or unknowable."""

        if not self.order:
            return None
        for name in self.order:
            player = state.player(name)
            owes = self.current_bet - player.committed_this_street > _EPS
            acted = name in self.committed_when_acted
            if owes or not acted:
                return name
        return None

    def is_complete(self, state: "HandState") -> bool:
        """True when no player who can act still owes chips or has yet to act."""

        return self.next_actor(state) is None

    def unmatched(self, state: "HandState") -> list[str]:
        """Players who can still act and have not matched the current bet."""

        return [
            p.name
            for p in state.players
            if p.can_act
            and self.current_bet - p.committed_this_street > _EPS
        ]

    def legal_actions(self, state: "HandState", name: str) -> list[LegalAction]:
        """Every action ``name`` may legally take right now."""

        player = state.player(name)
        if not player.can_act:
            return []

        owed = min(
            self.current_bet - player.committed_this_street, player.stack
        )
        ceiling = player.committed_this_street + player.stack
        out: list[LegalAction] = []

        if owed > _EPS:
            out.append(LegalAction(ActionType.FOLD))
            out.append(
                LegalAction(
                    ActionType.CALL,
                    player.committed_this_street + owed,
                    player.committed_this_street + owed,
                )
            )
        else:
            out.append(LegalAction(ActionType.CHECK))

        # A bet opens an unbet pot; a raise answers a live one.
        if self.current_bet <= _EPS:
            if player.stack > _EPS:
                out.append(
                    LegalAction(
                        ActionType.BET, min(self.big_blind, ceiling), ceiling
                    )
                )
        elif ceiling > self.current_bet + _EPS and self.raising_is_reopened_for(
            name, player.committed_this_street
        ):
            out.append(
                LegalAction(
                    ActionType.RAISE,
                    self.min_raise_to(player.committed_this_street, player.stack),
                    ceiling,
                )
            )

        return out

    # ------------------------------------------------------------ validation

    def validate(
        self, state: "HandState", action: Action, *, strict: bool
    ) -> None:
        """Raise if ``action`` is illegal. See the module docstring for tiers."""

        if action.street is not state.street:
            raise ValueError(
                f"action is tagged {action.street.value} but the hand is on "
                f"{state.street.value}"
            )

        player = state.player(action.actor)

        if strict and self.order:
            expected = self.next_actor(state)
            if expected is None:
                raise ValueError(
                    f"the {state.street.value} betting round is already complete; "
                    f"{action.actor} cannot act"
                )
            if expected != action.actor:
                raise ValueError(
                    f"it is {expected}'s turn to act, not {action.actor}'s"
                )

        if action.type is not ActionType.RAISE:
            return

        ceiling = player.committed_this_street + player.stack
        is_all_in = abs(action.amount - ceiling) <= _EPS
        if is_all_in:
            return  # a player may always move all-in

        minimum = self.min_raise_to(player.committed_this_street, player.stack)
        if action.amount < minimum - _EPS:
            raise ValueError(
                f"raise to {action.amount:g} is below the minimum of "
                f"{minimum:g} (current bet {self.current_bet:g}, last full "
                f"raise {self.last_full_raise:g})"
            )

        if strict and not self.raising_is_reopened_for(
            action.actor, player.committed_this_street
        ):
            raise ValueError(
                f"{action.actor} may not re-raise: the action was not reopened "
                "by a full raise since they last acted"
            )
