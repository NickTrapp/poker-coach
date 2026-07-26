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


def _street_start_commitments(state: "HandState") -> dict[str, float]:
    """What each player had committed when this street began.

    Zero everywhere except preflop, where the blinds are already posted.
    Antes are dead money and live in ``state.pot``, not here.
    """

    if state.street is not Street.PREFLOP:
        return {p.name: 0.0 for p in state.players}

    start: dict[str, float] = {}
    for player in state.players:
        if player.position is Position.SB:
            start[player.name] = min(state.small_blind, player.committed_this_street)
        elif player.position is Position.BB:
            start[player.name] = min(state.big_blind, player.committed_this_street)
        else:
            start[player.name] = 0.0
    return start


def _infer_lone_bet(state: "HandState") -> tuple[str, float, float] | None:
    """Recover a single postflop bet from state alone, or None.

    Only sound when exactly one player has money in on a street where nobody
    started with any — otherwise the increment is unknowable and MDF should be
    withheld rather than guessed.
    """

    if state.street is Street.PREFLOP or state.current_bet <= _EPS:
        return None

    live = [p for p in state.players if p.committed_this_street > _EPS]
    if len(live) != 1:
        return None

    bettor = live[0]
    return bettor.name, bettor.committed_this_street, state.pot


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
    #: Chips the last aggressor *added* on their aggressive action, and the pot
    #: as it stood immediately before it. Both None when no aggressive action
    #: can be identified. These, not hero's call, define MDF and alpha.
    last_aggressor: str | None = None
    last_wager: float | None = None
    pot_before_aggression: float | None = None

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

        # Replay the street's commitments so the last aggressor's *increment*
        # can be recovered. Their whole street commitment is not what they
        # risked: a small blind raising to 3 has already posted 0.5, so it
        # added 2.5 — and blinds, three-bets and postflop re-raises all break
        # the two apart.
        running = _street_start_commitments(state)
        last_aggressor: str | None = None
        last_wager: float | None = None
        pot_before: float | None = None

        for action in street_actions:
            if action.type.is_aggressive:
                increment = action.amount - previous_level
                if increment >= last_full_raise - _EPS:
                    last_full_raise = increment
                previous_level = action.amount

                last_aggressor = action.actor
                last_wager = action.amount - running.get(action.actor, 0.0)
                pot_before = state.pot + sum(running.values())

            running[action.actor] = action.amount
            committed_when_acted[action.actor] = action.amount

        if last_aggressor is None:
            # No recorded aggression. Postflop a single live bet is
            # unambiguous — nobody carries a commitment into a new street, so
            # the bettor's increment is their whole commitment. Preflop the
            # blinds make that inference false, so nothing is claimed.
            inferred = _infer_lone_bet(state)
            if inferred is not None:
                last_aggressor, last_wager, pot_before = inferred

        return cls(
            street=state.street,
            current_bet=state.current_bet,
            big_blind=state.big_blind,
            last_full_raise=last_full_raise,
            committed_when_acted=committed_when_acted,
            order=tuple(name for name in ordered if state.player(name).can_act),
            has_history=bool(street_actions),
            last_aggressor=last_aggressor,
            last_wager=last_wager,
            pot_before_aggression=pot_before,
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

        # Owing chips always demands a response, even from a lone player whose
        # opponents are all in — they still have to call or fold.
        for name in self.order:
            player = state.player(name)
            if self.current_bet - player.committed_this_street > _EPS:
                return name

        # Nobody owes anything. A player yet to act only matters if there is
        # somebody to bet into: with fewer than two able to act, there is no
        # betting round to finish and a check would be recorded into a pot
        # nobody can contest.
        if len(self.order) < 2:
            return None

        for name in self.order:
            if name not in self.committed_when_acted:
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

        # Folding is legal whenever it is your turn — including when you could
        # check for free. It is a terrible action there, not an illegal one,
        # and hand histories do record it.
        out.append(LegalAction(ActionType.FOLD))

        if owed > _EPS:
            out.append(
                LegalAction(
                    ActionType.CALL,
                    player.committed_this_street + owed,
                    player.committed_this_street + owed,
                )
            )
        else:
            out.append(LegalAction(ActionType.CHECK))

        # Aggression needs someone able to answer it. With every opponent
        # all-in there is no live money to win, so a bet would build a side pot
        # nobody can contest — chip-conserving, but not a legal hand.
        if not self._has_a_live_opponent(state, name):
            return out

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

    @staticmethod
    def _has_a_live_opponent(state: "HandState", name: str) -> bool:
        return any(p.can_act for p in state.active_players if p.name != name)

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

        # Validate against the listed options rather than re-deriving legality
        # here. Duplicated rules drift: the earlier version returned early for
        # every non-raise, and returned early again for any all-in raise —
        # which let a player shove after an under-raise that never reopened
        # the betting, an action `legal_actions` had already excluded.
        options = self.legal_actions(state, action.actor)
        matching = [option for option in options if option.type is action.type]

        if not matching:
            offered = ", ".join(sorted(o.type.value for o in options)) or "nothing"
            raise ValueError(
                f"{action.actor} may not {action.type.value} here; "
                f"legal actions are: {offered}"
            )

        if not any(option.permits(action.amount) for option in matching):
            allowed = "; ".join(str(option) for option in matching)
            raise ValueError(
                f"{action.type.value} to {action.amount:g} is not a legal size; "
                f"allowed: {allowed}"
            )
