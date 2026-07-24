"""Hand state: players, board, pot, and a minimal betting state machine.

`HandState` is deliberately immutable-by-convention: :meth:`HandState.apply`
returns a new state rather than mutating in place, so a coaching session can
hold onto every intermediate node of a hand for explanation and replay.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .action import Action
from .cards import Card
from .enums import ActionType, Position, Street

__all__ = ["PlayerState", "HandState"]


class PlayerState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    name: str
    position: Position
    stack: float = Field(ge=0.0, description="Chips still behind.")
    hole_cards: tuple[Card, Card] | None = None
    has_folded: bool = False
    is_all_in: bool = False
    committed_this_street: float = Field(default=0.0, ge=0.0)
    committed_total: float = Field(default=0.0, ge=0.0)
    is_hero: bool = False

    @property
    def is_active(self) -> bool:
        """Still in the hand (may or may not still have chips to bet)."""

        return not self.has_folded

    @property
    def can_act(self) -> bool:
        return not self.has_folded and not self.is_all_in

    def __str__(self) -> str:  # pragma: no cover - trivial
        cards = "".join(str(c) for c in self.hole_cards) if self.hole_cards else "??"
        return f"{self.name}({self.position.value}, {self.stack:g}, {cards})"


class HandState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    players: list[PlayerState]
    board: list[Card] = Field(default_factory=list)
    street: Street = Street.PREFLOP
    small_blind: float = Field(default=0.5, gt=0.0)
    big_blind: float = Field(default=1.0, gt=0.0)
    ante: float = Field(default=0.0, ge=0.0)
    pot: float = Field(
        default=0.0,
        ge=0.0,
        description="Chips already gathered from *previous* streets.",
    )
    actions: list[Action] = Field(default_factory=list)

    # ------------------------------------------------------------------ checks

    @model_validator(mode="after")
    def _validate(self) -> "HandState":
        if not self.players:
            raise ValueError("a hand needs at least one player")

        names = [p.name for p in self.players]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate player names: {names}")

        positions = [p.position for p in self.players]
        if len(set(positions)) != len(positions):
            raise ValueError(f"duplicate positions: {positions}")

        expected = self.street.board_size
        if len(self.board) != expected:
            raise ValueError(
                f"{self.street.value} expects {expected} board cards, "
                f"got {len(self.board)}"
            )

        seen: set[Card] = set()
        for card in self._all_known_cards():
            if card in seen:
                raise ValueError(f"duplicate card in hand: {card}")
            seen.add(card)

        return self

    def _all_known_cards(self) -> Iterable[Card]:
        yield from self.board
        for player in self.players:
            if player.hole_cards:
                yield from player.hole_cards

    # -------------------------------------------------------------- accessors

    def player(self, name: str) -> PlayerState:
        for player in self.players:
            if player.name == name:
                return player
        raise KeyError(f"no player named {name!r}")

    @property
    def hero(self) -> PlayerState | None:
        for player in self.players:
            if player.is_hero:
                return player
        return None

    @property
    def active_players(self) -> list[PlayerState]:
        return [p for p in self.players if p.is_active]

    @property
    def players_to_act(self) -> list[PlayerState]:
        return [p for p in self.players if p.can_act]

    @property
    def current_bet(self) -> float:
        """Highest street commitment any player has made."""

        return max((p.committed_this_street for p in self.players), default=0.0)

    @property
    def total_pot(self) -> float:
        """Everything in the middle, including this street's live bets."""

        return self.pot + sum(p.committed_this_street for p in self.players)

    def amount_to_call(self, name: str) -> float:
        """Chips ``name`` must add to continue — capped by their stack."""

        player = self.player(name)
        owed = self.current_bet - player.committed_this_street
        return max(0.0, min(owed, player.stack))

    def stack_behind(self, name: str) -> float:
        """Chips ``name`` still has *right now*, after any bet already made."""

        return self.player(name).stack

    def stack_at_street_start(self, name: str) -> float:
        """Chips ``name`` had when this street began.

        Equal to chips behind plus whatever they have already committed on this
        street. This is the conventional numerator for SPR.
        """

        player = self.player(name)
        return player.stack + player.committed_this_street

    def effective_stack(self, *names: str) -> float:
        """Smallest start-of-street stack among the named players.

        "Effective" means *the most either player can actually lose*, so it is
        a minimum across players and is meaningless for one player alone —
        passing a single name returns that player's own stack, which is why
        :meth:`spr` must never do so (it did, and reported a 10x-wrong SPR when
        stacks were asymmetric).
        """

        pool = [self.player(name) for name in names] if names else self.active_players
        if not pool:
            return 0.0
        return min(p.stack + p.committed_this_street for p in pool)

    def spr(self, name: str | None = None) -> float:
        """Stack-to-pot ratio: effective start-of-street stack over the pot.

        ``name`` is accepted for call-site readability but does not narrow the
        numerator — SPR is a property of the *pair*, not of one player. The
        denominator is :attr:`total_pot`, which includes any live bet, so a bet
        that is still pending lowers the ratio.
        """

        pot = self.total_pot
        if pot <= 0:
            return float("inf")
        return self.effective_stack() / pot

    def street_actions(self, street: Street | None = None) -> list[Action]:
        target = street or self.street
        return [a for a in self.actions if a.street is target]

    # ------------------------------------------------------------ transitions

    def apply(self, action: Action) -> "HandState":
        """Return a new state with ``action`` applied. Raises on illegal actions."""

        state = self.model_copy(deep=True)
        player = state.player(action.actor)

        if player.has_folded:
            raise ValueError(f"{player.name} has already folded")
        if player.is_all_in:
            raise ValueError(f"{player.name} is all-in and cannot act")

        if action.type is ActionType.FOLD:
            player.has_folded = True

        elif action.type is ActionType.CHECK:
            owed = state.amount_to_call(player.name)
            if owed > 0:
                raise ValueError(
                    f"{player.name} cannot check facing a bet of {owed:g}"
                )

        else:
            state._commit(player, action)

        state.actions = [*state.actions, action]
        return state

    def _commit(self, player: PlayerState, action: Action) -> None:
        current_bet = self.current_bet
        target = action.amount
        added = target - player.committed_this_street

        if added <= 0:
            raise ValueError(
                f"{action.type.value} to {target:g} does not increase "
                f"{player.name}'s commitment of {player.committed_this_street:g}"
            )
        if added > player.stack + 1e-9:
            raise ValueError(
                f"{player.name} cannot put in {added:g}, stack is {player.stack:g}"
            )

        if action.type is ActionType.BET and current_bet > 0:
            raise ValueError("cannot bet facing a bet; use raise")
        if action.type is ActionType.RAISE and target <= current_bet:
            raise ValueError(
                f"raise to {target:g} does not exceed current bet {current_bet:g}"
            )
        if action.type is ActionType.CALL:
            max_call = min(current_bet, player.committed_this_street + player.stack)
            if abs(target - max_call) > 1e-9:
                raise ValueError(
                    f"call must bring {player.name} to {max_call:g}, got {target:g}"
                )

        player.stack -= added
        player.committed_this_street = target
        player.committed_total += added
        if player.stack <= 1e-9:
            player.stack = 0.0
            player.is_all_in = True

    def advance_street(self, new_cards: Sequence[Card] = ()) -> "HandState":
        """Collect bets, deal ``new_cards``, and move to the next street."""

        if self.street is Street.SHOWDOWN:
            raise ValueError("the hand is already at showdown")

        next_street = self.street.next()
        board = [*self.board, *new_cards]
        if len(board) != next_street.board_size:
            raise ValueError(
                f"{next_street.value} needs {next_street.board_size} board cards, "
                f"got {len(board)}"
            )

        # Build the successor in one shot: `validate_assignment` would reject a
        # half-updated state (new board with the old street, and vice versa).
        players = [p.model_copy(deep=True) for p in self.players]
        for player in players:
            player.committed_this_street = 0.0

        return HandState(
            players=players,
            board=board,
            street=next_street,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            ante=self.ante,
            pot=self.total_pot,
            actions=list(self.actions),
        )

    def __str__(self) -> str:  # pragma: no cover - display helper
        board = "".join(str(c) for c in self.board) or "-"
        return (
            f"<HandState {self.street.value} board={board} "
            f"pot={self.total_pot:g} players={len(self.active_players)}>"
        )
