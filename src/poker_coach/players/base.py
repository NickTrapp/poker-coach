"""Simulated opponents.

These are **rule-based archetypes, not solver output.** Each one encodes how a
recognisable player type behaves — a nit folds too much, a station calls too
much — so the coach can be asked "what does this opponent do here?" and a
student can practise exploiting a known tendency. They are deliberately not
balanced, and beating them proves nothing about GTO play.

Decisions run through :mod:`poker_coach.calculations`, so an opponent's read on
its own hand uses the same equity engine the coach quotes. Every archetype takes
an explicit ``rng``, so a simulated hand is reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..calculations.pot_odds import required_equity
from ..calculations.ranges import Range
from ..domain.action import Action
from ..domain.enums import ActionType, Street
from ..domain.state import HandState

__all__ = ["Player", "PlayerStyle", "RuleBasedPlayer"]


@runtime_checkable
class Player(Protocol):
    """Anything that can take a turn."""

    name: str

    def act(self, state: HandState) -> Action: ...


@dataclass(frozen=True, slots=True)
class PlayerStyle:
    """The knobs that distinguish one archetype from another.

    Thresholds are hand equity against :attr:`assumed_villain_range`, expressed
    as fractions in ``[0, 1]``.
    """

    label: str
    open_range: str = "22+, A2s+, K9s+, QTs+, JTs, ATo+, KQo"
    #: The range credited to an opponent who has *not* bet.
    assumed_villain_range: str = "random"
    #: The range credited to an opponent who *has* bet. Crediting a bettor with
    #: a random range badly overstates the player's own equity — it is what
    #: made even the nit call K-high on an ace-high board — so aggression is
    #: read as strength here.
    assumed_bettor_range: str = "22+, A2s+, K8s+, Q9s+, J9s+, T9s, A8o+, KTo+, QJo"

    #: Equity at which the player raises rather than calls.
    raise_threshold: float = 0.75
    #: Equity at which the player bets an unbet pot.
    bet_threshold: float = 0.60
    #: Extra equity demanded above the pot-odds price before calling. Negative
    #: values mean the player calls at prices that lose money.
    call_margin: float = 0.0
    #: Chance of betting a hand below `bet_threshold` as a bluff.
    bluff_frequency: float = 0.0
    #: Bet size as a fraction of the pot.
    bet_fraction: float = 0.66

    def range(self) -> Range:
        return Range(self.open_range)


@dataclass
class RuleBasedPlayer:
    """Plays a fixed, legible policy driven by a :class:`PlayerStyle`.

    The policy, in full:

    * **Preflop**, a hand outside ``open_range`` never puts money in — it folds
      to a bet and checks otherwise.
    * **Facing a bet**, raise above ``raise_threshold``, call when equity clears
      the pot-odds price plus ``call_margin``, otherwise fold.
    * **Unbet pot**, bet above ``bet_threshold``, otherwise bluff with
      probability ``bluff_frequency``, otherwise check.

    Every returned action is legal for the given state — sizes are capped by the
    stack and collapse to an all-in when the player cannot cover the raise.
    """

    name: str
    style: PlayerStyle
    rng: random.Random = field(default_factory=random.Random)
    iterations: int = 400

    def act(self, state: HandState) -> Action:
        player = state.player(self.name)
        if player.hole_cards is None:
            raise ValueError(f"{self.name} has no hole cards and cannot act")
        if not player.can_act:
            raise ValueError(f"{self.name} cannot act (folded or all-in)")

        facing_bet = state.amount_to_call(self.name) > 0
        equity = self._equity(state, player.hole_cards, facing_bet=facing_bet)
        return self.decide(state, equity)

    def decide(self, state: HandState, equity: float) -> Action:
        """The policy itself, given an equity read.

        Split out from :meth:`act` so range conditioning can ask *this* policy
        what a hypothetical combo would do, using a precomputed equity, instead
        of maintaining a second copy of the decision rules. Two copies of a rule
        drift; this project has the scars.
        """

        player = state.player(self.name)
        to_call = state.amount_to_call(self.name)
        facing_bet = to_call > 0

        if state.street is Street.PREFLOP and not self._opens(player.hole_cards):
            return self._fold_or_check(facing_bet, state.street)

        if facing_bet:
            return self._respond_to_bet(state, equity, to_call)
        return self._open_action(state, equity)

    def action_probability(
        self, state: HandState, equity: float, taken: ActionType
    ) -> float:
        """How often this policy takes ``taken`` with that equity read.

        Every branch is deterministic except one: betting an unbet pot with a
        hand below the value threshold happens with probability
        ``bluff_frequency``. So the answer is 0, 1, or that frequency — computed
        rather than estimated by repeated sampling, which keeps the posterior
        from being brittle near the boundary.
        """

        player = state.player(self.name)
        to_call = state.amount_to_call(self.name)
        facing_bet = to_call > 0

        # A hand outside the opening range never puts money in preflop.
        if state.street is Street.PREFLOP and not self._opens(player.hole_cards):
            expected = ActionType.FOLD if facing_bet else ActionType.CHECK
            return 1.0 if taken is expected else 0.0

        if facing_bet:
            return 1.0 if self._respond_to_bet(state, equity, to_call).type is taken else 0.0

        # Unbet pot: the value branch is certain, the bluff branch is not.
        aggressive = (
            ActionType.RAISE if state.current_bet > 0 else ActionType.BET
        )
        if equity >= self.style.bet_threshold:
            return 1.0 if taken is aggressive else 0.0

        bluff = self.style.bluff_frequency
        if not self._may(state, aggressive):
            bluff = 0.0
        if taken is aggressive:
            return bluff
        if taken is ActionType.CHECK:
            return 1.0 - bluff
        return 0.0

    # ---------------------------------------------------------------- policy

    def _opens(self, hole_cards) -> bool:
        return list(hole_cards) in self.style.range()

    def _equity(self, state: HandState, hole_cards, *, facing_bet: bool) -> float:
        from ..calculations.equity import equity as compute_equity

        assumed = (
            self.style.assumed_bettor_range
            if facing_bet
            else self.style.assumed_villain_range
        )
        result = compute_equity(
            hole_cards,
            Range(assumed),
            state.board,
            iterations=self.iterations,
            rng=self.rng,
        )
        return result.equity

    def _may(self, state: HandState, kind: ActionType) -> bool:
        """Whether ``kind`` is currently legal, per the rules engine.

        The policy asks rather than assumes. An under-raise all-in does not
        reopen the betting, so a hand strong enough to raise may still only be
        allowed to call — a case this policy used to get wrong, and which the
        state machine now rejects outright.
        """

        return any(option.type is kind for option in state.legal_actions(self.name))

    def _respond_to_bet(
        self, state: HandState, equity: float, to_call: float
    ) -> Action:
        price = required_equity(state.total_pot, to_call)

        if equity >= self.style.raise_threshold and self._may(state, ActionType.RAISE):
            target = self._raise_target(state)
            if target is not None:
                return Action(
                    actor=self.name,
                    type=ActionType.RAISE,
                    amount=target,
                    street=state.street,
                )

        if equity >= price + self.style.call_margin:
            player = state.player(self.name)
            return Action(
                actor=self.name,
                type=ActionType.CALL,
                amount=player.committed_this_street + to_call,
                street=state.street,
            )

        return Action(actor=self.name, type=ActionType.FOLD, street=state.street)

    def _open_action(self, state: HandState, equity: float) -> Action:
        wants_value = equity >= self.style.bet_threshold
        wants_bluff = (
            not wants_value
            and self.style.bluff_frequency > 0
            and self.rng.random() < self.style.bluff_frequency
        )

        if wants_value or wants_bluff:
            if state.current_bet > 0 and self._may(state, ActionType.RAISE):
                # Owing nothing while a bet is live is the big blind's option:
                # everyone has matched, but the blind itself is a live bet, so
                # aggression here is a raise. Betting would be illegal.
                target = self._raise_target(state)
                if target is not None:
                    return Action(
                        actor=self.name,
                        type=ActionType.RAISE,
                        amount=target,
                        street=state.street,
                    )
            elif self._may(state, ActionType.BET):
                size = self._bet_size(state)
                if size is not None:
                    return Action(
                        actor=self.name,
                        type=ActionType.BET,
                        amount=size,
                        street=state.street,
                    )

        return Action(actor=self.name, type=ActionType.CHECK, street=state.street)

    # ----------------------------------------------------------------- sizes

    def _bet_size(self, state: HandState) -> float | None:
        """A pot-fraction bet, floored at one big blind and capped by the stack.

        A fraction of a small pot can land below the minimum bet — 0.66 of a
        1.2-chip pot is 0.79, under a 1-chip blind — which the rules engine
        rejects. Clamp up to the floor, then down to the stack, so a short
        stack still goes all-in rather than proposing something illegal.
        """

        player = state.player(self.name)
        if player.stack <= 0:
            return None
        wanted = self.style.bet_fraction * state.total_pot
        return min(max(wanted, state.big_blind), player.stack)

    def _raise_target(self, state: HandState) -> float | None:
        """A pot-sized raise, capped by the stack. None if a raise is illegal."""

        player = state.player(self.name)
        current_bet = state.current_bet
        to_call = state.amount_to_call(self.name)

        wanted = current_bet + (state.total_pot + to_call)
        ceiling = player.committed_this_street + player.stack
        # Never propose below the legal minimum; an all-in is always allowed.
        floor = state.betting_round().min_raise_to(
            player.committed_this_street, player.stack
        )
        target = min(max(wanted, floor), ceiling)

        # An "all-in" that cannot exceed the current bet is a call, not a raise.
        return target if target > current_bet else None

    def _fold_or_check(self, facing_bet: bool, street: Street) -> Action:
        kind = ActionType.FOLD if facing_bet else ActionType.CHECK
        return Action(actor=self.name, type=kind, street=street)
