"""Deal and play hands out between simulated opponents.

Supports two to nine seats. Two things make multiway harder than heads-up, and
both are handled here rather than approximated:

**Action order.** Seats sit in a fixed ring — SB, BB, UTG … CO, BTN. Preflop
the action opens to the *left of the big blind* and the blinds close it;
afterwards it opens at the small blind and the button closes it. Heads-up is
the standard exception: the small blind holds the button, so it opens preflop
and closes every later street.

**Side pots.** When a short stack is all-in, chips it could not match belong to
a separate pot it cannot win. Settlement walks the distinct contribution levels
and awards each layer only among the players who paid into it, so a short stack
can never scoop chips it never covered.

Every action goes through `HandState.apply`, so the runner cannot produce a hand
the rules engine would reject, and the returned `HandHistory` replays to the
same final state.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from ..calculations.hand_eval import HandRank, evaluate
from ..domain.action import Action
from ..domain.cards import Card, Deck
from ..domain.betting import action_order
from ..domain.enums import Position, Street
from ..domain.history import HandHistory, SeatRecord, StreetRecord
from ..domain.state import HandState
from .base import Player

__all__ = ["Seat", "HandResult", "play_hand", "MAX_SEATS"]

MAX_SEATS = 9

_STREET_DEAL = {Street.FLOP: 3, Street.TURN: 1, Street.RIVER: 1}

@dataclass(frozen=True, slots=True)
class Seat:
    """One player at the table, with their seat and starting chips."""

    player: Player
    position: Position
    stack: float = 100.0

    @property
    def name(self) -> str:
        return self.player.name


@dataclass(frozen=True, slots=True)
class HandResult:
    """The outcome of one played hand."""

    history: HandHistory
    final_state: HandState
    awards: dict[str, float]
    pot: float
    showdown: bool
    hands: dict[str, HandRank] = field(default_factory=dict)

    @property
    def winners(self) -> list[str]:
        """Everyone who won chips, best pot first is not implied — order is seat order."""

        return [name for name, amount in self.awards.items() if amount > 0]

    def net(self, name: str) -> float:
        """Chips won or lost by ``name`` across the whole hand."""

        won = self.awards.get(name, 0.0)
        return won - self.final_state.player(name).committed_total

    def __str__(self) -> str:  # pragma: no cover - display helper
        how = "showdown" if self.showdown else "no showdown"
        who = ", ".join(self.winners) or "nobody"
        return f"{who} wins {self.pot:g} ({how})"


def _play_street(
    state: HandState,
    players: dict[str, Player],
    order: list[str],
    record: list[Action],
) -> HandState:
    """Run one betting round to completion.

    A round closes when every player who can act has acted at least once *since
    the last aggression* and owes nothing. Re-scanning from the top of the order
    after each action is what gives a raise its correct effect: everyone else
    owes chips again and must act again.
    """

    # With fewer than two players able to act, there is no betting to do — the
    # remaining cards simply run out. Recording a check here would produce a
    # hand where someone bets into a pot nobody can contest.
    can_act = [p for p in state.players if p.can_act]
    someone_owes = any(state.amount_to_call(p.name) > 0 for p in can_act)
    if len(can_act) < 2 and not someone_owes:
        return state

    acted: set[str] = set()

    while len(state.active_players) > 1:
        progressed = False

        for name in order:
            player_state = state.player(name)
            if not player_state.can_act:
                continue
            if state.amount_to_call(name) == 0 and name in acted:
                continue

            action = players[name].act(state)
            action = action.model_copy(update={"street": state.street})
            state = state.apply(action, strict=True)
            record.append(action)
            acted.add(name)

            if action.type.is_aggressive:
                acted = {name}  # everyone else owes chips again

            progressed = True
            break  # re-scan from the top so the order is respected

        if not progressed:
            break

    return state


def _settle(
    state: HandState,
) -> tuple[dict[str, float], dict[str, HandRank], bool]:
    """Award the pot, splitting into side pots where stacks differ."""

    contributions = {p.name: p.committed_total for p in state.players}
    contenders = state.active_players

    if len(contenders) <= 1:
        total = sum(contributions.values())
        winner = contenders[0].name if contenders else None
        return ({winner: total} if winner else {}), {}, False

    hands: dict[str, HandRank] = {}
    for player in contenders:
        if player.hole_cards is None:  # pragma: no cover - runner always deals
            raise ValueError(f"{player.name} reached showdown without cards")
        hands[player.name] = evaluate([*player.hole_cards, *state.board])

    awards: dict[str, float] = defaultdict(float)
    previous = 0.0

    # Walk the distinct contribution levels. Each layer is funded by everyone
    # who paid into it — including folded players — but only contenders who
    # covered that level are eligible to win it.
    for level in sorted(set(contributions.values())):
        layer = sum(
            min(paid, level) - min(paid, previous) for paid in contributions.values()
        )
        previous = level
        if layer <= 0:
            continue

        eligible = [name for name in hands if contributions[name] >= level]
        if not eligible:
            # Only reachable if every contender folded below this level; the
            # chips belong to whoever is still in the hand.
            eligible = list(hands)

        best = max(hands[name].score for name in eligible)
        winners = [name for name in eligible if hands[name].score == best]
        for name in winners:
            awards[name] += layer / len(winners)

    return dict(awards), hands, True


def play_hand(
    seats: list[Seat],
    *,
    rng: random.Random | None = None,
    small_blind: float = 0.5,
    big_blind: float = 1.0,
    hero: str | None = None,
    board: list[Card] | None = None,
) -> HandResult:
    """Deal and play one hand.

    ``board`` pins the community cards for a reproducible scenario; otherwise
    they come off the shuffled deck.
    """

    if not 2 <= len(seats) <= MAX_SEATS:
        raise ValueError(
            f"a hand needs between 2 and {MAX_SEATS} seats, got {len(seats)}"
        )

    positions = {seat.position for seat in seats}
    if len(positions) != len(seats):
        raise ValueError("two seats share a position")
    missing = {Position.SB, Position.BB} - positions
    if missing:
        raise ValueError(
            "the blinds must be seated; missing "
            f"{', '.join(sorted(p.value for p in missing))}"
        )

    rng = rng or random.Random()
    deck = Deck(rng=rng).shuffle()

    # A pinned board must leave the deck *before* hole cards are dealt, or a
    # board card can be dealt into a player's hand and the state is rejected
    # as holding duplicates.
    runout: list[Card] = []
    if board is not None:
        if len(board) != 5:
            raise ValueError(f"a pinned board needs 5 cards, got {len(board)}")
        deck.remove(board)
        runout = list(board)

    hole: dict[str, tuple[Card, Card]] = {}
    for seat in seats:
        dealt = deck.deal(2)
        hole[seat.name] = (dealt[0], dealt[1])

    if not runout:
        runout = deck.deal(5)

    by_position = {seat.position: seat.name for seat in seats}
    players = {seat.name: seat.player for seat in seats}
    hero_name = hero or by_position[Position.BB]

    seat_records = [
        SeatRecord(
            name=seat.name,
            position=seat.position,
            starting_stack=seat.stack,
            hole_cards=hole[seat.name],
        )
        for seat in seats
    ]
    skeleton = HandHistory(
        seats=seat_records,
        hero=hero_name,
        small_blind=small_blind,
        big_blind=big_blind,
    )

    state = skeleton.initial_state()
    street_records: list[StreetRecord] = []
    dealt_so_far = 0

    for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
        new_cards: list[Card] = []
        if street is not Street.PREFLOP:
            count = _STREET_DEAL[street]
            new_cards = runout[dealt_so_far : dealt_so_far + count]
            dealt_so_far += count
            state = state.advance_street(new_cards)

        actions: list[Action] = []
        if len(state.active_players) > 1 and state.players_to_act:
            order = [by_position[p] for p in action_order(street, positions)]
            state = _play_street(state, players, order, actions)

        street_records.append(
            StreetRecord(street=street, cards=new_cards, actions=actions)
        )

        if len(state.active_players) <= 1:
            break

    history = HandHistory(
        seats=seat_records,
        hero=hero_name,
        small_blind=small_blind,
        big_blind=big_blind,
        streets=street_records,
    )

    awards, hands, showdown = _settle(state)
    return HandResult(
        history=history,
        final_state=state,
        awards=awards,
        pot=state.total_pot,
        showdown=showdown,
        hands=hands,
    )
