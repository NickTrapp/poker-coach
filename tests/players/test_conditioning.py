"""Conditioning an opponent range on the action it took.

The structural properties are checked exactly. The agreement check is
deliberately tolerant: conditioning ranks combos using one sampled equity grid,
and re-running the policy draws its own sample, so combos sitting within
sampling error of a threshold can legitimately fall on either side. That band is
a real property of the method, not a defect in the test, and pretending
otherwise would hide the least trustworthy part of the posterior.
"""

import random

import pytest

from poker_coach.calculations.ranges import Range
from poker_coach.domain import (
    Action,
    ActionType,
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)
from poker_coach.players import condition_range, make_player

STATION = "22+, A2s+, K2s+, Q6s+, J7s+, T7s+, A2o+, K8o+, Q9o+, JTo"


def unbet_flop(board: str = "Ac8d3s", hero: str = "7s7h") -> HandState:
    return HandState(
        players=[
            PlayerState(name="v", position=Position.BB, stack=94.0,
                        hole_cards=tuple(parse_cards("AhQd"))),
            PlayerState(name="you", position=Position.SB, stack=94.0,
                        hole_cards=tuple(parse_cards(hero)), is_hero=True),
        ],
        board=parse_cards(board), street=Street.FLOP, pot=12.0,
    )


def facing_bet_flop() -> HandState:
    return unbet_flop().apply(
        Action(actor="you", type=ActionType.BET, amount=6.0, street=Street.FLOP)
    )


def station(seed: int = 1):
    return make_player("v", "station", rng=random.Random(seed), iterations=300)


def condition(state, kind, amount=0.0, style="station", seed=2,
              prior=STATION, **kw):
    return condition_range(
        Range(prior), make_player("v", style, rng=random.Random(1), iterations=300),
        state,
        Action(actor="v", type=kind, amount=amount, street=state.street),
        rng=random.Random(seed), **kw,
    )


# ---------------------------------------------------------- structure


def test_the_posterior_is_a_subset_of_the_prior():
    posterior = condition(unbet_flop(), ActionType.BET, 6.0)
    prior = set(Range(STATION).combos(dead=parse_cards("Ac8d3s7s7h")))
    assert set(posterior.combos) <= prior
    assert 0 < posterior.survivors < len(prior)


def test_betting_and_checking_partition_a_deterministic_range():
    """A station never bluffs, so every combo does exactly one of the two."""

    state = unbet_flop()
    bet = condition(state, ActionType.BET, 6.0)
    check = condition(state, ActionType.CHECK)

    assert set(bet.combos) & set(check.combos) == set()
    assert bet.survivors + check.survivors == bet.considered
    assert bet.considered == check.considered


def test_a_bettors_range_is_stronger_than_a_checkers():
    """The point of the exercise: the posterior should move the equity."""

    state = unbet_flop()
    bet = condition(state, ActionType.BET, 6.0)
    check = condition(state, ActionType.CHECK)

    from poker_coach.calculations.equity import equity

    hero = parse_cards("7s7h")
    board = parse_cards("Ac8d3s")
    vs_bettor = equity(hero, bet.to_range(), board,
                       iterations=3000, rng=random.Random(4))
    vs_checker = equity(hero, check.to_range(), board,
                        iterations=3000, rng=random.Random(4))
    assert vs_bettor.equity < vs_checker.equity


def test_conditioning_on_a_fold_keeps_only_hands_that_fold():
    posterior = condition(facing_bet_flop(), ActionType.FOLD)
    assert posterior.survivors > 0
    calls = condition(facing_bet_flop(), ActionType.CALL, 6.0)
    assert set(posterior.combos) & set(calls.combos) == set()


def test_heros_cards_are_removed_from_the_opponents_range():
    # The opponent's own cards are irrelevant here — they are what is being
    # inferred — so the fixture gives it a holding that shares nothing.
    state = HandState(
        players=[
            PlayerState(name="v", position=Position.BB, stack=94.0,
                        hole_cards=tuple(parse_cards("2c2d"))),
            PlayerState(name="you", position=Position.SB, stack=94.0,
                        hole_cards=tuple(parse_cards("AhAd")), is_hero=True),
        ],
        board=parse_cards("Kc8h3s"), street=Street.FLOP, pot=12.0,
    )
    posterior = condition(state, ActionType.BET, 6.0)
    blocked = set(parse_cards("AhAd"))
    assert posterior.combos
    for combo in posterior.combos:
        assert not blocked & set(combo)


def test_board_cards_are_removed_too():
    posterior = condition(unbet_flop(), ActionType.BET, 6.0)
    board = set(parse_cards("Ac8d3s"))
    for combo in posterior.combos:
        assert not board & set(combo)


def test_an_impossible_action_yields_an_empty_posterior():
    # A station never bluffs, so no combo below its value threshold bets...
    # but conditioning on an action the policy simply cannot take is empty.
    posterior = condition(unbet_flop(), ActionType.RAISE, 6.0)
    assert posterior.survivors == 0
    assert "0 of" in posterior.describe()


# ------------------------------------------------- weights and honesty


def test_a_deterministic_policy_gives_a_certain_posterior():
    assert condition(unbet_flop(), ActionType.BET, 6.0).is_certain


def test_a_bluffing_policy_gives_fractional_weights():
    """A maniac bets weak hands only sometimes, and the posterior says so.

    Needs a prior wide enough to contain hands below the maniac's 35% value
    threshold — every combo in the station's range clears it against a random
    hand, so they would all read as value bets.
    """

    posterior = condition(unbet_flop(), ActionType.BET, 6.0, style="maniac",
                          prior="random")
    weights = {w for w in posterior.weights.values() if w > 0}
    assert any(w < 1.0 for w in weights)
    assert not posterior.is_certain
    assert "only bluff here sometimes" in posterior.describe()
    assert "proportionally less weight" in posterior.describe()


def test_the_description_reports_how_much_was_eliminated():
    posterior = condition(unbet_flop(), ActionType.BET, 6.0)
    text = posterior.describe()
    assert f"{posterior.survivors} of {posterior.considered}" in text
    assert "bet" in text and "flop" in text


# ------------------------------------------- agreement with the policy


def test_survivors_mostly_take_the_action_when_the_policy_is_re_run():
    """Independent check: put each surviving combo in the seat and ask.

    Not required to be perfect — the policy re-samples its own equity, so
    combos within sampling error of the bet threshold may land either way.
    """

    state = unbet_flop()
    posterior = condition(state, ActionType.BET, 6.0)
    checker = station(seed=99)

    agree = 0
    sample = posterior.combos[:60]
    for combo in sample:
        hypothetical = state.model_copy(deep=True)
        hypothetical.player("v").hole_cards = combo
        if checker.act(hypothetical).type is ActionType.BET:
            agree += 1

    assert agree / len(sample) > 0.85, f"only {agree}/{len(sample)} agreed"


def test_excluded_combos_mostly_do_not_take_the_action():
    state = unbet_flop()
    posterior = condition(state, ActionType.BET, 6.0)
    excluded = [c for c, w in posterior.weights.items() if w == 0][:60]
    checker = station(seed=99)

    agree = 0
    for combo in excluded:
        hypothetical = state.model_copy(deep=True)
        hypothetical.player("v").hole_cards = combo
        if checker.act(hypothetical).type is not ActionType.BET:
            agree += 1

    assert agree / len(excluded) > 0.85, f"only {agree}/{len(excluded)} agreed"


# ----------------------------------------------------------- equity grid


def test_the_grid_agrees_with_standalone_equity_within_sampling_error():
    from poker_coach.calculations.equity import equity, equity_grid

    board = parse_cards("Qh2c9d")
    combos = Range("AA, KK, 72o").combos(dead=board)
    grid = equity_grid(combos, Range(STATION), board,
                       iterations=3000, rng=random.Random(1))

    for combo in combos:
        solo = equity(combo, Range(STATION), board,
                      iterations=3000, rng=random.Random(2))
        assert abs(grid[combo] - solo.equity) < 0.06, combo


def test_the_grid_ranks_hands_the_way_equity_does():
    from poker_coach.calculations.equity import equity_grid

    board = parse_cards("Qh2c9d")
    combos = Range("AA, KK, 72o").combos(dead=board)
    grid = equity_grid(combos, Range(STATION), board,
                       iterations=2000, rng=random.Random(1))

    aces = [c for c in combos if all(x.rank.symbol == "A" for x in c)][0]
    trash = [c for c in combos
             if {x.rank.symbol for x in c} == {"7", "2"}][0]
    assert grid[aces] > grid[trash]


def test_the_grid_rejects_zero_iterations():
    from poker_coach.calculations.equity import equity_grid

    with pytest.raises(ValueError, match="iterations must be positive"):
        equity_grid([parse_cards("AsKs")], Range("22+"), parse_cards("Qh2c9d"),
                    iterations=0)


# ------------------------------------------------------------- chaining


def test_chaining_narrows_further_than_one_step():
    """Passing a posterior back in conditions what the last step left."""

    state = unbet_flop()
    player = make_player("v", "station", rng=random.Random(1), iterations=300)
    bet = Action(actor="v", type=ActionType.BET, amount=6.0, street=Street.FLOP)

    first = condition_range(Range(STATION), player, state, bet,
                            rng=random.Random(2))

    # A second action on a turn card that misses most of the range.
    turn = HandState(
        players=[
            PlayerState(name="v", position=Position.BB, stack=88.0,
                        hole_cards=tuple(parse_cards("AhQd"))),
            PlayerState(name="you", position=Position.SB, stack=88.0,
                        hole_cards=tuple(parse_cards("7s7h")), is_hero=True),
        ],
        board=parse_cards("Ac8d3s2h"), street=Street.TURN, pot=24.0,
    )
    second = condition_range(
        first, player,
        turn, Action(actor="v", type=ActionType.BET, amount=12.0,
                     street=Street.TURN),
        rng=random.Random(3),
    )

    assert set(second.combos) <= set(first.combos)
    assert len(second.steps) == 2
    # The reported denominator stays the size before any conditioning, so the
    # narrowing figure means the same thing at every step of the hand.
    assert second.steps[0].considered == first.considered
    assert second.steps[1].considered == len(
        [c for c in first.combos if not set(c) & set(parse_cards("2h"))]
    )


def test_chaining_multiplies_weights_rather_than_replacing_them():
    """A bluffing branch must not be reset to full weight by a later action.

    Chaining through `to_range()` would do exactly that: the notation keeps
    which combos survived and forgets how much of each one did.
    """

    state = unbet_flop()
    maniac = make_player("v", "maniac", rng=random.Random(1), iterations=300)
    bet = Action(actor="v", type=ActionType.BET, amount=6.0, street=Street.FLOP)

    first = condition_range(Range("random"), maniac, state, bet,
                            rng=random.Random(2))
    bluffs = {c for c, w in first.weights.items() if 0 < w < 1.0}
    assert bluffs, "fixture no longer produces a partial-weight branch"

    # Villain bet, hero raised: now villain has a second, genuine decision.
    facing_raise = state.apply(
        Action(actor="v", type=ActionType.BET, amount=6.0, street=Street.FLOP)
    ).apply(
        Action(actor="you", type=ActionType.RAISE, amount=18.0,
               street=Street.FLOP)
    )
    assert facing_raise.amount_to_call("v") > 0
    second = condition_range(
        first, maniac, facing_raise,
        Action(actor="v", type=ActionType.CALL, amount=18.0,
               street=Street.FLOP),
        rng=random.Random(3),
    )

    carried = [c for c in bluffs if second.weights.get(c, 0) > 0]
    assert carried, "no partial-weight combo survived the second step"
    for combo in carried:
        # Still fractional: the product of two likelihoods, not the later one.
        assert second.weights[combo] < 1.0
        assert second.weights[combo] <= first.weights[combo] + 1e-9


def test_the_line_reads_one_phrase_per_street():
    """It is printed at every decision, so it has to stay short.

    Two actions on one street are one line in poker — "check-call the flop",
    not "check on the flop, then call on the flop".
    """

    state = unbet_flop()
    player = make_player("v", "station", rng=random.Random(1), iterations=300)

    first = condition_range(
        Range(STATION), player, state,
        Action(actor="v", type=ActionType.CHECK, street=Street.FLOP),
        rng=random.Random(2),
    )
    facing = state.apply(
        Action(actor="you", type=ActionType.BET, amount=6.0, street=Street.FLOP)
    )
    second = condition_range(
        first, player, facing,
        Action(actor="v", type=ActionType.CALL, amount=6.0, street=Street.FLOP),
        rng=random.Random(3),
    )

    assert second.line() == "check-call the flop"
    assert len(second.steps) == 2
    # The counts stay per-action; only the wording is grouped.
    assert second.describe().count("combos") == 2
