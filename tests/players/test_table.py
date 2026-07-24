import itertools
import random

import pytest

from poker_coach.domain import ActionType, Position, Street, parse_cards
from poker_coach.players import STYLES, Seat, make_player, play_hand


def table(sb_style="tag", bb_style="tag", seed=7, stack=100.0):
    rng = random.Random(seed)
    return [
        Seat(make_player("sb", sb_style, rng=rng, iterations=120), Position.SB, stack),
        Seat(make_player("bb", bb_style, rng=rng, iterations=120), Position.BB, stack),
    ]


def run(sb_style="tag", bb_style="tag", seed=7, **kwargs):
    rng = random.Random(seed)
    return play_hand(table(sb_style, bb_style, seed), rng=rng, **kwargs)


# ------------------------------------------------------------- invariants


@pytest.mark.parametrize("seed", range(30))
def test_chips_are_conserved(seed):
    result = run(seed=seed)
    contributed = sum(p.committed_total for p in result.final_state.players)
    assert contributed == pytest.approx(result.pot)


@pytest.mark.parametrize("seed", range(30))
def test_history_replays_to_the_same_pot(seed):
    """The recorded hand must reproduce the hand that was played."""

    result = run(seed=seed)
    assert result.history.final_state().total_pot == pytest.approx(result.pot)


@pytest.mark.parametrize("seed", range(30))
def test_result_is_zero_sum(seed):
    result = run(seed=seed)
    assert sum(result.net(n) for n in ("sb", "bb")) == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(30))
def test_no_player_ends_with_a_negative_stack(seed):
    result = run(seed=seed)
    assert all(p.stack >= 0 for p in result.final_state.players)


@pytest.mark.parametrize("seed", range(20))
def test_every_hand_has_a_winner(seed):
    assert run(seed=seed).winners


@pytest.mark.parametrize(
    "pair", list(itertools.product(STYLES, STYLES)), ids=lambda p: str(p)
)
def test_every_archetype_pairing_plays_legally(pair):
    """The runner applies every action, so an illegal one would raise here."""

    sb_style, bb_style = pair
    for seed in range(4):
        result = run(sb_style, bb_style, seed=seed)
        assert result.pot > 0


# ------------------------------------------------------------ action order


def test_small_blind_acts_first_preflop():
    # Heads-up the small blind holds the button, so it opens preflop.
    result = run(seed=3)
    preflop = result.history.streets[0].actions
    assert preflop[0].actor == "sb"


def test_big_blind_acts_first_after_the_flop():
    # ... and is out of position on every later street.
    for seed in range(40):
        result = run("station", "station", seed=seed)
        if len(result.history.streets) > 1:
            flop = result.history.streets[1].actions
            if flop:
                assert flop[0].actor == "bb"
                return
    pytest.fail("no hand reached a flop with action")


def test_big_blind_gets_its_option_when_the_small_blind_limps():
    """Owing nothing while a blind is live must produce a raise, not a bet.

    Regression: the player policy treated "owes nothing" as an unbet pot and
    emitted an illegal BET on the big blind's option.
    """

    for seed in range(60):
        result = run("station", "maniac", seed=seed)
        preflop = result.history.streets[0].actions
        raises = [a for a in preflop if a.actor == "bb" and a.type is ActionType.RAISE]
        if raises:
            assert all(a.type is not ActionType.BET for a in preflop if a.actor == "bb")
            return
    pytest.fail("no hand produced a big-blind option raise")


def test_no_bet_action_ever_faces_a_live_bet():
    for seed in range(40):
        for street in run("maniac", "maniac", seed=seed).history.streets:
            bets = [a for a in street.actions if a.type is ActionType.BET]
            # A bet may only be the first money-in action of its street.
            for bet in bets:
                earlier = street.actions[: street.actions.index(bet)]
                assert not any(a.type.is_aggressive for a in earlier)


# ------------------------------------------------------------- structure


def test_streets_are_recorded_in_order():
    result = run("station", "station", seed=11)
    order = [r.street for r in result.history.streets]
    assert order == list(dict.fromkeys(order))
    assert order[0] is Street.PREFLOP


def test_board_is_dealt_progressively():
    for seed in range(40):
        result = run("station", "station", seed=seed)
        if len(result.history.streets) == 4:
            sizes = [len(r.cards) for r in result.history.streets]
            assert sizes == [0, 3, 1, 1]
            assert len(result.history.board) == 5
            return
    pytest.fail("no hand reached the river")


def test_a_fold_ends_the_hand_without_showdown():
    for seed in range(30):
        result = run("nit", "nit", seed=seed)
        if not result.showdown:
            assert len(result.winners) == 1
            assert result.hands == {}
            return
    pytest.fail("no hand ended in a fold")


def test_a_showdown_evaluates_both_hands():
    for seed in range(60):
        result = run("station", "station", seed=seed)
        if result.showdown:
            assert len(result.hands) == 2
            best = max(r.score for r in result.hands.values())
            assert all(result.hands[w].score == best for w in result.winners)
            return
    pytest.fail("no hand reached showdown")


def test_hole_cards_are_distinct_across_players_and_board():
    result = run(seed=5)
    seen = list(result.history.board)
    for seat in result.history.seats:
        seen.extend(seat.hole_cards)
    assert len(seen) == len(set(seen))


# ----------------------------------------------------------- configuration


def test_a_pinned_board_is_used():
    board = parse_cards("AhKhQh2c3d")
    result = run("station", "station", seed=4, board=board)
    dealt = result.history.board
    assert dealt == board[: len(dealt)]


def test_a_pinned_board_must_be_five_cards():
    with pytest.raises(ValueError, match="needs 5 cards"):
        run(seed=1, board=parse_cards("AhKhQh"))


def test_pinned_board_cards_are_never_dealt_as_hole_cards():
    board = parse_cards("AhKhQh2c3d")
    result = run("station", "station", seed=4, board=board)
    for seat in result.history.seats:
        assert not set(seat.hole_cards) & set(board)


def test_hero_defaults_to_the_big_blind():
    assert run(seed=2).history.hero == "bb"


def test_hero_can_be_chosen():
    assert run(seed=2, hero="sb").history.hero == "sb"


def test_blind_sizes_are_respected():
    result = run(seed=2, small_blind=1.0, big_blind=2.0)
    assert result.history.small_blind == 1.0
    assert result.history.big_blind == 2.0
    assert result.pot >= 3.0


def test_the_same_seed_reproduces_the_hand():
    a = run("lag", "station", seed=21)
    b = run("lag", "station", seed=21)
    assert a.history.model_dump_json() == b.history.model_dump_json()


def test_short_stacks_play_out_legally():
    rng = random.Random(9)
    seats = [
        Seat(make_player("sb", "maniac", rng=rng, iterations=120), Position.SB, 4.0),
        Seat(make_player("bb", "maniac", rng=rng, iterations=120), Position.BB, 4.0),
    ]
    result = play_hand(seats, rng=rng)
    assert all(p.stack >= 0 for p in result.final_state.players)
    assert result.winners


# -------------------------------------------------------------- guardrails


def test_a_third_seat_is_now_allowed():
    rng = random.Random(1)
    seats = table() + [
        Seat(make_player("utg", "tag", rng=rng, iterations=120),
             Position.UTG, 100.0)
    ]
    assert play_hand(seats, rng=rng).pot > 0


def test_one_seat_is_rejected():
    with pytest.raises(ValueError, match="between 2 and 9 seats"):
        play_hand(table()[:1], rng=random.Random(1))


def test_too_many_seats_are_rejected():
    rng = random.Random(1)
    positions = [
        Position.SB, Position.BB, Position.UTG, Position.UTG1, Position.MP,
        Position.LJ, Position.HJ, Position.CO, Position.BTN,
    ]
    seats = [
        Seat(make_player(f"p{i}", "tag", rng=rng), pos, 100.0)
        for i, pos in enumerate(positions)
    ]
    # Nine is the ceiling; a tenth has no seat to sit in.
    seats.append(Seat(make_player("extra", "tag", rng=rng), Position.BTN, 100.0))
    with pytest.raises(ValueError, match="between 2 and 9 seats"):
        play_hand(seats, rng=rng)


def test_the_blinds_must_be_seated():
    rng = random.Random(1)
    seats = [
        Seat(make_player("btn", "tag", rng=rng), Position.BTN, 100.0),
        Seat(make_player("bb", "tag", rng=rng), Position.BB, 100.0),
    ]
    with pytest.raises(ValueError, match="blinds must be seated"):
        play_hand(seats, rng=rng)


def test_duplicate_positions_are_rejected():
    rng = random.Random(1)
    seats = [
        Seat(make_player("a", "tag", rng=rng), Position.SB, 100.0),
        Seat(make_player("b", "tag", rng=rng), Position.SB, 100.0),
    ]
    with pytest.raises(ValueError, match="share a position"):
        play_hand(seats, rng=rng)


def test_result_summary_reads_cleanly():
    text = str(run(seed=3))
    assert "wins" in text
    assert "showdown" in text
