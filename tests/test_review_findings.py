"""Regressions for an external code review of the pushed repository.

Every test here corresponds to a defect the review identified and that was
reproduced before being fixed. They live together because their shared lesson
is that each bug produced a *plausible* number rather than an error, and the
existing suite passed throughout.
"""

import collections
import random

import pytest

from poker_coach.calculations.equity import _draw_joint, equity
from poker_coach.calculations.ranges import Range
from poker_coach.coaching.analysis import _is_terminal_call, analyze
from poker_coach.domain import (
    Action,
    ActionType,
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)


# ---------------------------------------------------- joint range sampling


def test_multiway_ranges_are_sampled_uniformly_over_joint_assignments():
    """Sequential conditioning over-weights whichever range blocks more.

    opp1 in {AsKs, 2c3c}, opp2 in {AsQs, AsJs, 2c4c}. AsKs blocks As, leaving
    opp2 exactly one combo; 2c3c blocks 2c, leaving two. Three legal joint
    assignments, so each should appear a third of the time. The sequential
    sampler produced 0.496 / 0.252 / 0.252.
    """

    p1 = Range("AsKs, 2c3c").combos()
    p2 = Range("AsQs, AsJs, 2c4c").combos()

    rng = random.Random(0)
    counts: collections.Counter = collections.Counter()
    accepted = 0
    for _ in range(30_000):
        _, hands = _draw_joint([p1, p2], set(), rng)
        if hands is None:
            continue
        accepted += 1
        counts[tuple("".join(map(str, h)) for h in hands)] += 1

    assert len(counts) == 3
    for share in (v / accepted for v in counts.values()):
        assert share == pytest.approx(1 / 3, abs=0.02)


def test_equity_does_not_depend_on_the_order_ranges_are_supplied():
    forwards = equity(
        "7h7d",
        [Range("AsKs, 2c3c"), Range("AsQs, AsJs, 2c4c")],
        board="9c4d2h",
        iterations=20_000,
        rng=random.Random(5),
    )
    backwards = equity(
        "7h7d",
        [Range("AsQs, AsJs, 2c4c"), Range("AsKs, 2c3c")],
        board="9c4d2h",
        iterations=20_000,
        rng=random.Random(5),
    )
    gap = abs(forwards.equity - backwards.equity)
    assert gap < 2 * forwards.margin_of_error


def test_pinned_combos_still_work_alongside_sampled_ranges():
    # A pinned combo lives in the dead set already; it must not be rejected
    # as colliding with itself.
    result = equity(
        "7h7d", ["AsKs", Range("QQ")], board="9c4d2h",
        iterations=2000, rng=random.Random(1),
    )
    assert result.samples > 1500


# ------------------------------------------------------- terminal call EV


def multiway_state(live_stack: float) -> HandState:
    return HandState(
        players=[
            PlayerState(name="hero", position=Position.BTN, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
            PlayerState(name="allin", position=Position.SB, stack=0.0,
                        is_all_in=True, committed_this_street=20.0),
            PlayerState(name="live", position=Position.BB, stack=live_stack,
                        committed_this_street=20.0),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=10.0,
    )


def test_one_all_in_opponent_does_not_make_a_call_terminal():
    """`min(...) <= 0` marked any spot with an all-in player as terminal.

    That promoted the EV figure from "upper bound" to "exact" while a live
    opponent could still bet future streets.
    """

    state = multiway_state(live_stack=80.0)
    assert _is_terminal_call(state, "hero", state.amount_to_call("hero")) is False


def test_a_call_is_terminal_once_every_opponent_is_all_in():
    state = multiway_state(live_stack=0.0)
    assert _is_terminal_call(state, "hero", state.amount_to_call("hero")) is True


# ------------------------------------------------------------ MDF vs raise


def raise_spot() -> HandState:
    """Pot 10; hero bets 5; villain raises to 15. Hero faces 10 more."""

    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.BTN, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
            PlayerState(name="villain", position=Position.BB, stack=100.0),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=10.0,
    )
    state = state.apply(
        Action(actor="hero", type=ActionType.BET, amount=5.0, street=Street.FLOP)
    )
    return state.apply(
        Action(actor="villain", type=ActionType.RAISE, amount=15.0, street=Street.FLOP)
    )


def test_mdf_against_a_raise_uses_what_the_raiser_risked():
    """Villain risked 15 to win the 15 already out there, so alpha is 50%.

    Deriving it from hero's 10-chip call gave 33.3% — the figure for a bet
    that was never made.
    """

    analysis = analyze(raise_spot(), villain_range="random",
                       iterations=300, rng=random.Random(1))
    assert analysis.total_pot == pytest.approx(30.0)
    assert analysis.to_call == pytest.approx(10.0)
    assert analysis.alpha == pytest.approx(0.5)
    assert analysis.mdf == pytest.approx(0.5)


def test_mdf_against_a_plain_bet_is_unchanged():
    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.BTN, stack=97.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
            PlayerState(name="villain", position=Position.BB, stack=97.0),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=6.0,
    ).apply(
        Action(actor="villain", type=ActionType.BET, amount=6.0, street=Street.FLOP)
    )
    analysis = analyze(state, villain_range="random",
                       iterations=300, rng=random.Random(1))
    assert analysis.alpha == pytest.approx(0.5)  # pot-sized bet


def test_posted_blinds_produce_no_bluff_frequency():
    """Nobody chose to post a blind, so a break-even bluff rate is meaningless."""

    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.SB, stack=99.5,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True,
                        committed_this_street=0.5),
            PlayerState(name="villain", position=Position.BB, stack=99.0,
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    )
    analysis = analyze(state, villain_range="random",
                       iterations=300, rng=random.Random(1))
    assert analysis.to_call > 0
    assert analysis.mdf is None
    assert analysis.alpha is None


def test_a_recorded_preflop_raise_produces_a_bluff_frequency():
    """Built through apply() so the raiser's increment is on record."""

    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.BB, stack=99.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True,
                        committed_this_street=1.0),
            PlayerState(name="villain", position=Position.SB, stack=99.5,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=0.5),
        ],
        street=Street.PREFLOP,
    ).apply(
        Action(actor="villain", type=ActionType.RAISE, amount=3.0,
               street=Street.PREFLOP),
        strict=True,
    )
    analysis = analyze(state, villain_range="random",
                       iterations=300, rng=random.Random(1))
    # SB added 2.5 to win the 1.5 already posted: 2.5 / 4.0.
    assert analysis.alpha == pytest.approx(0.625)


def test_an_unrecorded_preflop_bet_reports_no_bluff_frequency():
    """The blinds make the raiser's increment unknowable; nothing is claimed."""

    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.BB, stack=99.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True,
                        committed_this_street=1.0),
            PlayerState(name="villain", position=Position.SB, stack=96.0,
                        committed_this_street=4.0),
        ],
        street=Street.PREFLOP,
    )
    analysis = analyze(state, villain_range="random",
                       iterations=300, rng=random.Random(1))
    assert analysis.to_call > 0
    assert analysis.alpha is None
    assert analysis.mdf is None


# --------------------------------------------------------- multiway guard


def test_multiway_analysis_is_refused_rather_than_answered_wrongly():
    """One range cannot describe two opponents; a wrong number is worse than none."""

    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.BTN, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
            PlayerState(name="v1", position=Position.SB, stack=100.0,
                        committed_this_street=10.0),
            PlayerState(name="v2", position=Position.BB, stack=100.0,
                        committed_this_street=10.0),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=15.0,
    )
    with pytest.raises(NotImplementedError, match="multiway analysis"):
        analyze(state, villain_range="22+", iterations=200, rng=random.Random(3))


def test_a_folded_third_player_leaves_a_analysable_heads_up_spot():
    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.BTN, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
            PlayerState(name="folded", position=Position.SB, stack=100.0,
                        has_folded=True),
            PlayerState(name="v2", position=Position.BB, stack=100.0,
                        committed_this_street=10.0),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=15.0,
    )
    analysis = analyze(state, villain_range="22+", iterations=200,
                       rng=random.Random(3))
    assert analysis.hero_cards == "AsKs"


# ------------------------------------------------------- betting legality


def preflop_state() -> HandState:
    return HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")),
                        committed_this_street=0.5),
            PlayerState(name="b", position=Position.BB, stack=100.0,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    )


def test_an_action_tagged_with_the_wrong_street_is_rejected():
    with pytest.raises(ValueError, match="tagged river but the hand is on"):
        preflop_state().apply(
            Action(actor="a", type=ActionType.RAISE, amount=3.0, street=Street.RIVER)
        )


def test_an_undersized_raise_is_rejected():
    """Raise to 3, then to 3.5: an increment of 0.5 against a full raise of 2."""

    state = preflop_state().apply(
        Action(actor="a", type=ActionType.RAISE, amount=3.0, street=Street.PREFLOP)
    )
    with pytest.raises(ValueError, match="not a legal size"):
        state.apply(
            Action(actor="b", type=ActionType.RAISE, amount=3.5, street=Street.PREFLOP)
        )


def test_a_full_raise_is_accepted():
    state = preflop_state().apply(
        Action(actor="a", type=ActionType.RAISE, amount=3.0, street=Street.PREFLOP)
    )
    state.apply(
        Action(actor="b", type=ActionType.RAISE, amount=5.0, street=Street.PREFLOP)
    )


def test_an_all_in_below_the_minimum_raise_is_still_legal():
    """A short stack may always move all in, however small the increment."""

    # Built through apply() so the raise is on record: a raises the blind of 1
    # up to 6, a full increment of 5, so the next full raise would be to 11.
    # b has 7 behind on top of its 1 and can only reach 8 — short of a full
    # raise, and legal anyway because it is all-in.
    state = HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")),
                        committed_this_street=0.5),
            PlayerState(name="b", position=Position.BB, stack=7.0,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    ).apply(
        Action(actor="a", type=ActionType.RAISE, amount=6.0, street=Street.PREFLOP)
    )

    assert state.betting_round().last_full_raise == pytest.approx(5.0)
    assert state.betting_round().min_raise_to(1.0, 7.0) == pytest.approx(8.0)

    after = state.apply(
        Action(actor="b", type=ActionType.RAISE, amount=8.0, street=Street.PREFLOP)
    )
    assert after.player("b").is_all_in


def test_without_history_the_minimum_raise_falls_back_to_the_big_blind():
    """A stated boundary, not an oversight.

    The last full raise cannot be recovered from a state built directly, so the
    floor degrades to the big blind. That is permissive rather than wrong — it
    accepts some raises a full history would reject, and never the reverse.
    """

    state = HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")),
                        committed_this_street=6.0),
            PlayerState(name="b", position=Position.BB, stack=100.0,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    )
    round_ = state.betting_round()
    assert round_.has_history is False
    assert round_.last_full_raise == pytest.approx(state.big_blind)


def test_a_street_cannot_be_left_with_a_bet_outstanding():
    state = preflop_state().apply(
        Action(actor="a", type=ActionType.RAISE, amount=30.0, street=Street.PREFLOP)
    )
    with pytest.raises(ValueError, match="have not matched the bet"):
        state.advance_street(parse_cards("2c7s9d"))


def test_a_street_may_be_left_once_the_bet_is_matched():
    state = preflop_state().apply(
        Action(actor="a", type=ActionType.RAISE, amount=30.0, street=Street.PREFLOP)
    ).apply(
        Action(actor="b", type=ActionType.CALL, amount=30.0, street=Street.PREFLOP)
    )
    assert state.advance_street(parse_cards("2c7s9d")).street is Street.FLOP


def test_acting_out_of_turn_is_rejected_in_strict_mode():
    state = preflop_state().apply(
        Action(actor="a", type=ActionType.RAISE, amount=3.0, street=Street.PREFLOP),
        strict=True,
    )
    with pytest.raises(ValueError, match="it is b's turn"):
        state.apply(
            Action(actor="a", type=ActionType.RAISE, amount=9.0, street=Street.PREFLOP),
            strict=True,
        )


def test_turn_order_is_not_enforced_without_history():
    """Directly-built states cannot supply a turn; the boundary is documented.

    `b` acting first is out of order under strict rules, but a state carrying
    no action history has no way to know that.
    """

    state = HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")),
                        committed_this_street=0.5),
            PlayerState(name="b", position=Position.BB, stack=100.0,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    )
    assert state.betting_round().has_history is False
    state.apply(
        Action(actor="b", type=ActionType.RAISE, amount=3.0, street=Street.PREFLOP)
    )


def test_legal_actions_is_the_source_of_truth():
    state = preflop_state()
    options = {a.type: a for a in state.legal_actions("a")}
    assert set(options) == {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
    assert options[ActionType.CALL].min_amount == pytest.approx(1.0)
    assert options[ActionType.RAISE].min_amount == pytest.approx(2.0)
    assert options[ActionType.RAISE].max_amount == pytest.approx(100.5)


def test_legal_actions_offers_a_check_when_nothing_is_owed():
    state = HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs"))),
            PlayerState(name="b", position=Position.BB, stack=100.0,
                        hole_cards=tuple(parse_cards("7h7d"))),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=10.0,
    )
    types = {a.type for a in state.legal_actions("a")}
    # Folding is offered too: legal whenever it is your turn, merely awful when
    # checking is free.
    assert types == {ActionType.CHECK, ActionType.BET, ActionType.FOLD}


def test_every_listed_legal_action_is_actually_accepted():
    state = preflop_state()
    for option in state.legal_actions("a"):
        state.apply(
            Action(actor="a", type=option.type, amount=option.min_amount,
                   street=Street.PREFLOP)
        )


def test_bundled_example_hands_replay_under_strict_rules():
    import glob

    from poker_coach.domain import HandHistory

    paths = sorted(glob.glob("examples/hands/*.json"))
    assert paths
    for path in paths:
        HandHistory.from_json_file(path).final_state()  # strict by default


def test_simulated_hands_replay_under_strict_rules():
    """The runner enforces legality, so its output must survive re-checking."""

    import random as _random

    from poker_coach.domain import HandHistory
    from poker_coach.players import STYLES, Seat, make_player, play_hand

    seating = [Position.SB, Position.BB, Position.UTG, Position.HJ]
    styles = list(STYLES)

    for i in range(24):
        n = 2 + (i % 3)
        rng = _random.Random(4000 + i)
        seats = [
            Seat(
                make_player(f"p{j}", styles[(i + j) % len(styles)], rng=rng,
                            iterations=60),
                seating[j],
                100.0 if j % 2 else 25.0,   # mixed stacks force all-ins
            )
            for j in range(n)
        ]
        result = play_hand(seats, rng=rng)
        replayed = result.history.final_state()   # strict by default
        assert replayed.total_pot == pytest.approx(result.pot)


def test_a_player_never_proposes_an_action_the_rules_reject():
    import random as _random

    from poker_coach.players import STYLES, make_player

    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.SB, stack=40.0,
                        hole_cards=tuple(parse_cards("AhAd")),
                        committed_this_street=6.0),
            PlayerState(name="villain", position=Position.BB, stack=9.0,
                        hole_cards=tuple(parse_cards("KsKc")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    )
    for label in STYLES:
        player = make_player("villain", label, rng=_random.Random(2), iterations=60)
        action = player.act(state)
        state.apply(action)   # raises if the policy proposed something illegal


# ============================ second review round ============================


def short_stack_underraise() -> HandState:
    """a opens 1->6 (full raise 5); b is all-in for 8, an under-raise."""

    state = HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")),
                        committed_this_street=0.5),
            PlayerState(name="b", position=Position.BB, stack=7.0,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    ).apply(
        Action(actor="a", type=ActionType.RAISE, amount=6.0, street=Street.PREFLOP),
        strict=True,
    )
    return state.apply(
        Action(actor="b", type=ActionType.RAISE, amount=8.0, street=Street.PREFLOP),
        strict=True,
    )


def test_an_all_in_underraise_does_not_reopen_the_betting():
    """The early `if is_all_in: return` skipped this check entirely."""

    state = short_stack_underraise()
    assert state.player("b").is_all_in
    assert ActionType.RAISE not in {o.type for o in state.legal_actions("a")}
    with pytest.raises(ValueError, match="may not raise here"):
        state.apply(
            Action(actor="a", type=ActionType.RAISE, amount=100.5,
                   street=Street.PREFLOP),
            strict=True,
        )


def test_calling_remains_legal_after_an_underraise():
    state = short_stack_underraise()
    state.apply(
        Action(actor="a", type=ActionType.CALL, amount=8.0, street=Street.PREFLOP),
        strict=True,
    )


def unbet_flop() -> HandState:
    return HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs"))),
            PlayerState(name="b", position=Position.BB, stack=100.0,
                        hole_cards=tuple(parse_cards("7h7d"))),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=10.0,
    )


@pytest.mark.parametrize(
    "action",
    [
        Action(actor="a", type=ActionType.BET, amount=0.1, street=Street.FLOP),
        Action(actor="a", type=ActionType.RAISE, amount=5.0, street=Street.FLOP),
        Action(actor="a", type=ActionType.POST_BLIND, amount=1.0, street=Street.FLOP),
    ],
    ids=["bet-below-one-big-blind", "raise-into-an-unopened-pot", "blind-mid-street"],
)
def test_actions_absent_from_legal_actions_are_rejected(action):
    """`apply()` now validates against the listed options rather than a
    parallel set of hand-written checks that drifted from them."""

    state = unbet_flop()
    assert action.type not in {
        o.type for o in state.legal_actions("a")
    } or not any(
        o.permits(action.amount)
        for o in state.legal_actions("a")
        if o.type is action.type
    )
    with pytest.raises(ValueError, match="not a legal|may not"):
        state.apply(action)


def test_a_street_cannot_advance_before_anyone_acts():
    """`unmatched()` is empty on an unbet street, so it could not catch this.

    Uses a state reached through play, since a wholly synthetic one carries no
    action anywhere and cannot know whether the round has closed.
    """

    state = HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs")),
                        committed_this_street=0.5),
            PlayerState(name="b", position=Position.BB, stack=100.0,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    ).apply(
        Action(actor="a", type=ActionType.CALL, amount=1.0, street=Street.PREFLOP),
        strict=True,
    ).apply(
        Action(actor="b", type=ActionType.CHECK, street=Street.PREFLOP), strict=True
    ).advance_street(parse_cards("Qs2s9c"))

    assert state.betting_round().unmatched(state) == []
    with pytest.raises(ValueError, match="betting round is not complete"):
        state.advance_street(parse_cards("4d"))


def test_a_street_cannot_advance_after_only_one_check():
    state = unbet_flop().apply(
        Action(actor="b", type=ActionType.CHECK, street=Street.FLOP), strict=True
    )
    with pytest.raises(ValueError, match="betting round is not complete"):
        state.advance_street(parse_cards("4d"))


def test_a_street_advances_once_both_have_checked():
    state = unbet_flop().apply(
        Action(actor="b", type=ActionType.CHECK, street=Street.FLOP), strict=True
    ).apply(
        Action(actor="a", type=ActionType.CHECK, street=Street.FLOP), strict=True
    )
    assert state.advance_street(parse_cards("4d")).street is Street.TURN


def test_mdf_uses_the_increment_when_the_raiser_was_already_committed():
    """SB posts 0.5 then raises to 3: it added 2.5 to win 1.5, so 62.5%.

    Treating the whole 3-chip commitment as the wager gave 75%.
    """

    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.BB, stack=99.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True,
                        committed_this_street=1.0),
            PlayerState(name="villain", position=Position.SB, stack=99.5,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=0.5),
        ],
        street=Street.PREFLOP,
    ).apply(
        Action(actor="villain", type=ActionType.RAISE, amount=3.0,
               street=Street.PREFLOP),
        strict=True,
    )
    betting = state.betting_round()
    assert betting.last_wager == pytest.approx(2.5)
    assert betting.pot_before_aggression == pytest.approx(1.5)

    analysis = analyze(state, villain_range="random", iterations=200,
                       rng=random.Random(1))
    assert analysis.alpha == pytest.approx(0.625)
    assert analysis.mdf == pytest.approx(0.375)


def test_mdf_uses_the_increment_for_a_three_bet():
    """a opens to 6, b three-bets to 20: b added 19 to win the 7 in front."""

    state = HandState(
        players=[
            PlayerState(name="a", position=Position.SB, stack=99.5,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True,
                        committed_this_street=0.5),
            PlayerState(name="b", position=Position.BB, stack=99.0,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    ).apply(
        Action(actor="a", type=ActionType.RAISE, amount=6.0, street=Street.PREFLOP),
        strict=True,
    ).apply(
        Action(actor="b", type=ActionType.RAISE, amount=20.0, street=Street.PREFLOP),
        strict=True,
    )
    betting = state.betting_round()
    assert betting.last_wager == pytest.approx(19.0)
    assert betting.pot_before_aggression == pytest.approx(7.0)


def all_in_opponent_flop() -> HandState:
    return HandState(
        players=[
            PlayerState(name="hero", position=Position.SB, stack=80.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
            PlayerState(name="allin", position=Position.BB, stack=0.0,
                        hole_cards=tuple(parse_cards("7h7d")), is_all_in=True),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=40.0,
    )


def test_no_betting_into_a_pot_nobody_can_contest():
    """Aggression needs an opponent able to answer it."""

    state = all_in_opponent_flop()
    assert ActionType.BET not in {o.type for o in state.legal_actions("hero")}
    with pytest.raises(ValueError, match="may not bet here"):
        state.apply(
            Action(actor="hero", type=ActionType.BET, amount=10.0,
                   street=Street.FLOP)
        )


def test_a_street_with_nobody_to_act_records_no_action():
    import random as _random

    from poker_coach.players import make_player
    from poker_coach.players.table import _play_street

    record: list[Action] = []
    state = all_in_opponent_flop()
    players = {"hero": make_player("hero", "maniac", rng=_random.Random(1),
                                   iterations=40)}
    after = _play_street(state, players, record)
    assert record == []
    assert after.total_pot == state.total_pot


def test_analyse_refuses_a_folded_hero():
    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.SB, stack=80.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True,
                        has_folded=True),
            PlayerState(name="v", position=Position.BB, stack=80.0),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=40.0,
    )
    with pytest.raises(ValueError, match="has folded"):
        analyze(state, villain_range="22+", iterations=100, rng=random.Random(1))


def test_analyse_refuses_a_finished_hand():
    state = HandState(
        players=[
            PlayerState(name="hero", position=Position.SB, stack=80.0,
                        hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
            PlayerState(name="v", position=Position.BB, stack=80.0,
                        has_folded=True),
        ],
        board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=40.0,
    )
    with pytest.raises(ValueError, match="the hand is over"):
        analyze(state, villain_range="22+", iterations=100, rng=random.Random(1))


# ============================= third review round ============================


def three_handed_preflop() -> HandState:
    """BTN, SB, BB with blinds posted. Preflop order is BTN, SB, BB."""

    return HandState(
        players=[
            PlayerState(name="btn", position=Position.BTN, stack=100.0,
                        hole_cards=tuple(parse_cards("AsKs"))),
            PlayerState(name="sb", position=Position.SB, stack=99.5,
                        hole_cards=tuple(parse_cards("7h7d")),
                        committed_this_street=0.5),
            PlayerState(name="bb", position=Position.BB, stack=99.0,
                        hole_cards=tuple(parse_cards("QhQd")),
                        committed_this_street=1.0),
        ],
        street=Street.PREFLOP,
    )


def test_action_continues_clockwise_after_a_raise():
    """BTN calls, SB raises — BB must act before BTN, not after.

    Scanning from the top of the street order returned BTN, because BTN owed
    chips again. Heads-up hid this: there is only one candidate after the
    aggressor.
    """

    state = three_handed_preflop().apply(
        Action(actor="btn", type=ActionType.CALL, amount=1.0,
               street=Street.PREFLOP),
        strict=True,
    ).apply(
        Action(actor="sb", type=ActionType.RAISE, amount=4.0,
               street=Street.PREFLOP),
        strict=True,
    )
    assert state.betting_round().next_actor(state) == "bb"


def test_the_cursor_wraps_back_round_to_the_original_caller():
    state = three_handed_preflop().apply(
        Action(actor="btn", type=ActionType.CALL, amount=1.0,
               street=Street.PREFLOP),
        strict=True,
    ).apply(
        Action(actor="sb", type=ActionType.RAISE, amount=4.0,
               street=Street.PREFLOP),
        strict=True,
    ).apply(
        Action(actor="bb", type=ActionType.CALL, amount=4.0,
               street=Street.PREFLOP),
        strict=True,
    )
    assert state.betting_round().next_actor(state) == "btn"

    closed = state.apply(
        Action(actor="btn", type=ActionType.CALL, amount=4.0,
               street=Street.PREFLOP),
        strict=True,
    )
    assert closed.betting_round().is_complete(closed)


def test_acting_out_of_turn_after_a_raise_is_rejected():
    state = three_handed_preflop().apply(
        Action(actor="btn", type=ActionType.CALL, amount=1.0,
               street=Street.PREFLOP),
        strict=True,
    ).apply(
        Action(actor="sb", type=ActionType.RAISE, amount=4.0,
               street=Street.PREFLOP),
        strict=True,
    )
    with pytest.raises(ValueError, match="it is bb's turn"):
        state.apply(
            Action(actor="btn", type=ActionType.CALL, amount=4.0,
                   street=Street.PREFLOP),
            strict=True,
        )


def test_a_folded_callers_chips_stay_in_the_reconstructed_pot():
    """BTN raises 3, SB calls, BB raises 10, BTN folds, SB raises 30.

    The pot in front of SB's raise is 3 + 3 + 10 = 16. Replaying BTN's fold as
    a commitment of zero erased its 3, giving 13 and an alpha of 67.5% instead
    of 62.79%.
    """

    state = three_handed_preflop()
    for actor, kind, amount in [
        ("btn", ActionType.RAISE, 3.0),
        ("sb", ActionType.CALL, 3.0),
        ("bb", ActionType.RAISE, 10.0),
        ("btn", ActionType.FOLD, 0.0),
        ("sb", ActionType.RAISE, 30.0),
    ]:
        state = state.apply(
            Action(actor=actor, type=kind, amount=amount, street=Street.PREFLOP)
        )

    betting = state.betting_round()
    assert betting.pot_before_aggression == pytest.approx(16.0)
    assert betting.last_wager == pytest.approx(27.0)

    from poker_coach.calculations.pot_odds import bluff_success_threshold

    assert bluff_success_threshold(
        betting.pot_before_aggression, betting.last_wager
    ) == pytest.approx(0.6279, abs=1e-4)


def test_a_folded_aggressors_chips_stay_in_the_reconstructed_pot():
    """The player who folds is the one who bet earlier in the street."""

    state = three_handed_preflop()
    for actor, kind, amount in [
        ("btn", ActionType.RAISE, 6.0),
        ("sb", ActionType.CALL, 6.0),
        ("bb", ActionType.RAISE, 20.0),
        ("btn", ActionType.FOLD, 0.0),     # folds having put in 6
        ("sb", ActionType.RAISE, 60.0),
    ]:
        state = state.apply(
            Action(actor=actor, type=kind, amount=amount, street=Street.PREFLOP)
        )
    betting = state.betting_round()
    # 6 (folded BTN) + 6 (SB) + 20 (BB) = 32 in front of SB's raise.
    assert betting.pot_before_aggression == pytest.approx(32.0)
    assert betting.last_wager == pytest.approx(54.0)


def test_a_check_does_not_erase_a_posted_blind():
    """The big blind's option: checking must not zero its posted chip."""

    state = three_handed_preflop().apply(
        Action(actor="btn", type=ActionType.CALL, amount=1.0,
               street=Street.PREFLOP),
        strict=True,
    ).apply(
        Action(actor="sb", type=ActionType.CALL, amount=1.0,
               street=Street.PREFLOP),
        strict=True,
    ).apply(
        Action(actor="bb", type=ActionType.CHECK, street=Street.PREFLOP),
        strict=True,
    )
    assert state.betting_round().committed_when_acted["bb"] == pytest.approx(1.0)


# --- an order oracle written from the rules, not from the implementation ---


def expected_order(street: Street, seats: list[Position]) -> list[Position]:
    """Independent statement of hold'em action order.

    Deliberately not derived from `action_order`: two copies of the same
    mistake agree with each other, which is exactly how the multiway ordering
    bug survived a green 1500-hand replay suite.
    """

    clockwise = [
        Position.SB, Position.BB, Position.UTG, Position.UTG1, Position.MP,
        Position.LJ, Position.HJ, Position.CO, Position.BTN,
    ]
    seated = [p for p in clockwise if p in seats]

    if street is Street.PREFLOP:
        # Opens left of the big blind; the blinds close.
        cut = seated.index(Position.BB) + 1
        return seated[cut:] + seated[:cut]

    # The button closes; heads-up the small blind holds it.
    if Position.BTN in seats:
        button = Position.BTN
    elif len(seated) == 2:
        button = seated[0]
    else:
        button = seated[-1]
    cut = seated.index(button) + 1
    return seated[cut:] + seated[:cut]


@pytest.mark.parametrize(
    "seats",
    [
        [Position.SB, Position.BB],
        [Position.BTN, Position.BB],
        [Position.SB, Position.BB, Position.BTN],
        [Position.SB, Position.BB, Position.CO],
        [Position.SB, Position.BB, Position.UTG, Position.BTN],
        [Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO,
         Position.BTN],
    ],
    ids=lambda s: f"{len(s)}-handed",
)
@pytest.mark.parametrize(
    "street", [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
)
def test_action_order_matches_an_independent_oracle(seats, street):
    from poker_coach.domain.betting import action_order

    assert action_order(street, set(seats)) == expected_order(street, seats)


def test_three_handed_preflop_order_is_button_then_blinds():
    """Spelled out rather than derived, as a sanity anchor for the oracle."""

    from poker_coach.domain.betting import action_order

    seats = {Position.SB, Position.BB, Position.BTN}
    assert action_order(Street.PREFLOP, seats) == [
        Position.BTN, Position.SB, Position.BB
    ]
    assert action_order(Street.FLOP, seats) == [
        Position.SB, Position.BB, Position.BTN
    ]
