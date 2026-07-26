"""Range conditioning as the practice session uses it.

`tests/players/test_conditioning.py` checks the inference. This checks the
wiring: that the narrowed range is what the coach was actually handed, that it
keeps narrowing as the hand goes on, that the old fixed-range behaviour is still
reachable, and that the two failure modes — an emptied range, and an exact
equity measured against an inexact range — are stated rather than hidden.
"""

import random

from poker_coach.coaching.analysis import RangeConditioning
from poker_coach.domain.action import Action
from poker_coach.domain.enums import ActionType, Position, Street
from poker_coach.domain.cards import parse_cards
from poker_coach.domain.state import HandState, PlayerState
from poker_coach.practice.feedback import Feedback
from poker_coach.practice.session import (
    DEFAULT_RANGES,
    HeroTurn,
    ObservedOpponent,
    PracticeConfig,
    PracticeSession,
    _RangeTracker,
)
from poker_coach.players import make_player


def passive_session(**kwargs) -> tuple[PracticeSession, list[HeroTurn]]:
    """A session whose hero calls or checks, so hands reach later streets."""

    seen: list[HeroTurn] = []

    def on_turn(turn: HeroTurn) -> Action:
        seen.append(turn)
        preferred = (
            [o for o in turn.legal if o.type is ActionType.CHECK]
            or [o for o in turn.legal if o.type is ActionType.CALL]
            or turn.legal
        )
        option = preferred[0]
        return Action(actor="you", type=option.type,
                      amount=option.min_amount, street=turn.state.street)

    config = PracticeConfig(
        iterations=400, opponent_iterations=60,
        conditioning_iterations=120, **kwargs,
    )
    session = PracticeSession(
        config=config, rng=random.Random(3), on_turn=on_turn, coach=False,
    )
    return session, seen


# ------------------------------------------------------------- the wiring


def test_the_coach_is_handed_the_narrowed_range():
    session, seen = passive_session()
    session.play_hand(seed=11)

    after_action = [t for t in seen if t.analysis.range_assumption.narrowing]
    assert after_action, "no decision followed an opponent action"

    for turn in after_action:
        assumption = turn.analysis.range_assumption
        assert assumption.conditioning is RangeConditioning.ACTION_CONDITIONED
        assert assumption.combos < assumption.combos_before
        # The equity really was measured against the posterior, not the prior.
        assert assumption.notation != "random"
        assert len(assumption.range.combos(dead=())) >= assumption.combos


def test_the_range_keeps_narrowing_as_the_hand_goes_on():
    """Each action conditions what the last one left, not the original prior."""

    session, seen = passive_session()
    for seed in (11, 23, 37, 41):
        session.play_hand(seed=seed)

    by_hand: dict[int, list] = {}
    for turn, record in zip(seen, session.records):
        by_hand.setdefault(record.hand_number, []).append(
            turn.analysis.range_assumption
        )

    multi = [v for v in by_hand.values()
             if len([a for a in v if a.narrowing]) >= 2]
    assert multi, "no hand had two decisions after an opponent action"

    for assumptions in multi:
        conditioned = [a for a in assumptions if a.narrowing]
        counts = [a.combos for a in conditioned]
        assert counts == sorted(counts, reverse=True), counts
        # The denominator is the size before *any* conditioning, so the
        # narrowing figure stays comparable across the hand.
        assert len({a.combos_before for a in conditioned}) == 1


def test_before_the_opponent_acts_the_range_is_pre_action():
    """Hand one puts hero on the button, which acts first preflop. There is no
    action to condition on, so a pre-action range is not a fallback — it is the
    correct answer, and it has to say which of the two it is.
    """

    session, seen = passive_session()
    session.play_hand(seed=11)

    first = seen[0].analysis.range_assumption
    assert seen[0].analysis.street == "preflop"
    assert seen[0].analysis.hero_position == "SB"
    assert first.narrowing is None
    assert first.conditioning is RangeConditioning.PRE_ACTION
    assert "has not acted yet" in first.provenance


def steps_in(assumption) -> int:
    """How many actions have been folded into a range, read off its label."""

    if assumption.narrowing is None:
        return 0
    return assumption.display.count("then") + 1


def test_a_read_never_carries_across_hands():
    """The observer and the tracker are both built per hand, so a read made in
    one hand cannot leak into the next — where it would be about cards that
    have already been mucked.
    """

    session, seen = passive_session()
    for seed in (11, 23, 37, 41):
        session.play_hand(seed=seed)

    by_hand: dict[int, list[int]] = {}
    for turn, record in zip(seen, session.records):
        by_hand.setdefault(record.hand_number, []).append(
            steps_in(turn.analysis.range_assumption)
        )

    assert len(by_hand) >= 2, "need more than one hand to test carry-over"
    deep = [c for c in by_hand.values() if max(c) >= 3]
    assert deep, "no hand accumulated enough steps for carry-over to show"

    for counts in by_hand.values():
        # Heads-up, at most one opponent action can precede hero's first
        # decision. A carried-over read would start at the previous hand's total.
        assert counts[0] <= 1, counts
        # And within a hand steps only ever accumulate.
        assert counts == sorted(counts), counts


# ------------------------------------------------- the old behaviour stands


def test_conditioning_can_be_turned_off():
    session, seen = passive_session(condition_on_action=False)
    session.play_hand(seed=11)

    for turn in seen:
        assumption = turn.analysis.range_assumption
        assert assumption.notation == DEFAULT_RANGES["station"]
        assert assumption.conditioning is RangeConditioning.PRE_ACTION
        assert assumption.narrowing is None
        assert not assumption.sampled


def test_conditioning_changes_the_equity_it_reports():
    """The point of the exercise. If it changed nothing it would be theatre."""

    on, seen_on = passive_session()
    off, seen_off = passive_session(condition_on_action=False)
    on.play_hand(seed=11)
    off.play_hand(seed=11)

    paired = [
        (a, b) for a, b in zip(seen_on, seen_off)
        if a.analysis.range_assumption.narrowing
        and a.analysis.board == b.analysis.board
    ]
    assert paired, "no comparable decision"
    assert any(
        abs(a.analysis.equity.equity_pct - b.analysis.equity.equity_pct) > 1.0
        for a, b in paired
    )


# --------------------------------------------------------- failure modes


def facing_bet_flop() -> HandState:
    state = HandState(
        players=[
            PlayerState(name="villain", position=Position.BB, stack=94.0,
                        hole_cards=tuple(parse_cards("AhQd"))),
            PlayerState(name="you", position=Position.SB, stack=94.0,
                        hole_cards=tuple(parse_cards("7s7h")), is_hero=True),
        ],
        board=parse_cards("Ac8d3s"), street=Street.FLOP, pot=12.0,
    )
    return state.apply(
        Action(actor="you", type=ActionType.BET, amount=6.0, street=Street.FLOP)
    )


def test_an_emptied_range_falls_back_and_says_why():
    """A station never folds aces, so conditioning a pocket-aces prior on a
    fold leaves nothing. An empty range has no equity, so the tracker has to
    fall back — the requirement is that it does not pretend it didn't.
    """

    config = PracticeConfig(villain_range="AA", conditioning_iterations=120)
    opponent = ObservedOpponent(
        make_player("villain", "station", rng=random.Random(1), iterations=60)
    )
    state = facing_bet_flop()
    opponent.seen.append(
        (state, Action(actor="villain", type=ActionType.FOLD,
                       street=Street.FLOP))
    )

    tracker = _RangeTracker(config, opponent, random.Random(2))
    assumption = tracker.assumption()

    assert tracker.stopped is not None
    assert assumption.notation == "AA"
    assert assumption.conditioning is RangeConditioning.PRE_ACTION
    assert "no read" in assumption.provenance
    assert "no holding left" in assumption.provenance
    # And it must not read like the opponent simply hasn't acted.
    assert "has not acted yet" not in assumption.provenance


def test_conditioning_stops_for_the_hand_once_it_breaks():
    """Later actions condition on a range already known to exclude the truth."""

    config = PracticeConfig(villain_range="AA", conditioning_iterations=120)
    opponent = ObservedOpponent(
        make_player("villain", "station", rng=random.Random(1), iterations=60)
    )
    state = facing_bet_flop()
    opponent.seen += [
        (state, Action(actor="villain", type=ActionType.FOLD,
                       street=Street.FLOP)),
        (state, Action(actor="villain", type=ActionType.CALL, amount=6.0,
                       street=Street.FLOP)),
    ]

    tracker = _RangeTracker(config, opponent, random.Random(2))
    tracker.assumption()
    assert tracker.consumed == 2
    assert tracker.posterior is None


def test_an_exact_equity_against_a_sampled_range_is_not_called_deterministic():
    """The confidence boundary moves up a level once the range is inferred.

    A river enumeration is exact arithmetic — but on an input that was narrowed
    by simulation. Announcing that as "deterministically computed" repeats, one
    layer up, the error the fact categories exist to prevent.
    """

    sampled_range = Feedback(
        action_taken="call", unresolved=[],
        verified=["Showdown equity vs the assumed range: 61.00%."],
        equity_exact=True, range_sampled=True,
    ).render()
    assert "narrowed by simulation" in sampled_range
    assert "Deterministically computed" not in sampled_range

    fixed_range = Feedback(
        action_taken="call", unresolved=[],
        verified=["Showdown equity vs the assumed range: 61.00%."],
        equity_exact=True, range_sampled=False,
    ).render()
    assert "Deterministically computed under the stated assumptions" in fixed_range

    # A sampled equity still leads with its own margin of error.
    sampled_equity = Feedback(
        action_taken="call", unresolved=[], verified=["x"],
        equity_exact=False, range_sampled=True,
    ).render()
    assert "margin of error" in sampled_equity


def test_a_conditioned_range_is_marked_sampled():
    session, seen = passive_session()
    session.play_hand(seed=11)

    conditioned = [t for t in seen if t.analysis.range_assumption.narrowing]
    assert conditioned
    assert all(t.analysis.range_assumption.sampled for t in conditioned)


# ------------------------------------------------------- what reaches the eye


def test_the_student_is_told_which_range_the_equity_is_against():
    session, seen = passive_session()
    session.play_hand(seed=11)

    conditioned = [
        r for r in session.records if r.range_narrowing is not None
    ]
    assert conditioned
    for record in conditioned:
        verified = " ".join(record.verified)
        assert "Villain's range:" in verified
        assert record.range_narrowing in verified
        assert "not of a person" in verified


def test_the_facts_block_states_the_narrowing_and_its_source():
    session, seen = passive_session()
    session.play_hand(seed=11)

    turn = next(t for t in seen if t.analysis.range_assumption.narrowing)
    block = turn.analysis.to_prompt_block()
    assumption = turn.analysis.range_assumption

    assert assumption.narrowing in block
    assert "action-conditioned" in block
    assert "not of a person" in block
    # The combo dump stays out of the prompt; the label stands in for it.
    assert assumption.notation not in block


def test_the_prompt_block_still_grounds_itself_when_conditioned():
    """The rule that made this project: the facts must ground their own numbers.

    Conditioning adds two counts and a range description to the block, and a
    range description is exactly the kind of text that has silently corrupted
    number extraction here before.
    """

    from poker_coach.evaluation.grounding import check_grounding

    session, seen = passive_session()
    for seed in (11, 23, 37):
        session.play_hand(seed=seed)

    conditioned = [t for t in seen if t.analysis.range_assumption.narrowing]
    assert conditioned
    for turn in conditioned:
        report = check_grounding(turn.analysis.to_prompt_block(), turn.analysis)
        assert report.is_grounded, report.summary()


def test_invented_combo_counts_are_still_caught():
    from poker_coach.evaluation.grounding import check_grounding

    session, seen = passive_session()
    session.play_hand(seed=11)
    analysis = next(
        t.analysis for t in seen if t.analysis.range_assumption.narrowing
    )

    quoted = analysis.range_assumption.narrowing
    assert check_grounding(f"Only {quoted} combos get here.", analysis).is_grounded
    invented = check_grounding("Only 613 of 927 combos get here.", analysis)
    assert not invented.is_grounded


def test_the_cli_can_ask_for_a_fixed_range(tmp_path):
    import io

    from poker_coach.practice.cli import run_practice

    out = io.StringIO()
    records = run_practice(
        hands=2, opponent="station", model=None, seed=11,
        stdin=io.StringIO("c\nc\nk\nk\nk\nk\nk\nk\n"), stdout=out,
        log_dir=str(tmp_path), iterations=200, condition_on_action=False,
    )
    text = out.getvalue()
    assert "not inferred from play" in text
    assert "villain's range:" not in text
    assert records
    assert all(r.range_narrowing is None for r in records)


def test_the_cli_narrows_the_range_by_default(tmp_path):
    import io

    from poker_coach.practice.cli import run_practice

    out = io.StringIO()
    records = run_practice(
        hands=2, opponent="station", model=None, seed=11,
        stdin=io.StringIO("c\nc\nk\nk\nk\nk\nk\nk\n"), stdout=out,
        log_dir=str(tmp_path), iterations=200,
    )
    text = out.getvalue()
    assert "narrowed by each action" in text
    assert "villain's range:" in text
    assert any(r.range_narrowing is not None for r in records)
