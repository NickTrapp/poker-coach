import random
from pathlib import Path

import pytest

from poker_coach.coaching.coach import Coach
from poker_coach.coaching.prompts.system import (
    COACH_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
)
from poker_coach.coaching.review import build_review_facts
from poker_coach.domain import (
    Action,
    ActionType,
    HandHistory,
    Position,
    SeatRecord,
    Street,
    StreetRecord,
    parse_cards,
)
from poker_coach.models.base import EchoModel

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "hands"


def A(actor, type_, amount=0.0):
    return Action(actor=actor, type=type_, amount=amount)


def history() -> HandHistory:
    return HandHistory(
        hero="hero",
        seats=[
            SeatRecord(name="hero", position=Position.BTN, starting_stack=100.0,
                       hole_cards=tuple(parse_cards("AsKs"))),
            SeatRecord(name="villain", position=Position.BB, starting_stack=100.0,
                       hole_cards=tuple(parse_cards("QhQd"))),
        ],
        streets=[
            StreetRecord(street=Street.PREFLOP, actions=[
                A("hero", ActionType.RAISE, 3.0),
                A("villain", ActionType.CALL, 3.0),
            ]),
            StreetRecord(street=Street.FLOP, cards=parse_cards("Qs2s9c"), actions=[
                A("villain", ActionType.BET, 4.0),
                A("hero", ActionType.CALL, 4.0),
            ]),
        ],
    )


def facts(**kwargs):
    kwargs.setdefault("iterations", 400)
    kwargs.setdefault("rng", random.Random(5))
    return build_review_facts(history(), **kwargs)


def test_only_hero_decisions_are_reviewed():
    reviewed = facts()
    assert len(reviewed) == 2
    assert [d.action_type for d in reviewed.decisions] == ["raise", "call"]


def test_decisions_are_numbered_and_carry_their_street():
    reviewed = facts()
    assert [d.index for d in reviewed.decisions] == [1, 2]
    assert [d.street for d in reviewed.decisions] == ["preflop", "flop"]


def test_each_decision_carries_the_facts_as_they_stood():
    first, second = facts().decisions
    # Preflop the board is empty; on the flop it is not.
    assert first.analysis.board == ""
    assert second.analysis.board == "Qs2s9c"
    # The flop call was facing a 4 into 6.
    assert second.analysis.to_call == 4.0
    assert second.analysis.total_pot == 10.0


def test_action_taken_is_recorded_with_its_size():
    first, second = facts().decisions
    assert first.action_taken == "raise to 3"
    assert second.action_taken == "call to 4"


def test_folds_and_checks_render_without_an_amount():
    hand = HandHistory(
        hero="hero",
        seats=[
            SeatRecord(name="hero", position=Position.BTN, starting_stack=100.0,
                       hole_cards=tuple(parse_cards("7h2d"))),
            SeatRecord(name="villain", position=Position.BB, starting_stack=100.0),
        ],
        streets=[
            StreetRecord(street=Street.PREFLOP, actions=[A("hero", ActionType.FOLD)]),
        ],
    )
    reviewed = build_review_facts(hand, iterations=200, rng=random.Random(1))
    assert reviewed.decisions[0].action_taken == "fold"


def test_call_gets_an_arithmetic_verdict():
    _, call = facts().decisions
    verdict = call.call_ev_verdict
    assert verdict is not None
    assert ("profitable" in verdict) or ("losing" in verdict)
    assert call.was_a_call is True


def test_non_calls_get_no_arithmetic_verdict():
    raise_decision = facts().decisions[0]
    assert raise_decision.was_a_call is False
    assert raise_decision.call_ev_verdict is None


def test_verdict_agrees_with_the_ev_number():
    call = facts().decisions[1]
    ev = call.analysis.ev_of_calling
    verdict = call.call_ev_verdict
    assert ("profitable" in verdict) == (ev > 0)


def test_prompt_block_pairs_facts_with_the_action_taken():
    block = facts().to_prompt_block()
    assert "DECISION 1 (preflop)" in block
    assert "DECISION 2 (flop)" in block
    assert "Action taken: raise to 3" in block
    assert "Action taken: call to 4" in block
    assert "Hero equity" in block


def test_final_state_is_carried_through():
    reviewed = facts()
    assert reviewed.final_state.total_pot == 14.0
    assert reviewed.hero == "hero"


def test_hands_with_no_hero_decisions_are_handled():
    hand = HandHistory(
        hero="hero",
        seats=[
            SeatRecord(name="hero", position=Position.BTN, starting_stack=100.0,
                       hole_cards=tuple(parse_cards("AsKs"))),
            SeatRecord(name="villain", position=Position.BB, starting_stack=100.0),
        ],
        streets=[StreetRecord(street=Street.PREFLOP, actions=[])],
    )
    reviewed = build_review_facts(hand, iterations=200, rng=random.Random(1))
    assert len(reviewed) == 0
    assert "faced no decisions" in reviewed.to_prompt_block()


def test_hero_without_hole_cards_is_skipped_not_fatal():
    hand = HandHistory(
        hero="hero",
        seats=[
            SeatRecord(name="hero", position=Position.BTN, starting_stack=100.0),
            SeatRecord(name="villain", position=Position.BB, starting_stack=100.0),
        ],
        streets=[
            StreetRecord(street=Street.PREFLOP, actions=[A("hero", ActionType.FOLD)]),
        ],
    )
    reviewed = build_review_facts(hand, iterations=200, rng=random.Random(1))
    assert len(reviewed) == 0


def test_villain_range_assumption_flows_into_every_decision():
    reviewed = facts(villain_range="QQ+")
    assert all(d.analysis.villain_range == "QQ+" for d in reviewed.decisions)


# ------------------------------------------------------------- Coach.review


def make_coach(model=None) -> Coach:
    return Coach(model=model or EchoModel(), iterations=400, rng=random.Random(5))


def test_review_uses_the_review_system_prompt_not_the_coaching_one():
    model = EchoModel()
    make_coach(model).review(history())
    assert model.last_system_prompt == REVIEW_SYSTEM_PROMPT
    assert model.last_system_prompt != COACH_SYSTEM_PROMPT


def test_review_system_prompt_keeps_the_grounding_rule():
    # The prompt is hard-wrapped prose, so collapse whitespace before matching:
    # otherwise a reflow silently breaks the assertion.
    flat = " ".join(REVIEW_SYSTEM_PROMPT.lower().split())
    assert "never recompute" in flat
    assert "authoritative" in flat
    assert "range assumption" in flat


def test_review_sends_every_decision_to_the_model():
    model = EchoModel()
    make_coach(model).review(history())
    prompt = model.last_user_message
    assert "DECISION 1" in prompt and "DECISION 2" in prompt
    assert "Action taken:" in prompt


def test_review_does_not_pollute_conversation_history():
    coach = make_coach()
    coach.review(history())
    assert coach.history == []


def test_review_returns_the_facts_alongside_the_text():
    review = make_coach().review(history())
    assert len(review.decisions) == 2
    assert review.facts.hero == "hero"
    assert review.response.model == "echo"


def test_review_default_question_asks_for_the_biggest_leak():
    model = EchoModel()
    make_coach(model).review(history())
    assert "biggest leak" in model.last_user_message


def test_review_custom_question_is_used():
    model = EchoModel()
    make_coach(model).review(history(), question="Was the flop call bad?")
    assert "Was the flop call bad?" in model.last_user_message


def test_review_style_is_applied_to_the_review_prompt():
    coach = Coach(model=EchoModel(), style="terse", iterations=200)
    coach.review(history())
    assert "terse" in coach.model.last_system_prompt
    assert coach.model.last_system_prompt.startswith(REVIEW_SYSTEM_PROMPT)


def test_review_is_traced():
    coach = make_coach()
    coach.review(history())
    assert [s.name for s in coach.trace.spans] == [
        "build_review_facts",
        "model.complete",
    ]
    assert coach.trace.find("model.complete")[0].tags["kind"] == "review"


@pytest.mark.parametrize(
    "name",
    ["thin-river-call", "dominated-ace-4bet-call", "river-fold-facing-shove"],
)
def test_bundled_examples_review_end_to_end(name):
    hand = HandHistory.from_json_file(str(EXAMPLES / f"{name}.json"))
    coach = Coach(model=EchoModel(), iterations=300, rng=random.Random(2))
    review = coach.review(hand, villain_range="22+, ATs+, KQs, AJo+")
    assert len(review.decisions) >= 1
    assert "DECISION 1" in review.facts.to_prompt_block()
