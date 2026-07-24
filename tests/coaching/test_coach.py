import random

import pytest

from poker_coach.coaching.coach import Coach
from poker_coach.coaching.prompts.system import (
    COACH_SYSTEM_PROMPT,
    render_system_prompt,
)
from poker_coach.domain import HandState, PlayerState, Position, Street, parse_cards
from poker_coach.models.base import EchoModel, LanguageModel, ScriptedModel


def flop_state() -> HandState:
    return HandState(
        players=[
            PlayerState(
                name="hero",
                position=Position.BTN,
                stack=97.0,
                hole_cards=tuple(parse_cards("AsKs")),
                is_hero=True,
            ),
            PlayerState(name="villain", position=Position.BB, stack=97.0),
        ],
        board=parse_cards("Qs2s9c"),
        street=Street.FLOP,
        pot=6.0,
    )


def make_coach(model=None) -> Coach:
    return Coach(
        model=model or EchoModel(),
        iterations=500,
        rng=random.Random(3),
    )


def test_stub_models_satisfy_the_protocol():
    assert isinstance(EchoModel(), LanguageModel)
    assert isinstance(ScriptedModel(["hi"]), LanguageModel)


def test_advise_returns_text_grounded_in_the_analysis():
    coach = make_coach()
    advice = coach.advise(flop_state())
    assert advice.analysis.hero_cards == "AsKs"
    assert "Hero equity" in advice.facts
    assert advice.response.model == "echo"


def test_the_facts_block_reaches_the_model():
    model = EchoModel()
    coach = make_coach(model)
    coach.advise(flop_state())
    assert "FACTS" in model.last_user_message
    assert "AsKs" in model.last_user_message
    assert "Qs2s9c" in model.last_user_message


def test_the_system_prompt_reaches_the_model():
    model = EchoModel()
    coach = make_coach(model)
    coach.advise(flop_state())
    assert model.last_system_prompt == COACH_SYSTEM_PROMPT


def test_system_prompt_forbids_recomputing_numbers():
    # Collapse whitespace first: the prompt is hard-wrapped, so a reflow would
    # otherwise break this assertion without changing the meaning.
    flat = " ".join(COACH_SYSTEM_PROMPT.lower().split())
    assert "never recompute" in flat
    assert "quote it as given" in flat


def test_system_prompt_teaches_the_fact_categories():
    flat = " ".join(COACH_SYSTEM_PROMPT.lower().split())
    for heading in (
        "exact state facts",
        "sampled calculations",
        "assumption-conditioned calculations",
        "theoretical reference values",
        "questions these facts do not answer",
    ):
        assert heading in flat, heading


def test_system_prompt_separates_call_from_raise():
    """The semantic fix has to survive in the prompt, not just the facts."""

    flat = " ".join(COACH_SYSTEM_PROMPT.lower().split())
    assert "does not mean calling is better than raising" in flat


def test_system_prompt_warns_that_equity_is_an_upper_bound():
    flat = " ".join(COACH_SYSTEM_PROMPT.lower().split())
    assert "upper bound" in flat


def test_system_prompt_denies_that_mdf_is_an_obligation():
    flat = " ".join(COACH_SYSTEM_PROMPT.lower().split())
    assert "describes a bet size, not an obligation" in flat


def test_style_is_appended_to_the_system_prompt():
    coach = Coach(model=EchoModel(), style="blunt, no hedging", iterations=200)
    assert coach.system_prompt.startswith(COACH_SYSTEM_PROMPT)
    assert "blunt, no hedging" in coach.system_prompt


def test_render_system_prompt_without_style_is_the_base():
    assert render_system_prompt(COACH_SYSTEM_PROMPT) == COACH_SYSTEM_PROMPT


def test_default_question_is_used_when_none_is_given():
    model = EchoModel()
    make_coach(model).advise(flop_state())
    assert "What should the hero do here" in model.last_user_message


def test_custom_question_is_passed_through():
    model = EchoModel()
    make_coach(model).advise(flop_state(), question="Should I check-raise?")
    assert "Should I check-raise?" in model.last_user_message


def test_history_accumulates_across_turns():
    coach = make_coach(ScriptedModel(["first", "second"]))
    coach.advise(flop_state())
    response = coach.ask("What if villain is a nit?")
    assert response.text == "second"
    assert [m.role for m in coach.history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_ask_before_advise_is_an_error():
    with pytest.raises(ValueError, match="no spot has been analysed"):
        make_coach().ask("what now?")


def test_reset_clears_history_but_keeps_configuration():
    coach = make_coach(ScriptedModel(["a", "b"]))
    coach.advise(flop_state())
    coach.reset()
    assert coach.history == []
    coach.advise(flop_state())  # uses the second scripted reply
    assert len(coach.history) == 2


def test_scripted_model_runs_out_loudly():
    coach = make_coach(ScriptedModel(["only one"]))
    coach.advise(flop_state())
    with pytest.raises(RuntimeError, match="ran out of replies"):
        coach.ask("again?")


def test_analyse_alone_does_not_call_the_model():
    model = EchoModel()
    coach = make_coach(model)
    analysis = coach.analyse(flop_state())
    assert analysis.hero_cards == "AsKs"
    assert model.calls == []


def test_villain_range_override_flows_through_to_the_analysis():
    coach = make_coach()
    advice = coach.advise(flop_state(), villain_range="QQ+")
    assert advice.analysis.villain_range == "QQ+"
    assert "QQ+" in advice.facts


def test_default_villain_range_is_configurable():
    coach = Coach(model=EchoModel(), default_villain_range="22+", iterations=300)
    assert coach.advise(flop_state()).analysis.villain_range == "22+"


def test_tracing_records_the_call_path():
    coach = make_coach()
    coach.advise(flop_state())
    assert [s.name for s in coach.trace.spans] == ["analyse", "model.complete"]
    assert coach.trace.find("model.complete")[0].tags["model"] == "echo"
    assert coach.trace.total_ms >= 0.0
