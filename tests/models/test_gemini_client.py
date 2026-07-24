"""Tests for the Gemini client.

Runs against a fake that mimics the SDK's response shape — no network, no API
key. The request-building tests use the *real* `google.genai.types`, so a
change in the SDK's construction surface would surface here.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from poker_coach.models.base import LanguageModel, Message
from poker_coach.models.gemini_client import (
    DEFAULT_MODEL,
    GeminiModel,
    extract_text,
    list_models,
)

genai_types = pytest.importorskip(
    "google.genai.types", reason="google-genai is an optional dependency"
)


# --------------------------------------------------------------- fake SDK


@dataclass
class FakePart:
    text: str | None = None
    thought: bool = False


@dataclass
class FakeContent:
    parts: list[FakePart]


@dataclass
class FakeReason:
    name: str


@dataclass
class FakeCandidate:
    content: FakeContent
    finish_reason: FakeReason | None = field(default_factory=lambda: FakeReason("STOP"))


@dataclass
class FakeUsage:
    prompt_token_count: int = 0
    candidates_token_count: int = 0


@dataclass
class FakeFeedback:
    block_reason: str | None = None


@dataclass
class FakeRaw:
    candidates: list[FakeCandidate]
    usage_metadata: FakeUsage = field(default_factory=FakeUsage)
    model_version: str | None = None
    prompt_feedback: FakeFeedback | None = None


@dataclass
class FakeModels:
    raw: FakeRaw
    calls: list[dict] = field(default_factory=list)
    listed: list[Any] = field(default_factory=list)

    def generate_content(self, **kwargs: Any) -> FakeRaw:
        self.calls.append(kwargs)
        return self.raw

    def list(self) -> list[Any]:
        return self.listed


@dataclass
class FakeClient:
    models: FakeModels


def raw_text(text: str = "Fold.", **kwargs: Any) -> FakeRaw:
    return FakeRaw(
        candidates=[FakeCandidate(content=FakeContent(parts=[FakePart(text=text)]))],
        **kwargs,
    )


def make_model(raw: FakeRaw | None = None, **kwargs: Any) -> GeminiModel:
    return GeminiModel(client=FakeClient(models=FakeModels(raw=raw or raw_text())),
                       **kwargs)


def user(text: str = "hi") -> list[Message]:
    return [Message.user(text)]


# ------------------------------------------------------------------ basics


def test_satisfies_the_language_model_protocol():
    assert isinstance(make_model(), LanguageModel)


def test_name_reports_the_model_id():
    assert make_model().name == DEFAULT_MODEL


def test_complete_returns_the_text():
    response = make_model().complete("sys", user())
    assert response.text == "Fold."
    assert response.stop_reason == "end_turn"


def test_usage_is_carried_through():
    raw = raw_text(usage_metadata=FakeUsage(prompt_token_count=90,
                                            candidates_token_count=12))
    response = make_model(raw).complete("sys", user())
    assert response.usage.input_tokens == 90
    assert response.usage.output_tokens == 12
    assert response.usage.total_tokens == 102


def test_model_version_is_preferred_when_reported():
    raw = raw_text(model_version="gemini-2.5-pro-002")
    assert make_model(raw).complete("sys", user()).model == "gemini-2.5-pro-002"


def test_model_falls_back_to_the_configured_id():
    assert make_model().complete("sys", user()).model == DEFAULT_MODEL


def test_last_raw_is_exposed():
    model = make_model()
    model.complete("sys", user())
    assert model.last_raw is not None


# ------------------------------------------------------ text extraction


def test_thought_parts_are_not_treated_as_answer_text():
    raw = FakeRaw(candidates=[FakeCandidate(content=FakeContent(parts=[
        FakePart(text="villain is wide", thought=True),
        FakePart(text="Call."),
    ]))])
    assert make_model(raw).complete("sys", user()).text == "Call."


def test_multiple_parts_are_joined():
    raw = FakeRaw(candidates=[FakeCandidate(content=FakeContent(parts=[
        FakePart(text="Call. "), FakePart(text="You have the odds."),
    ]))])
    assert make_model(raw).complete("sys", user()).text == "Call. You have the odds."


def test_parts_without_text_are_skipped():
    raw = FakeRaw(candidates=[FakeCandidate(content=FakeContent(parts=[
        FakePart(text=None), FakePart(text="kept"),
    ]))])
    assert extract_text(raw) == "kept"


def test_no_candidates_yields_empty_text():
    assert extract_text(FakeRaw(candidates=[])) == ""


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "reason",
    ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"],
)
def test_safety_finishes_normalise_to_refusal(reason):
    """One vocabulary across providers, so the coach sees a single signal."""

    raw = FakeRaw(candidates=[FakeCandidate(
        content=FakeContent(parts=[]), finish_reason=FakeReason(reason)
    )])
    response = make_model(raw).complete("sys", user())
    assert response.stop_reason == "refusal"
    assert "declined" in response.text


def test_a_blocked_prompt_with_no_candidates_is_a_refusal():
    raw = FakeRaw(candidates=[], prompt_feedback=FakeFeedback(block_reason="SAFETY"))
    assert make_model(raw).complete("sys", user()).stop_reason == "refusal"


def test_no_candidates_without_a_block_is_not_a_refusal():
    raw = FakeRaw(candidates=[], prompt_feedback=FakeFeedback(block_reason=None))
    response = make_model(raw).complete("sys", user())
    assert response.stop_reason is None
    assert response.text == ""


def test_max_tokens_finish_is_reported():
    raw = raw_text()
    raw.candidates[0].finish_reason = FakeReason("MAX_TOKENS")
    assert make_model(raw).complete("sys", user()).stop_reason == "max_tokens"


def test_unknown_finish_reasons_pass_through_lowercased():
    raw = raw_text()
    raw.candidates[0].finish_reason = FakeReason("OTHER")
    assert make_model(raw).complete("sys", user()).stop_reason == "other"


def test_missing_finish_reason_is_none():
    raw = raw_text()
    raw.candidates[0].finish_reason = None
    assert make_model(raw).complete("sys", user()).stop_reason is None


# ------------------------------------------------------------ request body


def test_the_system_prompt_becomes_a_system_instruction():
    model = make_model()
    model.complete("COACH RULES", user())
    config = model.client.models.calls[0]["config"]
    assert config.system_instruction == "COACH RULES"


def test_max_tokens_maps_to_max_output_tokens():
    model = make_model(max_tokens=2048)
    model.complete("sys", user())
    assert model.client.models.calls[0]["config"].max_output_tokens == 2048


def test_max_tokens_can_be_overridden_per_call():
    model = make_model(max_tokens=2048)
    model.complete("sys", user(), max_tokens=512)
    assert model.client.models.calls[0]["config"].max_output_tokens == 512


def test_the_model_id_is_sent():
    model = make_model(model="gemini-2.5-flash")
    model.complete("sys", user())
    assert model.client.models.calls[0]["model"] == "gemini-2.5-flash"


def test_the_assistant_role_is_renamed_to_model():
    """Gemini rejects the role name "assistant"; it calls that role "model"."""

    model = make_model()
    model.complete("sys", [Message.user("a"), Message.assistant("b"),
                           Message.user("c")])
    roles = [c.role for c in model.client.models.calls[0]["contents"]]
    assert roles == ["user", "model", "user"]


def test_message_text_survives_the_conversion():
    model = make_model()
    model.complete("sys", [Message.user("what now?")])
    contents = model.client.models.calls[0]["contents"]
    assert contents[0].parts[0].text == "what now?"


def test_temperature_is_omitted_unless_configured():
    model = make_model()
    model.complete("sys", user())
    assert model.client.models.calls[0]["config"].temperature is None


def test_temperature_is_sent_when_configured():
    # Gemini accepts sampling parameters, unlike current Claude models — so it
    # is an instance setting here rather than a protocol argument.
    model = make_model(temperature=0.2)
    model.complete("sys", user())
    assert model.client.models.calls[0]["config"].temperature == pytest.approx(0.2)


def test_thinking_budget_is_omitted_unless_configured():
    model = make_model()
    model.complete("sys", user())
    assert model.client.models.calls[0]["config"].thinking_config is None


def test_thinking_budget_is_sent_when_configured():
    model = make_model(thinking_budget=1024)
    model.complete("sys", user())
    config = model.client.models.calls[0]["config"]
    assert config.thinking_config.thinking_budget == 1024


def test_the_config_is_a_real_sdk_object():
    # Built with google.genai.types, so an SDK change breaks this test.
    model = make_model()
    model.complete("sys", user())
    assert isinstance(
        model.client.models.calls[0]["config"], genai_types.GenerateContentConfig
    )


def test_contents_are_real_sdk_objects():
    model = make_model()
    model.complete("sys", user())
    contents = model.client.models.calls[0]["contents"]
    assert all(isinstance(c, genai_types.Content) for c in contents)


# -------------------------------------------------------------- validation


def test_empty_message_list_is_rejected():
    with pytest.raises(ValueError, match="at least one message"):
        make_model().complete("sys", [])


def test_first_message_must_be_from_the_user():
    with pytest.raises(ValueError, match="first message must come from the user"):
        make_model().complete("sys", [Message.assistant("hi")])


# ------------------------------------------------------------------ extras


def test_list_models_reports_ids():
    @dataclass
    class M:
        name: str

    client = FakeClient(models=FakeModels(raw=raw_text()))
    client.models.listed = [M("models/gemini-2.5-pro"), M("models/gemini-2.5-flash")]
    assert list_models(client) == ["models/gemini-2.5-pro", "models/gemini-2.5-flash"]


def test_no_api_key_is_accepted_as_a_constructor_argument():
    # Credentials come from the environment so a key cannot land in a repr.
    import inspect

    assert "api_key" not in inspect.signature(GeminiModel).parameters


def test_repr_cannot_leak_a_key():
    assert "api_key" not in repr(make_model())


def test_gemini_model_is_lazily_exported():
    import poker_coach.models as models

    assert models.GeminiModel is GeminiModel


def test_both_providers_are_exported():
    import poker_coach.models as models

    assert models.AnthropicModel.__name__ == "AnthropicModel"
    assert models.GeminiModel.__name__ == "GeminiModel"


def test_unknown_attribute_still_raises():
    import poker_coach.models as models

    with pytest.raises(AttributeError, match="no attribute"):
        models.NotAThing
