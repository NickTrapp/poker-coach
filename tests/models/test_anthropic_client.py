"""Tests for the Anthropic client.

Everything here runs against a hand-rolled fake that mimics the SDK's response
shape. No network, no API key, no `anthropic` package required.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from poker_coach.models.anthropic_client import (
    DEFAULT_MODEL,
    STREAMING_THRESHOLD,
    AnthropicModel,
    extract_text,
)
from poker_coach.models.base import LanguageModel, Message


# --------------------------------------------------------------- fake SDK


@dataclass
class FakeBlock:
    type: str
    text: str = ""
    thinking: str = ""


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeStopDetails:
    type: str = "refusal"
    category: str | None = None


@dataclass
class FakeRaw:
    content: list[FakeBlock]
    model: str = DEFAULT_MODEL
    stop_reason: str | None = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_details: FakeStopDetails | None = None


class FakeStream:
    def __init__(self, raw: FakeRaw) -> None:
        self._raw = raw

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> FakeRaw:
        return self._raw


@dataclass
class FakeMessages:
    raw: FakeRaw
    create_calls: list[dict] = field(default_factory=list)
    stream_calls: list[dict] = field(default_factory=list)

    def create(self, **kwargs: Any) -> FakeRaw:
        self.create_calls.append(kwargs)
        return self.raw

    def stream(self, **kwargs: Any) -> FakeStream:
        self.stream_calls.append(kwargs)
        return FakeStream(self.raw)


@dataclass
class FakeClient:
    messages: FakeMessages


def make_model(raw: FakeRaw | None = None, **kwargs: Any) -> AnthropicModel:
    raw = raw or FakeRaw(content=[FakeBlock(type="text", text="Fold.")])
    return AnthropicModel(client=FakeClient(messages=FakeMessages(raw=raw)), **kwargs)


def user(text: str = "hi") -> list[Message]:
    return [Message.user(text)]


# ------------------------------------------------------------------ basics


def test_satisfies_the_language_model_protocol():
    assert isinstance(make_model(), LanguageModel)


def test_default_model_is_the_current_opus():
    assert AnthropicModel(client=object()).model == "claude-opus-4-8"


def test_name_reports_the_model_id():
    assert make_model().name == DEFAULT_MODEL


def test_complete_returns_the_text_blocks():
    response = make_model().complete("sys", user())
    assert response.text == "Fold."
    assert response.stop_reason == "end_turn"


def test_usage_is_carried_through():
    raw = FakeRaw(
        content=[FakeBlock(type="text", text="ok")],
        usage=FakeUsage(input_tokens=120, output_tokens=34),
    )
    response = make_model(raw).complete("sys", user())
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 34
    assert response.usage.total_tokens == 154


def test_model_id_comes_from_the_response():
    raw = FakeRaw(content=[FakeBlock(type="text", text="ok")], model="claude-opus-4-8")
    assert make_model(raw).complete("sys", user()).model == "claude-opus-4-8"


def test_last_raw_is_exposed_for_inspection():
    model = make_model()
    model.complete("sys", user())
    assert model.last_raw is not None


# ------------------------------------------------------ text block handling


def test_thinking_blocks_are_not_treated_as_answer_text():
    raw = FakeRaw(
        content=[
            FakeBlock(type="thinking", thinking="villain's range is wide"),
            FakeBlock(type="text", text="Call."),
        ]
    )
    assert make_model(raw).complete("sys", user()).text == "Call."


def test_multiple_text_blocks_are_joined():
    raw = FakeRaw(
        content=[
            FakeBlock(type="text", text="Call. "),
            FakeBlock(type="text", text="You have the odds."),
        ]
    )
    assert make_model(raw).complete("sys", user()).text == "Call. You have the odds."


def test_extract_text_ignores_unknown_block_types():
    raw = FakeRaw(
        content=[
            FakeBlock(type="tool_use"),
            FakeBlock(type="text", text="kept"),
            FakeBlock(type="redacted_thinking"),
        ]
    )
    assert extract_text(raw) == "kept"


def test_empty_content_yields_empty_text():
    assert make_model(FakeRaw(content=[])).complete("sys", user()).text == ""


# ---------------------------------------------------------------- refusals


def test_refusal_is_returned_not_raised():
    raw = FakeRaw(content=[], stop_reason="refusal")
    response = make_model(raw).complete("sys", user())
    assert response.stop_reason == "refusal"
    assert "declined" in response.text


def test_refusal_category_is_surfaced_when_present():
    raw = FakeRaw(
        content=[],
        stop_reason="refusal",
        stop_details=FakeStopDetails(category="cyber"),
    )
    assert "cyber" in make_model(raw).complete("sys", user()).text


def test_refusal_without_details_still_reads_cleanly():
    raw = FakeRaw(content=[], stop_reason="refusal", stop_details=None)
    text = make_model(raw).complete("sys", user()).text
    assert "declined" in text and "category" not in text


# ------------------------------------------------------------ request body


def test_system_prompt_and_messages_are_sent():
    model = make_model()
    model.complete("COACH RULES", [Message.user("what now?")])
    sent = model.client.messages.create_calls[0]
    assert sent["system"] == "COACH RULES"
    assert sent["messages"] == [{"role": "user", "content": "what now?"}]


def test_multi_turn_history_is_sent_in_order():
    model = make_model()
    model.complete(
        "sys",
        [Message.user("a"), Message.assistant("b"), Message.user("c")],
    )
    roles = [m["role"] for m in model.client.messages.create_calls[0]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_no_sampling_parameters_are_ever_sent():
    # Current models reject temperature/top_p/top_k with a 400.
    model = make_model()
    model.complete("sys", user())
    sent = model.client.messages.create_calls[0]
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert "top_k" not in sent


def test_adaptive_thinking_is_requested_explicitly():
    # Omitting the field runs without thinking, so it must be set explicitly.
    model = make_model()
    model.complete("sys", user())
    assert model.client.messages.create_calls[0]["thinking"] == {"type": "adaptive"}


def test_thinking_can_be_disabled_by_omitting_the_field():
    model = make_model(enable_thinking=False)
    model.complete("sys", user())
    assert "thinking" not in model.client.messages.create_calls[0]


def test_summarized_thinking_is_opt_in():
    model = make_model(show_thinking=True)
    model.complete("sys", user())
    assert model.client.messages.create_calls[0]["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


def test_effort_is_nested_under_output_config():
    model = make_model(effort="xhigh")
    model.complete("sys", user())
    sent = model.client.messages.create_calls[0]
    assert sent["output_config"] == {"effort": "xhigh"}
    assert "effort" not in sent  # not a top-level field


def test_effort_can_be_omitted():
    model = make_model(effort=None)
    model.complete("sys", user())
    assert "output_config" not in model.client.messages.create_calls[0]


def test_max_tokens_defaults_to_the_instance_setting():
    model = make_model(max_tokens=2048)
    model.complete("sys", user())
    assert model.client.messages.create_calls[0]["max_tokens"] == 2048


def test_max_tokens_can_be_overridden_per_call():
    model = make_model(max_tokens=2048)
    model.complete("sys", user(), max_tokens=512)
    assert model.client.messages.create_calls[0]["max_tokens"] == 512


# -------------------------------------------------------------- validation


def test_empty_message_list_is_rejected():
    with pytest.raises(ValueError, match="at least one message"):
        make_model().complete("sys", [])


def test_first_message_must_be_from_the_user():
    with pytest.raises(ValueError, match="first message must come from the user"):
        make_model().complete("sys", [Message.assistant("hi")])


# --------------------------------------------------------------- streaming


def test_small_requests_use_the_plain_create_path():
    model = make_model(max_tokens=4096)
    model.complete("sys", user())
    assert len(model.client.messages.create_calls) == 1
    assert model.client.messages.stream_calls == []


def test_large_requests_switch_to_streaming():
    # Non-streaming requests with a big max_tokens risk an HTTP timeout.
    model = make_model(max_tokens=STREAMING_THRESHOLD + 1)
    response = model.complete("sys", user())
    assert model.client.messages.create_calls == []
    assert len(model.client.messages.stream_calls) == 1
    assert response.text == "Fold."


def test_streaming_threshold_is_not_crossed_at_the_boundary():
    model = make_model(max_tokens=STREAMING_THRESHOLD)
    model.complete("sys", user())
    assert len(model.client.messages.create_calls) == 1


def test_per_call_max_tokens_drives_the_streaming_decision():
    model = make_model(max_tokens=1024)
    model.complete("sys", user(), max_tokens=STREAMING_THRESHOLD + 1)
    assert len(model.client.messages.stream_calls) == 1


def test_streaming_sends_the_same_body_as_create():
    streamed = make_model(max_tokens=STREAMING_THRESHOLD + 1)
    streamed.complete("sys", user())
    body = streamed.client.messages.stream_calls[0]
    assert body["system"] == "sys"
    assert body["thinking"] == {"type": "adaptive"}


# ----------------------------------------------------------- build_kwargs


def test_build_kwargs_is_usable_without_a_call():
    model = make_model()
    body = model.build_kwargs("sys", user(), max_tokens=99)
    assert body["model"] == DEFAULT_MODEL
    assert body["max_tokens"] == 99
    assert model.client.messages.create_calls == []


# ------------------------------------------------------------- optional dep


def test_the_package_imports_without_the_anthropic_sdk():
    # `poker_coach.models` must not require the optional dependency.
    import importlib

    module = importlib.import_module("poker_coach.models")
    assert module.EchoModel is not None


def test_anthropic_model_is_lazily_exported():
    import poker_coach.models as models

    assert models.AnthropicModel is AnthropicModel


def test_unknown_attribute_still_raises():
    import poker_coach.models as models

    with pytest.raises(AttributeError, match="no attribute"):
        models.NotAThing
