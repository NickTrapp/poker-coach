"""The language-model seam.

Phase 2 defines the interface only. Nothing in this package opens a socket —
concrete clients land in Phase 3, and until then :class:`EchoModel` and
:class:`ScriptedModel` let the coaching layer be exercised end-to-end in tests
without a network or an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "Message",
    "ModelResponse",
    "Usage",
    "LanguageModel",
    "EchoModel",
    "ScriptedModel",
]


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    text: str
    model: str = "stub"
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str | None = None


@runtime_checkable
class LanguageModel(Protocol):
    """Anything the coach can talk to.

    Implementations must be side-effect free with respect to the caller's
    state: the coach owns the conversation and passes the full history each
    time, so a model instance can be shared across sessions.

    Note the absence of a ``temperature`` knob. Current Claude models reject
    ``temperature``/``top_p``/``top_k`` outright — sampling parameters were
    removed — so a protocol carrying one could not be implemented by a real
    client. Response depth is a property of the model instance (see the
    ``effort`` setting on :class:`~poker_coach.models.anthropic.AnthropicModel`),
    not a per-call argument.
    """

    name: str

    def complete(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
    ) -> ModelResponse: ...


@dataclass
class EchoModel:
    """A deterministic stand-in that echoes the last user message.

    Useful for wiring tests: it proves the prompt reached the model without
    asserting anything about model behaviour.
    """

    name: str = "echo"
    prefix: str = ""
    calls: list[tuple[str, list[Message]]] = field(default_factory=list)

    def complete(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        self.calls.append((system, list(messages)))
        last = messages[-1].content if messages else ""
        return ModelResponse(
            text=f"{self.prefix}{last}",
            model=self.name,
            usage=Usage(input_tokens=len(system) + len(last), output_tokens=len(last)),
            stop_reason="end_turn",
        )

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0] if self.calls else ""

    @property
    def last_user_message(self) -> str:
        if not self.calls or not self.calls[-1][1]:
            return ""
        return self.calls[-1][1][-1].content


@dataclass
class ScriptedModel:
    """Replays a fixed list of replies, in order.

    Lets a test drive a multi-turn coaching session deterministically.
    """

    replies: list[str]
    name: str = "scripted"
    calls: list[tuple[str, list[Message]]] = field(default_factory=list)
    _index: int = 0

    def __init__(self, replies: Iterable[str], name: str = "scripted") -> None:
        self.replies = list(replies)
        self.name = name
        self.calls = []
        self._index = 0

    def complete(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        self.calls.append((system, list(messages)))
        if self._index >= len(self.replies):
            raise RuntimeError(
                f"ScriptedModel ran out of replies after {self._index} calls"
            )
        text = self.replies[self._index]
        self._index += 1
        return ModelResponse(text=text, model=self.name, stop_reason="end_turn")
