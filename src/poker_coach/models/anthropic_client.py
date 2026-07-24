"""Anthropic-backed implementation of :class:`LanguageModel`.

The `anthropic` package is an optional dependency — install with
``pip install -e ".[anthropic]"``. It is imported lazily so that the rest of
the package (and the whole test suite) keeps working without it.

Design notes tied to the current API surface:

* **No sampling parameters.** ``temperature``/``top_p``/``top_k`` are rejected
  with a 400 on current models. Depth is controlled by ``effort`` instead.
* **Thinking is off unless asked for.** Omitting the ``thinking`` field runs
  without thinking, so `enable_thinking` sets ``{"type": "adaptive"}``
  explicitly. Reasoning text is omitted by default; `show_thinking` opts into
  the summarized form.
* **Large outputs must stream.** Non-streaming requests with a big
  ``max_tokens`` risk an HTTP timeout, so this client switches to the
  streaming path automatically above :data:`STREAMING_THRESHOLD`.
* **Refusals are not exceptions.** A declined request returns HTTP 200 with
  ``stop_reason == "refusal"`` and empty or partial content, so
  :meth:`AnthropicModel.complete` checks ``stop_reason`` before reading text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .base import Message, ModelResponse, Usage

__all__ = ["AnthropicModel", "DEFAULT_MODEL", "STREAMING_THRESHOLD"]

#: The current, most capable Opus-tier model.
DEFAULT_MODEL = "claude-opus-4-8"

#: Above this many output tokens, use the streaming path to dodge HTTP timeouts.
STREAMING_THRESHOLD = 16_000

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass
class AnthropicModel:
    """Calls Claude through the official SDK.

    ``effort`` is the main quality/cost dial. For a coaching turn the default
    of ``"high"`` is a good balance; drop to ``"medium"`` for cost-sensitive
    deployments and reach for ``"xhigh"`` only if evaluation shows it helps.

    This module is named ``anthropic_client`` rather than ``anthropic`` so it
    never reads as shadowing the SDK package it imports.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    effort: Effort | None = "high"
    enable_thinking: bool = True
    show_thinking: bool = False
    timeout: float | None = None
    max_retries: int | None = None
    client: Any = None

    #: Populated on each call so callers can inspect the last raw response.
    last_raw: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = _build_client(self.timeout, self.max_retries)

    @property
    def name(self) -> str:
        return self.model

    # ------------------------------------------------------------------ call

    def complete(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        limit = max_tokens or self.max_tokens
        kwargs = self.build_kwargs(system, messages, max_tokens=limit)

        if limit > STREAMING_THRESHOLD:
            with self.client.messages.stream(**kwargs) as stream:
                raw = stream.get_final_message()
        else:
            raw = self.client.messages.create(**kwargs)

        self.last_raw = raw
        return _to_response(raw, fallback_model=self.model)

    def build_kwargs(
        self,
        system: str,
        messages: Sequence[Message],
        *,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Assemble the request body. Split out so it can be tested directly."""

        if not messages:
            raise ValueError("at least one message is required")
        if messages[0].role != "user":
            raise ValueError("the first message must come from the user")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

        if self.enable_thinking:
            thinking: dict[str, Any] = {"type": "adaptive"}
            if self.show_thinking:
                thinking["display"] = "summarized"
            kwargs["thinking"] = thinking

        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        return kwargs


def _build_client(timeout: float | None, max_retries: int | None) -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the 'anthropic' package is required for AnthropicModel; "
            'install it with: pip install -e ".[anthropic]"'
        ) from exc

    options: dict[str, Any] = {}
    if timeout is not None:
        options["timeout"] = timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    # Credentials are resolved from the environment by the SDK; never take an
    # API key as a constructor argument, so one cannot end up in a repr or log.
    return anthropic.Anthropic(**options)


def extract_text(raw: Any) -> str:
    """Join every text block in a response, skipping thinking and tool blocks."""

    return "".join(
        block.text for block in getattr(raw, "content", []) if block.type == "text"
    )


def _to_response(raw: Any, *, fallback_model: str) -> ModelResponse:
    stop_reason = getattr(raw, "stop_reason", None)
    usage = getattr(raw, "usage", None)

    if stop_reason == "refusal":
        # A refusal is a successful HTTP 200 with empty or partial content.
        # Surface it as a response rather than raising, so the coaching layer
        # can say plainly that the model declined.
        details = getattr(raw, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        suffix = f" (category: {category})" if category else ""
        return ModelResponse(
            text=f"The model declined to answer this request{suffix}.",
            model=getattr(raw, "model", fallback_model),
            usage=_to_usage(usage),
            stop_reason="refusal",
        )

    return ModelResponse(
        text=extract_text(raw),
        model=getattr(raw, "model", fallback_model),
        usage=_to_usage(usage),
        stop_reason=stop_reason,
    )


def _to_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )
