"""Gemini-backed implementation of :class:`LanguageModel`.

The `google-genai` package is an optional dependency — install with
``pip install -e ".[gemini]"``. It is imported lazily, so the rest of the
package (and the whole test suite) keeps working without it.

This exists alongside `anthropic_client.py` to keep the model seam honest: two
providers behind one protocol is the proof that nothing above `models/` is
coupled to a vendor.

Translation this adapter performs
---------------------------------
====================  =====================================================
This codebase         Gemini
====================  =====================================================
``system=``           ``config.system_instruction``
role ``"assistant"``  role ``"model"`` — Gemini names the reply role
                      differently, and sending ``"assistant"`` is rejected
``max_tokens``        ``config.max_output_tokens``
``ModelResponse``     joined text parts + ``usage_metadata``
====================  =====================================================

Stop reasons are normalised to the same vocabulary the Anthropic client uses,
so the coaching layer sees one signal regardless of provider — in particular a
safety block becomes ``"refusal"`` rather than a provider-specific enum name.

Unlike current Claude models, Gemini accepts sampling parameters. ``temperature``
is therefore available here as an *instance* setting, not a per-call argument,
which keeps :class:`~poker_coach.models.base.LanguageModel` provider-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import Message, ModelResponse, Usage

__all__ = ["GeminiModel", "DEFAULT_MODEL", "list_models"]

#: Verified reachable against a live key on 2026-07-22. Gemini ids move faster
#: than this file does — the previous default here (``gemini-2.5-pro``) was
#: already stale and returned an error. Call :func:`list_models` to see what
#: your key can actually reach; prefer a ``-pro`` tier for coaching quality if
#: one is listed.
DEFAULT_MODEL = "gemini-flash-latest"

#: Finish reasons that mean "the model declined", mapped to one shared signal.
_REFUSAL_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "RECITATION"}
)

_STOP_REASONS = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}


@dataclass
class GeminiModel:
    """Calls Gemini through the official `google-genai` SDK.

    Credentials are resolved from the environment by the SDK (``GEMINI_API_KEY``
    or ``GOOGLE_API_KEY``). No API key is accepted as a constructor argument, so
    one cannot end up in a repr, a log line, or a traceback.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    temperature: float | None = None
    thinking_budget: int | None = None
    client: Any = None

    #: Populated on each call so callers can inspect the last raw response.
    last_raw: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = _build_client()

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
        raw = self.client.models.generate_content(
            model=self.model,
            contents=self.build_contents(messages),
            config=self.build_config(system, max_tokens=limit),
        )
        self.last_raw = raw
        return _to_response(raw, fallback_model=self.model)

    def build_contents(self, messages: Sequence[Message]) -> list[Any]:
        """Convert the conversation to Gemini `Content` objects.

        Split out so the request shape can be tested without a client.
        """

        from google.genai import types

        if not messages:
            raise ValueError("at least one message is required")
        if messages[0].role != "user":
            raise ValueError("the first message must come from the user")

        return [
            types.Content(
                # Gemini calls the assistant role "model".
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]

    def build_config(self, system: str, *, max_tokens: int) -> Any:
        from google.genai import types

        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.thinking_budget is not None:
            config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        return types.GenerateContentConfig(**config)


def _build_client() -> Any:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the 'google-genai' package is required for GeminiModel; "
            'install it with: pip install -e ".[gemini]"'
        ) from exc

    return genai.Client()


def list_models(client: Any = None) -> list[str]:
    """Model ids the configured credentials can reach.

    Worth calling before trusting :data:`DEFAULT_MODEL` — the id set changes
    independently of this codebase.
    """

    client = client or _build_client()
    return [model.name for model in client.models.list()]


def extract_text(raw: Any) -> str:
    """Join the text parts of the first candidate, skipping thought parts."""

    candidates = getattr(raw, "candidates", None) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []

    return "".join(
        part.text
        for part in parts
        if getattr(part, "text", None) and not getattr(part, "thought", False)
    )


def _finish_reason(raw: Any) -> str | None:
    candidates = getattr(raw, "candidates", None) or []
    if not candidates:
        # No candidate at all means the *prompt* was blocked upstream.
        feedback = getattr(raw, "prompt_feedback", None)
        return "refusal" if getattr(feedback, "block_reason", None) else None

    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None

    name = getattr(reason, "name", str(reason))
    if name in _REFUSAL_REASONS:
        return "refusal"
    return _STOP_REASONS.get(name, name.lower())


def _to_response(raw: Any, *, fallback_model: str) -> ModelResponse:
    stop_reason = _finish_reason(raw)
    model = getattr(raw, "model_version", None) or fallback_model

    if stop_reason == "refusal":
        # Mirrors the Anthropic client: a decline is a response to surface, not
        # an exception to raise, so the coach can say plainly that it declined.
        return ModelResponse(
            text="The model declined to answer this request.",
            model=model,
            usage=_to_usage(getattr(raw, "usage_metadata", None)),
            stop_reason="refusal",
        )

    return ModelResponse(
        text=extract_text(raw),
        model=model,
        usage=_to_usage(getattr(raw, "usage_metadata", None)),
        stop_reason=stop_reason,
    )


def _to_usage(metadata: Any) -> Usage:
    if metadata is None:
        return Usage()
    return Usage(
        input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
        output_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
    )
