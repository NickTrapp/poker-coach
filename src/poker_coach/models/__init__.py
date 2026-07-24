"""Language-model clients and the protocol they satisfy.

`EchoModel` and `ScriptedModel` are deterministic stubs for tests.
`AnthropicModel` and `GeminiModel` are the real clients. Both live behind lazy
imports because their SDKs are optional dependencies, so importing this package
never requires either one — and having two providers behind one protocol is the
proof that nothing above `models/` is coupled to a vendor.
"""

from typing import TYPE_CHECKING, Any

from .base import (
    EchoModel,
    LanguageModel,
    Message,
    ModelResponse,
    ScriptedModel,
    Usage,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .anthropic_client import AnthropicModel
    from .gemini_client import GeminiModel

__all__ = [
    "AnthropicModel",
    "EchoModel",
    "GeminiModel",
    "LanguageModel",
    "Message",
    "ModelResponse",
    "ScriptedModel",
    "Usage",
]

_LAZY = {
    "AnthropicModel": ".anthropic_client",
    "GeminiModel": ".gemini_client",
}


def __getattr__(name: str) -> Any:
    """Resolve the provider clients on first use, not at import time."""

    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module, __name__), name)
