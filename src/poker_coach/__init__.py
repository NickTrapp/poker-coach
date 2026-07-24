"""poker-coach — an AI-native no-limit hold'em coaching system.

Layering (lower layers never import higher ones):

``domain``
    Cards, actions, hand state. Pure data and rules.
``calculations``
    Deterministic math — hand evaluation, ranges, equity, pot odds.
``knowledge`` / ``solver``
    Strategic reference material and solver-derived baselines.
``models``
    The language-model seam: a protocol, deterministic stubs, and the
    Anthropic client (optional dependency).
``coaching``
    The LLM-facing layer that turns state + numbers into explanation.
``players`` / ``evaluation`` / ``tracing``
    Simulated opponents, coaching-quality benchmarks, and run introspection.
"""

from .domain import (
    Action,
    ActionType,
    Card,
    Deck,
    HandState,
    PlayerState,
    Position,
    Rank,
    Street,
    Suit,
    parse_cards,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionType",
    "Card",
    "Deck",
    "HandState",
    "PlayerState",
    "Position",
    "Rank",
    "Street",
    "Suit",
    "__version__",
    "parse_cards",
]
