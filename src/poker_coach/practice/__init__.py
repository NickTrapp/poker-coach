"""Play hands against an archetype and get each decision critiqued.

The orchestration layer: state generation, turn pausing, legal actions,
opponent policy, deterministic analysis, model prompting, grounding
verification with one repair attempt, fallback, and per-decision logging.

Two things it deliberately does not do. It does not re-implement a betting
round — a human is just another `Player`, so `players.table.play_hand` runs the
hand unchanged. And it does not infer what the opponent holds: the range is
configuration the caller states, labelled as such everywhere it appears.
"""

from .critique import PRACTICE_SYSTEM_PROMPT, CritiqueResult, critique
from .feedback import Feedback, build_feedback, unresolved_lines, verified_lines
from .log import DecisionRecord, load_records, session_path, write_records
from .session import (
    DEFAULT_RANGES,
    CallbackPlayer,
    HeroTurn,
    PracticeConfig,
    PracticeSession,
)

__all__ = [
    "CallbackPlayer",
    "CritiqueResult",
    "DEFAULT_RANGES",
    "DecisionRecord",
    "Feedback",
    "HeroTurn",
    "PRACTICE_SYSTEM_PROMPT",
    "PracticeConfig",
    "PracticeSession",
    "build_feedback",
    "critique",
    "load_records",
    "session_path",
    "unresolved_lines",
    "verified_lines",
    "write_records",
]
