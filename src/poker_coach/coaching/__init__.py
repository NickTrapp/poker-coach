"""The LLM-facing coaching layer.

The contract with the layer below: the coach *quotes* numbers produced by
`poker_coach.calculations`, it never computes them itself. `analysis.py`
gathers those numbers into a `SpotAnalysis`, and `coach.py` renders them into a
prompt as authoritative facts.
"""

from .analysis import SpotAnalysis, analyze
from .coach import Advice, Coach, Review
from .prompts.system import (
    COACH_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    render_system_prompt,
)
from .review import HandReviewFacts, ReviewedDecision, build_review_facts

__all__ = [
    "Advice",
    "COACH_SYSTEM_PROMPT",
    "Coach",
    "HandReviewFacts",
    "REVIEW_SYSTEM_PROMPT",
    "Review",
    "ReviewedDecision",
    "SpotAnalysis",
    "analyze",
    "build_review_facts",
    "render_system_prompt",
]
