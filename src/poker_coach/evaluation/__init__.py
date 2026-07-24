"""Coaching-quality checks.

`grounding` is the one that matters: it verifies the "quote, don't compute"
rule from `coaching/prompts/system.py` — the property the whole layering
depends on. It is deterministic and needs no model to run, so it can gate a
response before the student ever sees it.

`cases` is a small fixed benchmark of spots whose correct action is settled by
arithmetic, used to check that a coach reaches the right recommendation *and*
grounds it in the supplied facts.
"""

from .benchmark import RunRecord, load_jsonl, run_benchmark, run_case, write_jsonl
from .cases import (
    STANDARD_CASES,
    Case,
    CaseResult,
    grade,
    recommended_action,
    verdict_of,
)
from .grounding import (
    GroundingReport,
    NumericClaim,
    check_grounding,
    extract_claims,
    grounded_values,
    mask_poker_notation,
)
from .report import MANUAL_CHECKS, ReviewFlag, flag_record, render_report

__all__ = [
    "Case",
    "CaseResult",
    "GroundingReport",
    "MANUAL_CHECKS",
    "NumericClaim",
    "ReviewFlag",
    "RunRecord",
    "STANDARD_CASES",
    "check_grounding",
    "extract_claims",
    "flag_record",
    "grade",
    "grounded_values",
    "load_jsonl",
    "mask_poker_notation",
    "recommended_action",
    "render_report",
    "run_benchmark",
    "run_case",
    "verdict_of",
    "write_jsonl",
]
