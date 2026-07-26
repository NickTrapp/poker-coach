"""A durable record of every practice decision.

The point of the practice loop is not the hands — it is the transcript. Twenty
logged decisions are what will show whether a repeated weakness sits in the
student's play, in the coach's reading, or in the facts it was handed, and they
are the evidence `knowledge/` should be designed from rather than guessed at.

So a record carries everything needed to reconstruct the decision without the
session that produced it: the seed and full state, what was legal, what was
chosen, the exact facts block, every raw model response including a repair
attempt, the grounding verdict, the resolved model id, and latency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from ..coaching.analysis import SpotAnalysis
from ..domain.action import Action
from ..domain.betting import LegalAction
from ..domain.state import HandState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .critique import CritiqueResult
    from .session import PracticeConfig

__all__ = ["DecisionRecord", "write_records", "load_records", "session_path"]


@dataclass
class DecisionRecord:
    """One hero decision, in full."""

    recorded_at: str
    hand_number: int
    decision_number: int
    seed: int

    # --- configuration, so the assumption is never lost -------------------
    opponent_style: str
    villain_range: str
    range_conditioning: str
    range_source: str

    # --- the spot ----------------------------------------------------------
    street: str
    hero_cards: str
    board: str
    pot: float
    to_call: float
    hero_stack_behind: float
    legal_actions: list[str]
    action_taken: str

    # --- what the coach was given and said ---------------------------------
    facts_block: str
    equity_pct: float
    equity_exact: bool
    ev_call_vs_fold: float
    ev_is_terminal: bool
    verified: list[str]
    unresolved: list[str]
    interpretation: str | None
    raw_responses: list[str]

    # --- how it was produced ------------------------------------------------
    grounding_ok: bool | None
    grounding_summary: str | None
    ungrounded_claims: list[str]
    repair_attempted: bool
    fell_back: bool
    warnings: list[str]
    model_name: str | None
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    error: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        hand_number: int,
        decision_number: int,
        seed: int,
        config: "PracticeConfig",
        state: HandState,
        legal: list[LegalAction],
        action: Action,
        analysis: SpotAnalysis,
        critique: "CritiqueResult",
    ) -> "DecisionRecord":
        assumption = config.range_for()
        report = critique.grounding
        feedback = critique.feedback

        return cls(
            recorded_at=datetime.now(timezone.utc).isoformat(),
            hand_number=hand_number,
            decision_number=decision_number,
            seed=seed,
            opponent_style=config.opponent_style,
            villain_range=assumption.notation,
            range_conditioning=assumption.conditioning.value,
            range_source="fixed practice configuration, not inferred",
            street=analysis.street,
            hero_cards=analysis.hero_cards,
            board=analysis.board,
            pot=analysis.total_pot,
            to_call=analysis.to_call,
            hero_stack_behind=analysis.hero_stack_behind,
            legal_actions=[str(option) for option in legal],
            action_taken=feedback.action_taken,
            facts_block=analysis.to_prompt_block(),
            equity_pct=analysis.equity.equity_pct,
            equity_exact=analysis.equity.exact,
            ev_call_vs_fold=analysis.ev_call_vs_fold,
            ev_is_terminal=analysis.ev_is_terminal,
            verified=list(feedback.verified),
            unresolved=list(feedback.unresolved),
            interpretation=feedback.interpretation,
            raw_responses=list(critique.raw_responses),
            grounding_ok=None if report is None else report.is_grounded,
            grounding_summary=None if report is None else report.summary(),
            ungrounded_claims=(
                [] if report is None else [c.text for c in report.ungrounded]
            ),
            repair_attempted=critique.repair_attempted,
            fell_back=critique.fell_back,
            warnings=list(feedback.warnings),
            model_name=critique.model_name,
            prompt_version=critique.prompt_version,
            input_tokens=critique.input_tokens,
            output_tokens=critique.output_tokens,
            latency_seconds=round(critique.latency_seconds, 3),
            error=critique.error,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def session_path(directory: str | Path = "practice") -> Path:
    """A timestamped JSONL path, one file per session."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(directory) / f"session-{stamp}.jsonl"


def write_records(path: str | Path, records: list[DecisionRecord]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
    return target


def load_records(path: str | Path) -> Iterator[DecisionRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield DecisionRecord(**json.loads(line))
