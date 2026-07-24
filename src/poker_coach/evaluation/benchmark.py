"""Run the standard cases against a real model and record what happened.

Every run is written as a JSONL record carrying enough to reproduce and audit
it: the exact facts the model saw, its raw reply, the deterministic verdicts,
token usage, latency, and the resolved model version.

Two things this deliberately does not do:

* **No LLM judge.** Every automatic check here is deterministic. The findings
  that need judgement (a conclusion stronger than the arithmetic, a vacuous
  principle) are surfaced as *review prompts* against the recorded text, for a
  human to label. A speculative automated strategic judge would manufacture
  confidence nobody has earned.
* **No floating alias as identity.** ``gemini-flash-latest`` is what you
  *asked* for; the record also stores what the provider said it actually ran,
  because the alias will point somewhere else next month.
"""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..coaching.coach import Coach
from ..coaching.prompts.system import PROMPT_VERSION
from ..models.base import LanguageModel
from .cases import STANDARD_CASES, Case, CaseResult, grade

__all__ = ["RunRecord", "run_case", "run_benchmark", "write_jsonl", "load_jsonl"]

DEFAULT_SEED = 17


@dataclass
class RunRecord:
    """One case, one model, one run — everything needed to audit it later."""

    case_id: str
    case_description: str
    provider: str
    requested_model: str
    resolved_model: str | None
    prompt_version: str
    seed: int
    iterations: int

    facts_block: str = ""
    response: str = ""

    recommendation: str = "unclear"
    verdict: str | None = None
    expected_verdict: str = ""
    action_correct: bool = False

    grounded: bool = False
    grounding_summary: str = ""
    ungrounded_claims: list[str] = field(default_factory=list)

    ev_is_terminal: bool | None = None
    equity_exact: bool | None = None
    range_conditioning: str = ""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    stop_reason: str | None = None
    error: str | None = None
    timestamp: str = ""

    @property
    def passed(self) -> bool:
        return self.error is None and self.action_correct and self.grounded

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def run_case(
    case: Case,
    model: LanguageModel,
    *,
    provider: str,
    requested_model: str,
    seed: int = DEFAULT_SEED,
    iterations: int = 20_000,
) -> RunRecord:
    """Run one case and record the outcome. Never raises — errors are recorded."""

    record = RunRecord(
        case_id=case.name,
        case_description=case.description,
        provider=provider,
        requested_model=requested_model,
        resolved_model=None,
        prompt_version=PROMPT_VERSION,
        seed=seed,
        iterations=iterations,
        expected_verdict=case.expected,
        range_conditioning=case.range_conditioning,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    try:
        analysis = case.analyse(iterations=iterations, rng=random.Random(seed))
        record.facts_block = analysis.to_prompt_block()
        record.ev_is_terminal = analysis.ev_is_terminal
        record.equity_exact = analysis.equity.exact

        coach = Coach(model=model, iterations=iterations, rng=random.Random(seed))
        prompt = coach._build_prompt(analysis, None)

        started = time.perf_counter()
        from ..models.base import Message

        response = model.complete(coach.system_prompt, [Message.user(prompt)])
        record.latency_seconds = round(time.perf_counter() - started, 3)

        record.response = response.text
        # What the provider says it actually ran, not the alias we asked for.
        record.resolved_model = response.model
        record.stop_reason = response.stop_reason
        record.input_tokens = response.usage.input_tokens
        record.output_tokens = response.usage.output_tokens

        result: CaseResult = grade(case, response.text, analysis=analysis)
        record.recommendation = result.action
        record.verdict = result.verdict
        record.action_correct = result.action_correct
        record.grounded = result.grounding.is_grounded
        record.grounding_summary = result.grounding.summary()
        record.ungrounded_claims = [c.text for c in result.grounding.ungrounded]

    except Exception as exc:  # recorded, not raised: one bad case must not
        record.error = f"{type(exc).__name__}: {exc}"  # abort the whole batch

    return record


def run_benchmark(
    model: LanguageModel,
    *,
    provider: str,
    requested_model: str,
    cases: Sequence[Case] = STANDARD_CASES,
    seed: int = DEFAULT_SEED,
    iterations: int = 20_000,
    on_result=None,
) -> list[RunRecord]:
    """Run every case, returning one record each."""

    records = []
    for case in cases:
        record = run_case(
            case,
            model,
            provider=provider,
            requested_model=requested_model,
            seed=seed,
            iterations=iterations,
        )
        records.append(record)
        if on_result is not None:
            on_result(record)
    return records


def write_jsonl(records: Iterable[RunRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
    return path


def load_jsonl(path: str | Path) -> list[RunRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        return [RunRecord(**json.loads(line)) for line in handle if line.strip()]


def environment() -> dict[str, str]:
    """Recorded alongside a run so a result set is reproducible."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prompt_version": PROMPT_VERSION,
    }
