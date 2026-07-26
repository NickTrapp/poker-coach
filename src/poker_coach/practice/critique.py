"""Ask the coach for an interpretation, and check it before showing it.

The flow is deliberately not "block anything that fails grounding". The checker
matches numeric magnitudes rather than meanings, so it is noisy in both
directions — it will pass "hero has 50% equity" when MDF happens to be 50%, and
flag a legitimately quoted figure it does not recognise. A noisy checker should
not silently delete useful coaching.

So:

1. Ask for an interpretation.
2. Check it.
3. On failure, ask once more, quoting the exact claims that failed.
4. If it fails again, withhold the prose and show the deterministic arithmetic
   alone, with a warning saying why.
5. Log every step either way.

Step 4 is safe to fall back to because the verified and unresolved sections are
generated without a model at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..coaching.analysis import SpotAnalysis
from ..coaching.prompts.system import PROMPT_VERSION
from ..domain.enums import ActionType
from ..evaluation.grounding import GroundingReport, check_grounding
from ..models.base import LanguageModel, Message
from .feedback import Feedback, build_feedback

__all__ = ["CritiqueResult", "PRACTICE_SYSTEM_PROMPT", "critique"]


PRACTICE_SYSTEM_PROMPT = """\
You are a no-limit hold'em coach reviewing one decision a student just made at
the table.

The student has already been shown, separately and in full: the exact
arithmetic (equity, price, EV, implied odds) and a list of questions the
calculations do not answer. **Do not restate either.** Your job is the part
those cannot cover — the read.

Write two or three sentences on:
- what the opponent's tendencies suggest here, given the stated assumed range
- what to watch for on later streets, or what would change the decision
- whether the student's action fits the exploitative picture, and why

Rules:
1. Never state a number that is not in the FACTS block. If you want a figure
   that is not there, describe it in words instead.
2. The assumed opponent range is configuration supplied by the student's
   practice setup, not something the system inferred. Treat it as an
   assumption and say so when it carries weight.
3. Do not claim an action is optimal, GTO, or a mistake. Nothing here computes
   a best action, and a confident verdict would be unearned.
4. Do not compare calling with raising as though a number settled it.
5. No preamble, no headings, no bullet points. Plain sentences.\
"""

_REPAIR_TEMPLATE = """\
Your previous reply cited {count} number(s) that were not in the FACTS block: \
{claims}.

Every figure you state must appear in the FACTS block exactly as given. Rewrite \
your reply, either quoting the supplied numbers correctly or describing the \
point in words with no number at all. Same length, same subject.\
"""


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    """The feedback plus everything needed to reconstruct how it was produced."""

    feedback: Feedback
    raw_responses: list[str]
    grounding: GroundingReport | None
    repair_attempted: bool
    fell_back: bool
    model_name: str | None
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    prompt_version: str = PROMPT_VERSION
    error: str | None = None


def _prompt(analysis: SpotAnalysis, action_taken: str, opponent: str) -> str:
    return "\n".join(
        [
            "FACTS",
            "-----",
            analysis.to_prompt_block(),
            "",
            f"Opponent archetype: {opponent}",
            f"The student chose: {action_taken}",
            "",
            "Give your read on this decision.",
        ]
    )


def critique(
    analysis: SpotAnalysis,
    action_taken: str,
    *,
    opponent: str,
    action_type: ActionType | None = None,
    model: LanguageModel | None = None,
    max_tokens: int = 400,
) -> CritiqueResult:
    """Build feedback for one decision, asking ``model`` for the read.

    With ``model=None`` the arithmetic and open questions are still produced —
    the deterministic half needs no model, which is what makes the fallback in
    step 4 honest rather than an empty screen.
    """

    if model is None:
        return CritiqueResult(
            feedback=build_feedback(analysis, action_taken, action_type=action_type),
            raw_responses=[],
            grounding=None,
            repair_attempted=False,
            fell_back=False,
            model_name=None,
            input_tokens=0,
            output_tokens=0,
            latency_seconds=0.0,
        )

    system = PRACTICE_SYSTEM_PROMPT
    history = [Message.user(_prompt(analysis, action_taken, opponent))]
    raw: list[str] = []
    started = time.perf_counter()
    tokens_in = tokens_out = 0

    try:
        response = model.complete(system, history, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        return CritiqueResult(
            feedback=build_feedback(
                analysis, action_taken, action_type=action_type,
                warnings=[f"the coach could not be reached: {exc}"],
            ),
            raw_responses=[], grounding=None, repair_attempted=False,
            fell_back=True, model_name=getattr(model, "name", None),
            input_tokens=0, output_tokens=0,
            latency_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    raw.append(response.text)
    tokens_in += response.usage.input_tokens
    tokens_out += response.usage.output_tokens
    report = check_grounding(response.text, analysis)
    repaired = False

    if not report.is_grounded:
        # One repair attempt, quoting exactly what failed.
        repaired = True
        claims = ", ".join(c.text for c in report.ungrounded)
        history.append(Message.assistant(response.text))
        history.append(
            Message.user(
                _REPAIR_TEMPLATE.format(count=len(report.ungrounded), claims=claims)
            )
        )
        try:
            response = model.complete(system, history, max_tokens=max_tokens)
            raw.append(response.text)
            tokens_in += response.usage.input_tokens
            tokens_out += response.usage.output_tokens
            report = check_grounding(response.text, analysis)
        except Exception as exc:  # noqa: BLE001
            return CritiqueResult(
                feedback=build_feedback(
                    analysis, action_taken, action_type=action_type,
                    grounding=report,
                    repair_attempted=True, fell_back=True,
                    warnings=[f"the repair request failed: {exc}"],
                ),
                raw_responses=raw, grounding=report, repair_attempted=True,
                fell_back=True, model_name=response.model if raw else None,
                input_tokens=tokens_in, output_tokens=tokens_out,
                latency_seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    elapsed = time.perf_counter() - started
    fell_back = not report.is_grounded
    warnings: list[str] = []
    if response.stop_reason == "refusal":
        warnings.append("the model declined to answer this one")
        fell_back = True
    if fell_back and not warnings:
        warnings.append(
            f"withheld after a repair attempt: {report.summary()}"
        )

    return CritiqueResult(
        feedback=build_feedback(
            analysis, action_taken, action_type=action_type,
            interpretation=None if fell_back else response.text,
            grounding=report,
            repair_attempted=repaired,
            fell_back=fell_back,
            warnings=warnings,
        ),
        raw_responses=raw,
        grounding=report,
        repair_attempted=repaired,
        fell_back=fell_back,
        model_name=response.model,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        latency_seconds=elapsed,
    )
