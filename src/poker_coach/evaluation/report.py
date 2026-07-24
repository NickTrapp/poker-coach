"""Turn benchmark records into something a human can actually review.

The automatic section is deterministic only. The manual section lists the
findings that need a person — each one paired with the text to look at and a
concrete question — because the alternative is an automated strategic judge
whose confidence nobody has earned.

Deterministic signals here are *heuristics for where to look*, not verdicts.
`ev-treated-as-realised` flags a response that cites a non-terminal EV figure
without ever mentioning realisation; that is a strong hint and not a proof, and
the report says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .benchmark import RunRecord

__all__ = ["ReviewFlag", "flag_record", "render_report", "MANUAL_CHECKS"]

#: Findings that need a human. Each is a question, not an assertion.
MANUAL_CHECKS: tuple[tuple[str, str], ...] = (
    ("incorrect-hand-feature",
     "Does the response assert any draw, texture, or blocker that the facts "
     "did not supply, or contradict one that they did?"),
    ("conclusion-exceeds-arithmetic",
     "Does it claim more than the numbers support — e.g. that calling is best, "
     "when only continue-vs-fold was computed?"),
    ("unjustified-call-vs-raise",
     "Does it compare calling with raising as though a figure settled it?"),
    ("range-assumption-unacknowledged",
     "Does it lean on an equity number without naming the range assumption, or "
     "treat a pre-action range as villain's betting range?"),
    ("equity-vs-realised-ev",
     "Does it treat showdown equity as value hero actually realises?"),
    ("overconfident",
     "Is the recommendation stated with more certainty than the spot warrants?"),
    ("low-value-principle",
     "Is the transferable principle generic filler, or would it change a "
     "future decision?"),
)

_REALISATION_WORDS = (
    "realis", "realiz", "future street", "later street", "implied",
    "check down", "checked down", "further betting", "next street",
)
_RANGE_WORDS = ("range", "assum", "read")
_RAISE_WORDS = ("raise", "raising", "shove", "jam")
_HEDGE_WORDS = ("if ", "unless", "depends", "assum", "would change", "close")


@dataclass(frozen=True, slots=True)
class ReviewFlag:
    """A place worth looking, with why."""

    code: str
    detail: str
    automatic: bool


def _mentions(text: str, words: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def flag_record(record: RunRecord) -> list[ReviewFlag]:
    """Deterministic signals only. These locate problems; they do not judge."""

    flags: list[ReviewFlag] = []
    text = record.response

    if record.error:
        return [ReviewFlag("error", record.error, automatic=True)]

    if not record.grounded:
        flags.append(
            ReviewFlag(
                "invented-numbers",
                f"claims absent from the facts: {', '.join(record.ungrounded_claims)}",
                automatic=True,
            )
        )

    if not record.action_correct:
        flags.append(
            ReviewFlag(
                "wrong-side-of-the-line",
                f"expected {record.expected_verdict}, response reads as "
                f"{record.verdict or 'unclear'}",
                automatic=True,
            )
        )

    if record.recommendation == "unclear":
        flags.append(
            ReviewFlag("no-clear-recommendation",
                       "no leading action word found", automatic=True)
        )

    # Non-terminal EV cited with no mention of realisation. A hint, not a proof.
    if record.ev_is_terminal is False and text:
        cites_ev = bool(re.search(r"[+-]?\d+\.\d+\s*chips", text)) or "ev" in text.lower()
        if cites_ev and not _mentions(text, _REALISATION_WORDS):
            flags.append(
                ReviewFlag(
                    "ev-treated-as-realised",
                    "cites a non-terminal EV figure without mentioning "
                    "realisation or future betting — check whether it reads as "
                    "settled value",
                    automatic=True,
                )
            )

    if text and not _mentions(text, _RANGE_WORDS):
        flags.append(
            ReviewFlag("range-unmentioned",
                       "no mention of the range assumption anywhere in the reply",
                       automatic=True)
        )

    if record.range_conditioning == "pre-action" and text:
        if not _mentions(text, ("pre-action", "before", "betting range", "narrow")):
            flags.append(
                ReviewFlag(
                    "pre-action-range-unqualified",
                    "the supplied range was pre-action, and the reply does not "
                    "note that villain's betting range is narrower",
                    automatic=True,
                )
            )

    if record.recommendation == "call" and _mentions(text, _RAISE_WORDS):
        flags.append(
            ReviewFlag(
                "discusses-raising",
                "mentions raising — check whether it claims the arithmetic "
                "settles call-vs-raise",
                automatic=True,
            )
        )

    if text and not _mentions(text, _HEDGE_WORDS):
        flags.append(
            ReviewFlag("no-qualification",
                       "no conditional language anywhere — check for "
                       "overconfidence", automatic=True)
        )

    return flags


def render_report(records: Sequence[RunRecord]) -> str:
    """A readable report: summary, per-case detail, then the manual checklist."""

    if not records:
        return "No records."

    passed = [r for r in records if r.passed]
    errored = [r for r in records if r.error]
    lines: list[str] = []

    first = records[0]
    lines += [
        "=" * 78,
        "COACHING BENCHMARK REPORT",
        "=" * 78,
        f"provider        : {first.provider}",
        f"requested model : {first.requested_model}",
        f"resolved model  : {first.resolved_model or '(not reported)'}",
        f"prompt version  : {first.prompt_version}",
        f"seed            : {first.seed}",
        f"cases           : {len(records)}",
        f"passed          : {len(passed)}/{len(records)}"
        + (f"   errored: {len(errored)}" if errored else ""),
        "",
        "A pass means: correct side of the continue/fold line, and every number",
        "traceable to the facts. It does NOT mean the reasoning was good — see",
        "the manual checklist at the end.",
        "",
    ]

    # ---- table -----------------------------------------------------------
    lines.append(f"{'case':36s} {'exp':9s} {'got':8s} {'ground':7s} {'tok':>6s} {'sec':>6s}")
    lines.append("-" * 78)
    for record in records:
        status = "ERROR" if record.error else ("pass" if record.passed else "FAIL")
        lines.append(
            f"{record.case_id:36s} {record.expected_verdict:9s} "
            f"{(record.verdict or record.recommendation):8s} "
            f"{'ok' if record.grounded else 'BAD':7s} "
            f"{record.output_tokens:6d} {record.latency_seconds:6.1f}  {status}"
        )
    lines.append("")

    # ---- per-case detail --------------------------------------------------
    for record in records:
        flags = flag_record(record)
        lines += ["=" * 78, f"CASE: {record.case_id}"]
        if record.case_description:
            lines.append(f"  {record.case_description}")
        lines += [
            f"  expected {record.expected_verdict} | read {record.recommendation} "
            f"({record.verdict or 'unclear'}) | "
            f"EV terminal: {record.ev_is_terminal} | "
            f"equity exact: {record.equity_exact} | "
            f"range: {record.range_conditioning}",
            "",
        ]
        if record.error:
            lines += [f"  ERROR: {record.error}", ""]
            continue

        lines += ["--- response " + "-" * 64, record.response.strip(), ""]
        lines.append(f"  grounding: {record.grounding_summary}")
        if flags:
            lines.append("  automatic flags (places to look, not verdicts):")
            for flag in flags:
                lines.append(f"    [{flag.code}] {flag.detail}")
        else:
            lines.append("  automatic flags: none")
        lines.append("")

    # ---- manual checklist -------------------------------------------------
    lines += [
        "=" * 78,
        "MANUAL REVIEW CHECKLIST",
        "=" * 78,
        "These need a human. They are not automated because doing so would mean",
        "building a strategic judge whose accuracy has not been established.",
        "",
    ]
    for code, question in MANUAL_CHECKS:
        lines.append(f"  [{code}]")
        lines.append(f"      {question}")
    lines.append("")

    return "\n".join(lines)
