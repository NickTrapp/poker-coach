"""Facts with their semantics attached.

A number alone is not honest. "EV of calling: +4.25 chips" is exact on a river
call and an assumption-laden estimate on the flop, and the first live run showed
a model treating the second as though it were the first.

So a fact carries what it is (:class:`FactCategory`), what had to be true for it
to hold (``assumptions``), where it came from (``provenance``), and what it does
*not* cover (``scope``). The prompt groups facts by category so the model can
see at a glance which figures are certain and which are conditional.

This is deliberately poker-shaped, not a general knowledge representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = ["FactCategory", "AnalysisFact", "render_fact_groups"]


class FactCategory(str, Enum):
    """What kind of claim a fact is — the thing that was previously implicit."""

    STATE = "state"
    """Exact fact about the hand as it stands. Not derived, not assumed."""

    DERIVED = "derived"
    """Exact calculation from state. True whenever the state is true."""

    SAMPLED = "sampled"
    """Monte Carlo estimate. Carries a margin of error; not exact."""

    ASSUMED = "assumed"
    """Correct *only* under a stated assumption. The assumption is the point."""

    REFERENCE = "reference"
    """A formula's output, offered as a benchmark. Not a prescription."""

    OPEN = "open"
    """A question the supplied calculations do not answer."""

    @property
    def heading(self) -> str:
        return _HEADINGS[self]

    @property
    def caveat(self) -> str:
        return _CAVEATS[self]


_HEADINGS: dict[FactCategory, str] = {
    FactCategory.STATE: "EXACT STATE FACTS",
    FactCategory.DERIVED: "EXACT CALCULATIONS",
    FactCategory.SAMPLED: "SAMPLED CALCULATIONS",
    FactCategory.ASSUMED: "ASSUMPTION-CONDITIONED CALCULATIONS",
    FactCategory.REFERENCE: "THEORETICAL REFERENCE VALUES",
    FactCategory.OPEN: "QUESTIONS THESE FACTS DO NOT ANSWER",
}

_CAVEATS: dict[FactCategory, str] = {
    FactCategory.STATE: "True by observation.",
    FactCategory.DERIVED: "Exact given the state above.",
    FactCategory.SAMPLED: (
        "Estimated by simulation. Do not draw conclusions finer than the "
        "stated margin of error."
    ),
    FactCategory.ASSUMED: (
        "Each figure holds only under the assumption printed beside it. If the "
        "assumption is wrong, the figure is wrong — say so when it matters."
    ),
    FactCategory.REFERENCE: (
        "Formula outputs, not strategy. They describe a break-even point under "
        "the stated conditions; they do not say what anyone should actually do."
    ),
    FactCategory.OPEN: (
        "Nothing above settles these. Do not present a conclusion about them as "
        "though the numbers established it."
    ),
}


@dataclass(frozen=True, slots=True)
class AnalysisFact:
    """One value handed to the model, with its semantics."""

    key: str
    rendered_value: str
    category: FactCategory
    assumptions: tuple[str, ...] = ()
    provenance: str = ""
    scope: str | None = None

    def render(self) -> str:
        """One line, with provenance, assumptions and scope inline.

        Provenance is rendered because it is the difference between "56.9% by
        exact enumeration" and "56.9%, sampled" — two figures that look
        identical and warrant different confidence. It was carried on the
        dataclass but dropped here, so the model saw only the category heading;
        that made "supplied by the caller" and "derived from the opponent's
        policy" indistinguishable, which is precisely the distinction a range
        assumption exists to make.
        """

        line = f"{self.key}: {self.rendered_value}"
        if self.provenance:
            line += f" [source: {self.provenance}]"
        if self.assumptions:
            line += f" [assumes: {'; '.join(self.assumptions)}]"
        if self.scope:
            line += f" [scope: {self.scope}]"
        return line

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.render()


def render_fact_groups(facts: list[AnalysisFact]) -> str:
    """Group facts by category under headings, in decreasing certainty."""

    order = [
        FactCategory.STATE,
        FactCategory.DERIVED,
        FactCategory.SAMPLED,
        FactCategory.ASSUMED,
        FactCategory.REFERENCE,
        FactCategory.OPEN,
    ]

    blocks: list[str] = []
    for category in order:
        group = [fact for fact in facts if fact.category is category]
        if not group:
            continue
        lines = [f"{category.heading}", f"({category.caveat})"]
        lines += [f"  {fact.render()}" for fact in group]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
