"""The coach: state in, explanation out.

Phase 2 wires the whole path — state -> deterministic analysis -> prompt ->
model -> advice — against the :class:`~poker_coach.models.base.LanguageModel`
protocol. Swapping in a real client is a constructor argument, not a code
change, and the stub models in `poker_coach.models` make the path testable
without a network.

What is deliberately *not* here yet (Phase 3): range estimation from villain
tendencies, solver-baseline comparison, and knowledge-base retrieval. Each has
a named seam below so the shape does not have to change when they land.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..calculations.ranges import Range
from ..domain.history import HandHistory
from ..domain.state import HandState
from ..models.base import LanguageModel, Message, ModelResponse
from ..tracing.trace import Trace
from .analysis import DEFAULT_VILLAIN_RANGE, SpotAnalysis, analyze
from .prompts.system import (
    COACH_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    render_system_prompt,
)
from .review import HandReviewFacts, build_review_facts

__all__ = ["Advice", "Coach", "Review"]


@dataclass(frozen=True, slots=True)
class Advice:
    """A coaching response plus the facts it was grounded in."""

    text: str
    analysis: SpotAnalysis
    response: ModelResponse

    @property
    def facts(self) -> str:
        return self.analysis.to_prompt_block()

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text


@dataclass(frozen=True, slots=True)
class Review:
    """A post-hoc review of a played hand, plus the facts behind each call."""

    text: str
    facts: HandReviewFacts
    response: ModelResponse

    @property
    def decisions(self):
        return self.facts.decisions

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text


@dataclass
class Coach:
    """Coaches a student through spots, holding conversation history.

    A `Coach` is bound to one student conversation. Create a new one per
    session; the model instance itself is stateless and can be shared.
    """

    model: LanguageModel
    style: str | None = None
    default_villain_range: str = DEFAULT_VILLAIN_RANGE
    iterations: int = 10_000
    rng: random.Random = field(default_factory=random.Random)
    trace: Trace = field(default_factory=Trace)
    history: list[Message] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        return render_system_prompt(COACH_SYSTEM_PROMPT, style=self.style)

    def analyse(
        self,
        state: HandState,
        *,
        hero: str | None = None,
        villain_range: str | Range | None = None,
    ) -> SpotAnalysis:
        """Run the deterministic half only — useful on its own, and cheap."""

        with self.trace.span("analyse"):
            return analyze(
                state,
                hero=hero,
                villain_range=villain_range or self.default_villain_range,
                iterations=self.iterations,
                rng=self.rng,
            )

    def advise(
        self,
        state: HandState,
        *,
        question: str | None = None,
        hero: str | None = None,
        villain_range: str | Range | None = None,
    ) -> Advice:
        """Analyse the spot and ask the model to explain the decision."""

        analysis = self.analyse(state, hero=hero, villain_range=villain_range)
        prompt = self._build_prompt(analysis, question)

        self.history.append(Message.user(prompt))
        with self.trace.span("model.complete", model=self.model.name):
            response = self.model.complete(self.system_prompt, self.history)
        self.history.append(Message.assistant(response.text))

        return Advice(text=response.text, analysis=analysis, response=response)

    def review(
        self,
        history: HandHistory,
        *,
        villain_range: str | Range | None = None,
        question: str | None = None,
    ) -> Review:
        """Replay a played hand and critique each of the hero's decisions.

        Uses its own system prompt and does *not* touch conversation history:
        a review is a self-contained artefact, not a turn in a dialogue.
        """

        with self.trace.span("build_review_facts"):
            facts = build_review_facts(
                history,
                villain_range=villain_range or self.default_villain_range,
                iterations=self.iterations,
                rng=self.rng,
            )

        prompt = "\n".join(
            [
                "FACTS",
                "-----",
                facts.to_prompt_block(),
                "",
                question.strip()
                if question
                else "Review each decision and name the biggest leak.",
            ]
        )

        system = render_system_prompt(REVIEW_SYSTEM_PROMPT, style=self.style)
        with self.trace.span("model.complete", model=self.model.name, kind="review"):
            response = self.model.complete(system, [Message.user(prompt)])

        return Review(text=response.text, facts=facts, response=response)

    def ask(self, question: str) -> ModelResponse:
        """Ask a follow-up in the current conversation, with no new analysis."""

        if not self.history:
            raise ValueError("no spot has been analysed yet; call advise() first")
        self.history.append(Message.user(question))
        with self.trace.span("model.complete", model=self.model.name):
            response = self.model.complete(self.system_prompt, self.history)
        self.history.append(Message.assistant(response.text))
        return response

    def reset(self) -> None:
        """Clear conversation history, keeping configuration and the model."""

        self.history.clear()

    @staticmethod
    def _build_prompt(analysis: SpotAnalysis, question: str | None) -> str:
        parts = ["FACTS", "-----", analysis.to_prompt_block(), ""]
        if question:
            parts.append(question.strip())
        else:
            parts.append("What should the hero do here, and why?")
        return "\n".join(parts)
