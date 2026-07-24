"""Lightweight span tracing.

Enough to answer "what did that coaching call actually do, and where did the
time go" without pulling in an observability stack. Spans nest, record wall
time and arbitrary tags, and capture exceptions without swallowing them.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["Span", "Trace"]


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            return 0.0
        return 1000.0 * (self.end - self.start)

    def __str__(self) -> str:
        indent = "  " * self.depth
        tags = " ".join(f"{k}={v}" for k, v in self.tags.items())
        suffix = f" [{tags}]" if tags else ""
        status = f" ERROR: {self.error}" if self.error else ""
        return f"{indent}{self.name} {self.duration_ms:.1f}ms{suffix}{status}"


@dataclass
class Trace:
    """A flat, ordered record of spans with recorded nesting depth."""

    spans: list[Span] = field(default_factory=list)
    enabled: bool = True
    _depth: int = 0

    @contextmanager
    def span(self, name: str, **tags: Any) -> Iterator[Span]:
        if not self.enabled:
            yield Span(name=name, start=0.0, end=0.0, tags=dict(tags))
            return

        record = Span(
            name=name, start=time.perf_counter(), tags=dict(tags), depth=self._depth
        )
        self.spans.append(record)
        self._depth += 1
        try:
            yield record
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._depth -= 1
            record.end = time.perf_counter()

    @property
    def total_ms(self) -> float:
        """Wall time of the top-level spans only, so nesting isn't double-counted."""

        return sum(s.duration_ms for s in self.spans if s.depth == 0)

    def find(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]

    def clear(self) -> None:
        self.spans.clear()
        self._depth = 0

    def render(self) -> str:
        return "\n".join(str(span) for span in self.spans)

    def __len__(self) -> int:
        return len(self.spans)
