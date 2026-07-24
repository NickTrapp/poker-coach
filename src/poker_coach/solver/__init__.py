"""Solver-derived baselines. (Phase 3)

Intended shape: a `Baseline` lookup mapping (spot, action) to a GTO frequency
and EV, so the coach can say "this is a 30%-of-range bluff" rather than
gesturing at balance. Sources are precomputed tables, not a live solver.

Deliberately empty in Phase 2 — `calculations` already covers the exact math
the coach quotes today, and a baseline store is only useful once the coaching
prompt is proven against it.
"""

__all__: list[str] = []
