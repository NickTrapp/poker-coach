"""Deterministic poker math.

This layer is the coach's source of truth. Nothing here calls a model or the
network: given the same inputs it always returns the same numbers (Monte Carlo
paths accept an explicit `rng` so they can be pinned in tests). The coaching
layer is expected to quote these results rather than compute arithmetic itself.
"""

from .equity import EquityResult, equity, equity_vs_random
from .hand_eval import HandCategory, HandRank, best_five, compare, evaluate
from .pot_odds import (
    PotOdds,
    bet_as_pot_fraction,
    bluff_success_threshold,
    break_even_fold_equity,
    ev_bluff,
    ev_call,
    minimum_defence_frequency,
    outs_to_equity,
    pot_odds,
    required_equity,
    stack_to_pot_ratio,
)
from .ranges import Range, canonical_class, class_combos, expand_classes

__all__ = [
    "EquityResult",
    "HandCategory",
    "HandRank",
    "PotOdds",
    "Range",
    "best_five",
    "bet_as_pot_fraction",
    "bluff_success_threshold",
    "break_even_fold_equity",
    "canonical_class",
    "class_combos",
    "compare",
    "equity",
    "equity_vs_random",
    "ev_bluff",
    "ev_call",
    "evaluate",
    "expand_classes",
    "minimum_defence_frequency",
    "outs_to_equity",
    "pot_odds",
    "required_equity",
    "stack_to_pot_ratio",
]
