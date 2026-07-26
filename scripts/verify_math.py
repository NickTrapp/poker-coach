#!/usr/bin/env python
"""Ground-truth checks for the evaluator and the equity sampler.

These are the two checks that establish the deterministic layer is actually
correct rather than merely self-consistent. They were documented as "run these
by hand" for months, which meant nobody ran them. Twenty seconds is cheap
enough to run on every push instead.

    python scripts/verify_math.py
    python scripts/verify_math.py --quick    # skip the full enumeration

Exit code 0 when everything matches.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poker_coach.calculations.equity import equity  # noqa: E402
from poker_coach.calculations.hand_eval import HandCategory, evaluate  # noqa: E402
from poker_coach.domain.cards import FULL_DECK  # noqa: E402

#: The published frequency of each five-card hand category among all
#: C(52,5) = 2,598,960 hands. Independent of this codebase — that is the point.
EXPECTED_COUNTS: dict[HandCategory, int] = {
    HandCategory.STRAIGHT_FLUSH: 40,
    HandCategory.FOUR_OF_A_KIND: 624,
    HandCategory.FULL_HOUSE: 3_744,
    HandCategory.FLUSH: 5_108,
    HandCategory.STRAIGHT: 10_200,
    HandCategory.THREE_OF_A_KIND: 54_912,
    HandCategory.TWO_PAIR: 123_552,
    HandCategory.PAIR: 1_098_240,
    HandCategory.HIGH_CARD: 1_302_540,
}

#: Turn spots: 44 unseen cards means one river card, so equity enumerates
#: exactly and the sampler can be checked against the true answer.
SAMPLER_SPOTS = [
    ("AsKs", "7h7d", "Qs2s9c4d"),
    ("AhQd", "AsKs", "Jc7h2d5s"),
    ("Ts9s", "AhKd", "8s7h2c3d"),
    ("2s2h", "AsKd", "Qc7h9d4s"),
    ("AsKh", "QhQd", "Ac5s8d2h"),
]


def check_hand_distribution() -> bool:
    """Every five-card hand, counted by category, against the known figures."""

    print("Enumerating all 2,598,960 five-card hands...")
    started = time.perf_counter()
    counts = Counter(evaluate(hand).category for hand in combinations(FULL_DECK, 5))
    elapsed = time.perf_counter() - started

    ok = True
    print(f"{'category':20s} {'counted':>10s} {'expected':>10s}")
    for category in sorted(EXPECTED_COUNTS, reverse=True):
        counted = counts[category]
        expected = EXPECTED_COUNTS[category]
        flag = "" if counted == expected else "  <-- MISMATCH"
        if counted != expected:
            ok = False
        print(f"{category.label:20s} {counted:10,} {expected:10,}{flag}")

    total = sum(counts.values())
    if total != 2_598_960:
        print(f"total {total:,} != 2,598,960")
        ok = False

    print(f"\n{'PASS' if ok else 'FAIL'} — {elapsed:.1f}s")
    return ok


def check_sampler_against_enumeration() -> bool:
    """The Monte Carlo path must be unbiased against the enumerated truth."""

    print("\nComparing the sampler against exact enumeration on turn spots...")
    ok = True
    print(f"{'spot':32s} {'exact':>8s} {'sampled':>8s} {'diff':>7s} {'2*MoE':>7s}")

    for hero, villain, board in SAMPLER_SPOTS:
        exact = equity(hero, villain, board)
        sampled = equity(
            hero, villain, board,
            iterations=40_000, rng=random.Random(5), exact_limit=0,
        )
        assert exact.exact and not sampled.exact

        diff = sampled.equity - exact.equity
        tolerance = 2 * sampled.margin_of_error
        flag = "" if abs(diff) <= tolerance else "  <-- OUTSIDE"
        if abs(diff) > tolerance:
            ok = False
        print(
            f"{hero} vs {villain} on {board:10s} "
            f"{exact.equity_pct:7.2f}% {sampled.equity_pct:7.2f}% "
            f"{100 * diff:+6.2f} {100 * tolerance:6.2f}{flag}"
        )

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the full enumeration (~20s) and check the sampler only.",
    )
    args = parser.parse_args()

    results = []
    if not args.quick:
        results.append(check_hand_distribution())
    results.append(check_sampler_against_enumeration())

    print()
    if all(results):
        print("All ground-truth checks passed.")
        return 0
    print("GROUND-TRUTH CHECK FAILED — do not adjust a test expectation to match.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
