#!/usr/bin/env python
"""Play many hands and assert the invariants that only break in volume.

Every bug found in the table runner so far needed hundreds of hands to surface:
an illegal big-blind option (32 of 300), a float underflow and a forbidden
re-raise (2 of 500), a round-completion disagreement (46 of 1500). No
single-hand test would have caught any of them, which is why this runs in CI.

    python scripts/stress_hands.py
    python scripts/stress_hands.py --hands 3000 --seed 99

Stacks are deliberately mixed so side pots actually trigger. Exit code 0 when
every hand holds every invariant.
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poker_coach.domain.enums import Position  # noqa: E402
from poker_coach.players import STYLES, Seat, make_player, play_hand  # noqa: E402

SEATING = [
    Position.SB,
    Position.BB,
    Position.UTG,
    Position.HJ,
    Position.CO,
    Position.BTN,
]

TOLERANCE = 1e-6


def check_hand(seats, result) -> list[str]:
    """Every invariant a completed hand must satisfy."""

    problems: list[str] = []
    state = result.final_state

    contributed = sum(p.committed_total for p in state.players)
    if abs(contributed - result.pot) > TOLERANCE:
        problems.append(f"pot {result.pot} != chips contributed {contributed}")

    awarded = sum(result.awards.values())
    if abs(awarded - result.pot) > TOLERANCE:
        problems.append(f"awards {awarded} != pot {result.pot}")

    if abs(sum(result.net(s.name) for s in seats)) > TOLERANCE:
        problems.append("results are not zero-sum")

    if not result.awards:
        problems.append("nobody won the pot")

    for player in state.players:
        if player.stack < -TOLERANCE:
            problems.append(f"{player.name} has a negative stack {player.stack}")

    # Nobody may win more than they could cover from each opponent.
    for seat in seats:
        mine = state.player(seat.name).committed_total
        cap = sum(min(p.committed_total, mine) for p in state.players)
        if result.awards.get(seat.name, 0.0) - cap > TOLERANCE:
            problems.append(f"{seat.name} won more than it covered")

    # The recorded hand must replay under strict rules to the same pot.
    try:
        replayed = result.history.final_state()
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        problems.append(f"history does not replay: {type(exc).__name__}: {exc}")
    else:
        if abs(replayed.total_pot - result.pot) > TOLERANCE:
            problems.append("replayed pot differs from the played pot")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--iterations", type=int, default=80,
                        help="Monte Carlo iterations per opponent decision.")
    args = parser.parse_args()

    styles = list(STYLES)
    stats: collections.Counter = collections.Counter()
    failures: list[tuple[int, int, str]] = []

    for index in range(args.hands):
        seat_count = 2 + (index % 5)
        rng = random.Random(args.seed + index)
        seats = [
            Seat(
                make_player(
                    f"p{i}", styles[(index + i) % len(styles)],
                    rng=rng, iterations=args.iterations,
                ),
                SEATING[i],
                # Mixed stacks, so short stacks go all-in and side pots form.
                100.0 if i % 3 else 18.0,
            )
            for i in range(seat_count)
        ]

        try:
            result = play_hand(seats, rng=rng)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append((index, seat_count, f"{type(exc).__name__}: {exc}"))
            continue

        for problem in check_hand(seats, result):
            failures.append((index, seat_count, problem))

        stats[f"{seat_count}-handed"] += 1
        stats["showdown" if result.showdown else "no showdown"] += 1

    played = sum(v for k, v in stats.items() if k.endswith("-handed"))
    print(f"hands played : {played} / {args.hands}")
    print("breakdown    :", dict(sorted(stats.items())))
    print(f"failures     : {len(failures)}")

    for index, seat_count, problem in failures[:15]:
        print(f"  hand {index} ({seat_count}-handed): {problem}")
    if len(failures) > 15:
        print(f"  ... and {len(failures) - 15} more")

    if failures or played != args.hands:
        print("\nSTRESS RUN FAILED")
        return 1
    print("\nAll invariants held across every hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
