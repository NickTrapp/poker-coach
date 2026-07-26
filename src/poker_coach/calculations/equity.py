"""Equity calculation by exact enumeration or Monte Carlo sampling.

The entry point is :func:`equity`, which runs the hero's hand against one or
more opponents. Each opponent is either a specific combo or a
:class:`~poker_coach.calculations.ranges.Range`.

Enumeration is used automatically when it is cheap enough (all opponents known
and few board runouts left), so river and turn spots come back exact. Otherwise
the calculation falls back to sampling, and :attr:`EquityResult.exact` records
which path ran — the coaching layer surfaces that so it never presents a
sampled number as a certainty.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from ..domain.cards import Card, parse_cards, remaining_deck
from .hand_eval import evaluate
from .ranges import Range

__all__ = ["EquityResult", "equity", "equity_grid", "equity_vs_random"]

Combo = tuple[Card, Card]
Opponent = "Range | str | Sequence[Card]"

#: Enumerate rather than sample when the runout count is at or below this.
DEFAULT_EXACT_LIMIT = 30_000


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Hero's share of the pot at showdown, as fractions in ``[0, 1]``."""

    equity: float
    win: float
    tie: float
    lose: float
    samples: int
    exact: bool

    @property
    def equity_pct(self) -> float:
        return 100.0 * self.equity

    @property
    def margin_of_error(self) -> float:
        """Approximate 95% confidence half-width. Zero for exact results."""

        if self.exact or self.samples == 0:
            return 0.0
        var = max(self.equity * (1.0 - self.equity), 1e-12)
        return 1.96 * math.sqrt(var / self.samples)

    def __str__(self) -> str:
        if self.exact:
            return f"{self.equity_pct:.2f}% (exact)"
        return (
            f"{self.equity_pct:.2f}% ±{100 * self.margin_of_error:.2f} "
            f"({self.samples:,} samples)"
        )


def _as_combo(value: Sequence[Card] | str) -> Combo:
    cards = parse_cards(value) if isinstance(value, str) else list(value)
    if len(cards) != 2:
        raise ValueError(f"a hole-card combo needs exactly 2 cards, got {len(cards)}")
    if cards[0] == cards[1]:
        raise ValueError("a combo cannot use the same card twice")
    return (cards[0], cards[1])


def _normalize_opponent(value: Opponent) -> Range | Combo:
    if isinstance(value, Range):
        return value
    if isinstance(value, str):
        try:
            return _as_combo(value)
        except ValueError:
            return Range(value)
    return _as_combo(value)


def _showdown_share(hero_score: int, villain_scores: Sequence[int]) -> float:
    best = max(villain_scores)
    if hero_score > best:
        return 1.0
    if hero_score < best:
        return 0.0
    winners = 1 + sum(1 for s in villain_scores if s == hero_score)
    return 1.0 / winners


def equity(
    hero: Sequence[Card] | str,
    villains: Sequence[Opponent] | Opponent,
    board: Sequence[Card] | str = (),
    *,
    iterations: int = 20_000,
    rng: random.Random | None = None,
    exact_limit: int = DEFAULT_EXACT_LIMIT,
) -> EquityResult:
    """Hero's equity against ``villains`` on ``board``.

    ``villains`` may be a single opponent or a sequence of them. Strings are
    read as a combo when they parse as one (``"AsKd"``) and as range notation
    otherwise (``"77+, AQs+"``).
    """

    rng = rng or random.Random()

    hero_cards = _as_combo(hero)
    board_cards = parse_cards(board) if isinstance(board, str) else list(board)
    if len(board_cards) > 5:
        raise ValueError(f"board cannot exceed 5 cards, got {len(board_cards)}")

    if isinstance(villains, (Range, str)) or (
        villains and isinstance(villains[0], Card)
    ):
        villains = [villains]  # type: ignore[list-item]
    opponents = [_normalize_opponent(v) for v in villains]
    if not opponents:
        raise ValueError("need at least one opponent")

    known = [*hero_cards, *board_cards]
    for opp in opponents:
        if isinstance(opp, tuple):
            known.extend(opp)
    if len(set(known)) != len(known):
        raise ValueError("duplicate cards among hero, board and opponents")

    opponents = _collapse_singletons(opponents, {*hero_cards, *board_cards})

    needed = 5 - len(board_cards)
    all_known = set(opponents_known(opponents)) | set(hero_cards) | set(board_cards)
    deck = remaining_deck(sorted(all_known, key=str))

    fixed = all(isinstance(o, tuple) for o in opponents)
    runouts = math.comb(len(deck), needed) if needed else 1
    if fixed and runouts <= exact_limit:
        return _enumerate(hero_cards, opponents, board_cards, deck, needed)  # type: ignore[arg-type]

    return _simulate(
        hero_cards, opponents, board_cards, needed, iterations, rng
    )


def _collapse_singletons(
    opponents: Sequence[Range | Combo], dead: set[Card]
) -> list[Range | Combo]:
    """Replace ranges holding exactly one legal combo with that combo.

    A one-combo range *is* a known hand — collapsing it lets the spot take the
    exact enumeration path instead of being sampled. This is what makes
    "villain turned over AhKh" come back exact.
    """

    blocked = set(dead)
    for opp in opponents:
        if isinstance(opp, tuple):
            blocked.update(opp)

    out: list[Range | Combo] = []
    for opp in opponents:
        if isinstance(opp, Range):
            candidates = opp.combos(dead=tuple(blocked))
            if len(candidates) == 1:
                opp = candidates[0]
                blocked.update(opp)
        out.append(opp)
    return out


def opponents_known(opponents: Sequence[Range | Combo]) -> list[Card]:
    out: list[Card] = []
    for opp in opponents:
        if isinstance(opp, tuple):
            out.extend(opp)
    return out


def _enumerate(
    hero: Combo,
    opponents: Sequence[Combo],
    board: Sequence[Card],
    deck: Sequence[Card],
    needed: int,
) -> EquityResult:
    total = 0
    equity_sum = 0.0
    wins = ties = losses = 0

    runouts = combinations(deck, needed) if needed else [()]
    for runout in runouts:
        full_board = [*board, *runout]
        hero_score = evaluate([*hero, *full_board]).score
        villain_scores = [evaluate([*opp, *full_board]).score for opp in opponents]

        share = _showdown_share(hero_score, villain_scores)
        equity_sum += share
        if share == 1.0:
            wins += 1
        elif share == 0.0:
            losses += 1
        else:
            ties += 1
        total += 1

    return EquityResult(
        equity=equity_sum / total,
        win=wins / total,
        tie=ties / total,
        lose=losses / total,
        samples=total,
        exact=True,
    )


def _simulate(
    hero: Combo,
    opponents: Sequence[Range | Combo],
    board: Sequence[Card],
    needed: int,
    iterations: int,
    rng: random.Random,
) -> EquityResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    base_dead = {*hero, *board}
    # Pre-expand each range once; blockers from hero/board never change.
    pools: list[list[Combo] | Combo] = []
    for opp in opponents:
        if isinstance(opp, tuple):
            pools.append(opp)
            base_dead.update(opp)
        else:
            pool = opp.combos(dead=tuple(base_dead))
            if not pool:
                raise ValueError(f"range {opp} has no combos after blockers")
            pools.append(pool)

    equity_sum = 0.0
    wins = ties = losses = 0
    completed = 0

    for _ in range(iterations):
        dead, villain_hands = _draw_joint(pools, base_dead, rng)
        if villain_hands is None:
            continue

        deck = [c for c in remaining_deck(sorted(dead, key=str))]
        runout = rng.sample(deck, needed) if needed else []
        full_board = [*board, *runout]

        hero_score = evaluate([*hero, *full_board]).score
        villain_scores = [evaluate([*h, *full_board]).score for h in villain_hands]

        share = _showdown_share(hero_score, villain_scores)
        equity_sum += share
        if share == 1.0:
            wins += 1
        elif share == 0.0:
            losses += 1
        else:
            ties += 1
        completed += 1

    if completed == 0:
        raise ValueError("no valid deals were possible for these ranges")

    return EquityResult(
        equity=equity_sum / completed,
        win=wins / completed,
        tie=ties / completed,
        lose=losses / completed,
        samples=completed,
        exact=False,
    )


#: Joint re-draws attempted before giving up on one iteration.
_JOINT_ATTEMPTS = 40


def _draw_joint(
    pools: Sequence[list[Combo] | Combo],
    base_dead: set[Card],
    rng: random.Random,
) -> tuple[set[Card], list[Combo] | None]:
    """Draw one combo per opponent, uniformly over *joint* assignments.

    Each opponent is drawn independently from its own full pool and the whole
    deal is rejected if any two collide. That matters: drawing opponents in
    sequence and conditioning each on the previous ones is **not** uniform over
    legal joint assignments — it over-weights assignments in which the earlier
    opponent blocks more of the later opponent's range, and makes the answer
    depend on the order the ranges were supplied.

    With ``opp1 ∈ {AsKs, 2c3c}`` and ``opp2 ∈ {AsQs, AsJs, 2c4c}`` there are
    three legal joint assignments. Sequential sampling gives the first a
    probability of 1/2; uniform gives 1/3. Measured on the old implementation
    it was 0.496.

    Every legal assignment here has probability ``∏ 1/|pool_i|`` before
    rejection, so conditioning on acceptance leaves them uniform.
    """

    for _ in range(_JOINT_ATTEMPTS):
        dead = set(base_dead)
        hands: list[Combo] = []
        ok = True

        for pool in pools:
            if isinstance(pool, tuple):
                # A pinned combo is already in `base_dead`, so it would always
                # "collide" with itself. Nothing to draw and nothing to reject.
                hands.append(pool)
                continue
            combo = pool[rng.randrange(len(pool))]
            if combo[0] in dead or combo[1] in dead:
                ok = False
                break
            dead.update(combo)
            hands.append(combo)

        if ok:
            return dead, hands

    return set(base_dead), None


def _draw_combo(
    pool: Sequence[Combo], dead: set[Card], rng: random.Random, tries: int = 24
) -> Combo | None:
    """Pick a combo from ``pool`` that avoids ``dead``, or None if we can't."""

    for _ in range(tries):
        combo = pool[rng.randrange(len(pool))]
        if combo[0] not in dead and combo[1] not in dead:
            return combo
    available = [c for c in pool if c[0] not in dead and c[1] not in dead]
    if not available:
        return None
    return available[rng.randrange(len(available))]


def equity_vs_random(
    hero: Sequence[Card] | str,
    board: Sequence[Card] | str = (),
    *,
    opponents: int = 1,
    iterations: int = 20_000,
    rng: random.Random | None = None,
) -> EquityResult:
    """Convenience wrapper: hero against ``opponents`` random hands."""

    return equity(
        hero,
        [Range("random") for _ in range(opponents)],
        board,
        iterations=iterations,
        rng=rng,
    )


def equity_grid(
    heroes: Sequence[Combo],
    villain: Range,
    board: Sequence[Card] = (),
    *,
    iterations: int = 400,
    rng: random.Random | None = None,
) -> dict[Combo, float]:
    """Equity for *many* hero combos against one range, sharing the sampling.

    Conditioning a range on an observed action means asking what every combo in
    it would have done, which means an equity figure per combo. Doing that with
    one `equity()` call each costs ~12ms apiece — 5.3 seconds for a 424-combo
    station range, per decision. Almost all of it is setup: expanding the range,
    rebuilding the deck, re-deriving blockers.

    Here the villain hand and runout are drawn once per iteration and scored
    against every hero combo that does not collide with them, so the whole grid
    costs about what a single call used to. Combos are scored on a shared set of
    deals rather than independent ones, which correlates their errors — fine for
    ranking combos against a threshold, which is what conditioning needs.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = rng or random.Random()
    board = list(board)
    needed = 5 - len(board)
    if needed < 0:
        raise ValueError(f"board cannot exceed 5 cards, got {len(board)}")

    hero_list = [tuple(h) for h in heroes]
    totals = {hero: 0.0 for hero in hero_list}
    counts = {hero: 0 for hero in hero_list}

    pool = villain.combos(dead=tuple(board))
    if not pool:
        raise ValueError(f"range {villain} has no combos after blockers")

    for _ in range(iterations):
        villain_hand = pool[rng.randrange(len(pool))]
        dead = {*board, *villain_hand}
        deck = remaining_deck(sorted(dead, key=str))
        runout = rng.sample(deck, needed) if needed else []
        full_board = [*board, *runout]
        used = {*full_board, *villain_hand}

        villain_score = evaluate([*villain_hand, *full_board]).score

        for hero in hero_list:
            if hero[0] in used or hero[1] in used:
                continue  # this deal is impossible for that combo
            hero_score = evaluate([*hero, *full_board]).score
            totals[hero] += _showdown_share(hero_score, [villain_score])
            counts[hero] += 1

    return {
        hero: (totals[hero] / counts[hero]) if counts[hero] else 0.0
        for hero in hero_list
    }
