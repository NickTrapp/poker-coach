"""Command-line entry point.

Exposes the Phase 1 math directly, which makes the deterministic layer usable
(and spot-checkable by hand) before any model is wired up.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Sequence

from .calculations.equity import equity
from .calculations.hand_eval import best_five, evaluate
from .calculations.pot_odds import (
    bluff_success_threshold,
    ev_call,
    minimum_defence_frequency,
    pot_odds,
)
from .calculations.ranges import Range
from .coaching.analysis import analyze
from .coaching.review import build_review_facts
from .domain.cards import cards_to_str, parse_cards
from .domain.enums import Position, Street
from .domain.history import HandHistory
from .domain.state import HandState, PlayerState
from .evaluation.grounding import check_grounding


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poker-coach",
        description="No-limit hold'em coaching tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eq = sub.add_parser("equity", help="Equity of a hand against opponents.")
    eq.add_argument("hero", help="Hero's hole cards, e.g. AsKs")
    eq.add_argument(
        "villains",
        nargs="+",
        help="Opponent combos (AhKh) or ranges ('77+, AQs+'), one per opponent.",
    )
    eq.add_argument("--board", default="", help="Board cards, e.g. Qs2s9c")
    eq.add_argument("--iterations", type=int, default=20_000)
    eq.add_argument("--seed", type=int, default=None)

    ev = sub.add_parser("eval", help="Evaluate the best five-card hand.")
    ev.add_argument("cards", help="Five to seven cards, e.g. AsKsQsJsTs2h")

    od = sub.add_parser("odds", help="Pot odds, MDF and calling EV.")
    od.add_argument("pot", type=float, help="Pot size including the bet faced.")
    od.add_argument("to_call", type=float, help="Amount needed to call.")
    od.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Hero equity as a percentage, to get the EV of calling.",
    )

    rg = sub.add_parser("range", help="Expand range notation.")
    rg.add_argument("notation", help="e.g. '77+, AQs+, -55'")
    rg.add_argument("--combos", action="store_true", help="List every combo.")

    rv = sub.add_parser(
        "review",
        help="Replay a saved hand and print the facts behind each hero decision.",
    )
    rv.add_argument("path", help="Path to a hand-history JSON file.")
    rv.add_argument(
        "--villain-range",
        default="random",
        help="Range assumption for equity, e.g. '22+, ATs+, AJo+'.",
    )
    rv.add_argument("--iterations", type=int, default=10_000)
    rv.add_argument("--seed", type=int, default=None)
    rv.add_argument(
        "--replay",
        action="store_true",
        help="Print the full action sequence instead of the decision facts.",
    )

    gr = sub.add_parser(
        "check",
        help="Check a coaching response for numbers that aren't in the facts.",
    )
    gr.add_argument(
        "response",
        help="The response text to check, or '-' to read from stdin.",
    )
    gr.add_argument("--hero", required=True, help="Hero's hole cards, e.g. AsKs")
    gr.add_argument("--board", default="", help="Board cards, e.g. Qs2s9c")
    gr.add_argument("--pot", type=float, required=True, help="Total pot size.")
    gr.add_argument("--to-call", type=float, default=0.0, help="Amount to call.")
    gr.add_argument("--villain-range", default="random")
    gr.add_argument("--iterations", type=int, default=10_000)
    gr.add_argument("--seed", type=int, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "equity":
        rng = random.Random(args.seed) if args.seed is not None else None
        result = equity(
            args.hero,
            list(args.villains),
            args.board,
            iterations=args.iterations,
            rng=rng,
        )
        print(f"Equity: {result}")
        print(f"  win {100 * result.win:.2f}%  "
              f"tie {100 * result.tie:.2f}%  "
              f"lose {100 * result.lose:.2f}%")
        return 0

    if args.command == "eval":
        cards = parse_cards(args.cards)
        rank = evaluate(cards)
        print(f"{rank.description}  ({cards_to_str(best_five(cards))})")
        return 0

    if args.command == "odds":
        odds = pot_odds(args.pot, args.to_call)
        print(odds)
        pot_before_bet = max(0.0, args.pot - args.to_call)
        print(f"  MDF: {100 * minimum_defence_frequency(pot_before_bet, args.to_call):.1f}%")
        print(f"  Alpha: {100 * bluff_success_threshold(pot_before_bet, args.to_call):.1f}%")
        if args.equity is not None:
            ev = ev_call(args.pot, args.to_call, args.equity / 100.0)
            print(f"  EV of calling at {args.equity:g}% equity: {ev:+.2f} chips")
        return 0

    if args.command == "range":
        rng_obj = Range(args.notation)
        combos = rng_obj.combos()
        print(f"{len(combos)} combos ({rng_obj.percent_of_hands:.1f}% of hands)")
        if args.combos:
            print(" ".join(cards_to_str(c) for c in combos))
        else:
            print(" ".join(rng_obj.classes))
        return 0

    if args.command == "review":
        history = HandHistory.from_json_file(args.path)

        if args.replay:
            for point in history.replay():
                print(point)
            print(f"\nFinal pot: {history.final_state().total_pot:g}")
            return 0

        rng = random.Random(args.seed) if args.seed is not None else None
        facts = build_review_facts(
            history,
            villain_range=args.villain_range,
            iterations=args.iterations,
            rng=rng,
        )
        print(facts.to_prompt_block())
        return 0

    if args.command == "check":
        text = sys.stdin.read() if args.response == "-" else args.response
        rng = random.Random(args.seed) if args.seed is not None else None

        board = parse_cards(args.board) if args.board else []
        street = {0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}
        if len(board) not in street:
            raise ValueError(f"a board of {len(board)} cards is not a street")

        state = HandState(
            players=[
                PlayerState(
                    name="hero",
                    position=Position.BTN,
                    stack=max(args.pot, args.to_call) * 10,
                    hole_cards=tuple(parse_cards(args.hero)),
                    is_hero=True,
                ),
                PlayerState(
                    name="villain",
                    position=Position.BB,
                    stack=max(args.pot, args.to_call) * 10,
                    committed_this_street=args.to_call,
                ),
            ],
            board=board,
            street=street[len(board)],
            pot=args.pot - args.to_call,
        )

        analysis = analyze(
            state,
            villain_range=args.villain_range,
            iterations=args.iterations,
            rng=rng,
        )
        report = check_grounding(text, analysis)
        print(report.summary())
        for claim in report.ungrounded:
            print(f"  not in facts: {claim}")
        return 0 if report.is_grounded else 1

    return 1  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
