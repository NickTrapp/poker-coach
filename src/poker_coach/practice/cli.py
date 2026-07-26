"""The terminal front end for a practice session.

Kept apart from :mod:`poker_coach.practice.session` so the session itself has
no opinion about input or output — the CLI supplies `on_turn`, tests supply a
scripted callback, and neither knows about the other.
"""

from __future__ import annotations

import random
from typing import TextIO

from ..domain.action import Action
from ..domain.betting import LegalAction
from ..domain.enums import ActionType
from ..models.base import LanguageModel
from .critique import CritiqueResult
from .log import DecisionRecord, session_path, write_records
from .session import HeroTurn, PracticeConfig, PracticeSession

__all__ = ["run_practice", "render_turn", "parse_action"]

_ABBREVIATIONS = {
    "f": ActionType.FOLD,
    "c": ActionType.CALL,
    "k": ActionType.CHECK,
    "b": ActionType.BET,
    "r": ActionType.RAISE,
}


def render_turn(turn: HeroTurn, hero: str) -> str:
    """The board, the price, and the arithmetic — before any model is asked."""

    analysis = turn.analysis
    lines = [
        "",
        "=" * 68,
        f"{analysis.street.upper()}   board {analysis.board or '(none)'}   "
        f"your hand {analysis.hero_cards}",
        "=" * 68,
        f"  pot {analysis.total_pot:g}   to call {analysis.to_call:g}   "
        f"your stack {analysis.hero_stack_behind:g}",
    ]

    features = [d for d in analysis.features.descriptions()]
    if features:
        lines.append(f"  you have: {'; '.join(features)}")

    # Name the range before quoting an equity against it. "Vs the assumed
    # range" is fine when the range is fixed and printed in the header; once it
    # moves with every action, the figure below is meaningless without it.
    assumption = analysis.range_assumption
    if assumption.narrowing is not None:
        lines.append(
            f"  villain's range: {assumption.narrowing} combos "
            f"({assumption.observed_line})"
        )

    lines.append(
        f"  equity vs the assumed range: {analysis.equity.equity_pct:.1f}%"
        + ("" if analysis.equity.exact else
           f" (±{100 * analysis.equity.margin_of_error:.1f})")
    )
    if analysis.facing_bet:
        lines.append(
            f"  price: need {analysis.odds.required_equity_pct:.1f}% to break even"
        )

    lines.append("")
    lines.append("  your options:")
    for option in turn.legal:
        lines.append(f"    {_option_help(option)}")
    return "\n".join(lines)


def _option_help(option: LegalAction) -> str:
    letter = next(k for k, v in _ABBREVIATIONS.items() if v is option.type)
    if option.min_amount == option.max_amount:
        if option.min_amount:
            return f"[{letter}] {option.type.value} {option.min_amount:g}"
        return f"[{letter}] {option.type.value}"
    return (
        f"[{letter}] {option.type.value} <amount>   "
        f"({option.min_amount:g} to {option.max_amount:g})"
    )


def parse_action(text: str, turn: HeroTurn, hero: str) -> Action:
    """Turn "r 12" into a legal Action, or explain why it isn't one."""

    parts = text.strip().lower().split()
    if not parts:
        raise ValueError("say something — try 'f', 'c', 'k', or 'r 12'")

    word = parts[0]
    kind = _ABBREVIATIONS.get(word)
    if kind is None:
        for candidate in ActionType:
            if candidate.value == word:
                kind = candidate
                break
    if kind is None:
        raise ValueError(f"{word!r} is not an action")

    options = [o for o in turn.legal if o.type is kind]
    if not options:
        offered = ", ".join(sorted(o.type.value for o in turn.legal))
        raise ValueError(f"you cannot {kind.value} here; legal: {offered}")

    option = options[0]
    if option.min_amount == option.max_amount:
        amount = option.min_amount
    elif len(parts) < 2:
        raise ValueError(
            f"{kind.value} needs an amount between "
            f"{option.min_amount:g} and {option.max_amount:g}"
        )
    else:
        try:
            amount = float(parts[1])
        except ValueError:
            raise ValueError(f"{parts[1]!r} is not a number") from None
        if not option.permits(amount):
            raise ValueError(
                f"{kind.value} must be between {option.min_amount:g} and "
                f"{option.max_amount:g}"
            )

    return Action(
        actor=hero, type=kind, amount=amount, street=turn.state.street
    )


def run_practice(
    *,
    hands: int = 5,
    opponent: str = "station",
    model: LanguageModel | None = None,
    seed: int | None = None,
    stdin: TextIO,
    stdout: TextIO,
    log_dir: str | None = "practice",
    iterations: int = 6000,
    condition_on_action: bool = True,
) -> list[DecisionRecord]:
    """Run an interactive session. Returns every decision recorded."""

    config = PracticeConfig(
        opponent_style=opponent, iterations=iterations,
        condition_on_action=condition_on_action,
    )
    rng = random.Random(seed) if seed is not None else random.Random()

    def show(text: str = "") -> None:
        print(text, file=stdout)

    def on_turn(turn: HeroTurn) -> Action:
        show(render_turn(turn, config.hero_name))
        while True:
            print("  > ", end="", file=stdout, flush=True)
            line = stdin.readline()
            if not line:  # EOF — fold out rather than loop forever
                show("(no input; folding)")
                fold = [o for o in turn.legal if o.type is ActionType.FOLD]
                kind = fold[0].type if fold else turn.legal[0].type
                return Action(actor=config.hero_name, type=kind,
                              street=turn.state.street)
            try:
                return parse_action(line, turn, config.hero_name)
            except ValueError as exc:
                show(f"  {exc}")

    def on_feedback(turn: HeroTurn, result: CritiqueResult) -> None:
        show()
        show(result.feedback.render())

    session = PracticeSession(
        config=config, model=model, rng=rng,
        on_turn=on_turn, on_feedback=on_feedback,
    )

    show("PRACTICE")
    show(config.describe())
    if model is None:
        show(
            "Coach             : none — arithmetic only "
            "(pass --provider to add a read)"
        )
    else:
        show(f"Coach             : {getattr(model, 'name', 'model')}")

    for hand_number in range(1, hands + 1):
        before = len(session.records)
        result = session.play_hand()
        acted = len(session.records) - before

        show()
        if acted == 0:
            # Heads-up the small blind acts first, so a hand where the student
            # is big blind and the opponent folds never reaches them. Say so —
            # a hand that silently vanishes reads like a bug.
            show(f"--- hand {hand_number}: no decision for you ---")
            show(
                "    you were big blind and the opponent folded preflop, "
                "so the action never reached you"
            )
        else:
            show(f"--- hand {hand_number} over: {result} ---")
        net = result.net(config.hero_name)
        show(f"    you {'won' if net >= 0 else 'lost'} {abs(net):.2f} chips")

    show()
    show(f"{session.hands_played} hands, {len(session.records)} decisions "
         f"(hands where the opponent folded preflop to your big blind reach "
         f"you for none).")

    if log_dir and session.records:
        path = write_records(session_path(log_dir), session.records)
        show(f"Session log: {path}")

    return session.records
