#!/usr/bin/env python
"""One live call against a real provider, to answer the only open question.

Everything in this repo is verified up to the wire and untested past it. This
script closes that gap: it sends one real coaching turn and checks whether the
model (a) reaches a sane recommendation and (b) obeys the "quote, don't compute"
rule that the whole design rests on.

It never reads, prints, or writes your API key — the SDK picks it up from the
environment. Cost is one short request.

    python scripts/live_check.py --list-models
    python scripts/live_check.py
    python scripts/live_check.py --provider anthropic
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poker_coach.coaching import Coach  # noqa: E402
from poker_coach.domain import (  # noqa: E402
    Action,
    ActionType,
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)
from poker_coach.evaluation import check_grounding, recommended_action  # noqa: E402

PROVIDERS = {
    "gemini": {
        "env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "extra": "gemini",
    },
    "anthropic": {
        "env": ("ANTHROPIC_API_KEY",),
        "extra": "anthropic",
    },
}


def build_spot() -> HandState:
    """A flop spot where the arithmetic clearly favours calling."""

    state = HandState(
        players=[
            PlayerState(
                name="hero",
                position=Position.BTN,
                stack=97.0,
                hole_cards=tuple(parse_cards("AsKs")),
                is_hero=True,
            ),
            PlayerState(name="villain", position=Position.BB, stack=97.0),
        ],
        board=parse_cards("Qs2s9c"),
        street=Street.FLOP,
        pot=6.0,
    )
    return state.apply(
        Action(actor="villain", type=ActionType.BET, amount=6.0, street=Street.FLOP)
    )


def check_credentials(provider: str) -> str | None:
    names = PROVIDERS[provider]["env"]
    found = next((n for n in names if os.environ.get(n)), None)
    if found:
        return found
    print(f"No credentials found for {provider}.", file=sys.stderr)
    print(f"Set one of: {', '.join(names)}", file=sys.stderr)
    print(f"\n  export {names[0]}='...'", file=sys.stderr)
    print("\n(Set it in your shell — do not paste it into a chat.)", file=sys.stderr)
    return None


def load_model(provider: str, model_id: str | None):
    extra = PROVIDERS[provider]["extra"]
    try:
        if provider == "gemini":
            from poker_coach.models import GeminiModel

            return GeminiModel(model=model_id) if model_id else GeminiModel()
        from poker_coach.models import AnthropicModel

        return AnthropicModel(model=model_id) if model_id else AnthropicModel()
    except ImportError:
        print(
            f'SDK not installed. Run:  pip install -e ".[{extra}]"',
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    parser.add_argument("--model", default=None, help="Override the model id.")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List reachable model ids and exit (verifies the default is real).",
    )
    parser.add_argument("--iterations", type=int, default=20_000)
    args = parser.parse_args()

    if not check_credentials(args.provider):
        return 2

    if args.list_models:
        if args.provider != "gemini":
            print("--list-models is implemented for gemini only.", file=sys.stderr)
            return 2
        from poker_coach.models.gemini_client import list_models

        for name in list_models():
            print(name)
        return 0

    model = load_model(args.provider, args.model)
    coach = Coach(model=model, iterations=args.iterations, rng=random.Random(11))

    print(f"provider : {args.provider}")
    print(f"model    : {model.name}")
    print()

    advice = coach.advise(build_spot(), villain_range="22+, ATs+, KQs, AJo+")

    print("--- FACTS GIVEN TO THE MODEL " + "-" * 30)
    print(advice.facts)
    print()
    print("--- WHAT THE MODEL SAID " + "-" * 35)
    print(advice.text.strip())
    print()

    # The two things worth measuring.
    action = recommended_action(advice.text)
    report = check_grounding(advice.text, advice.analysis)
    ev = advice.analysis.ev_of_calling
    expected = "call" if ev > 0 else "fold"

    print("--- VERDICT " + "-" * 47)
    print(f"recommendation : {action}  (arithmetic favours {expected}, EV {ev:+.2f})")
    print(f"grounding      : {report.summary()}")
    for claim in report.ungrounded:
        print(f"    invented: {claim}")

    usage = advice.response.usage
    print(f"tokens         : {usage.input_tokens} in / {usage.output_tokens} out")
    if advice.response.stop_reason == "refusal":
        print("NOTE: the model declined this request.")

    ok = action == expected and report.is_grounded
    print()
    print("PASS — sane recommendation, every number traceable" if ok else "FAIL — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
