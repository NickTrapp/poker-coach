#!/usr/bin/env python
"""Run the coaching benchmark against a real model.

Never reads, prints, or writes your API key — the SDK picks it up from the
environment.

    python scripts/benchmark.py --provider gemini --model gemini-flash-latest
    python scripts/benchmark.py --provider anthropic --dry-run
    python scripts/benchmark.py --report runs/gemini-20260722.jsonl

Costs one short request per case (nine by default).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from poker_coach.evaluation import (  # noqa: E402
    STANDARD_CASES,
    load_jsonl,
    render_report,
    run_benchmark,
    write_jsonl,
)
from poker_coach.models.base import EchoModel  # noqa: E402

PROVIDER_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def load_model(provider: str, model_id: str | None):
    if provider == "gemini":
        from poker_coach.models import GeminiModel

        return GeminiModel(model=model_id) if model_id else GeminiModel()
    from poker_coach.models import AnthropicModel

    return AnthropicModel(model=model_id) if model_id else AnthropicModel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDER_ENV), default="gemini")
    parser.add_argument("--model", default=None, help="Model id to request.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--out", default=None, help="JSONL output path.")
    parser.add_argument(
        "--case", action="append", default=None,
        help="Run only this case id (repeatable).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use the offline stub model — exercises the whole path, no API call.",
    )
    parser.add_argument(
        "--report", default=None,
        help="Render a report from an existing JSONL file and exit.",
    )
    args = parser.parse_args()

    if args.report:
        print(render_report(load_jsonl(args.report)))
        return 0

    cases = STANDARD_CASES
    if args.case:
        wanted = set(args.case)
        cases = tuple(c for c in STANDARD_CASES if c.name in wanted)
        if not cases:
            print(f"No cases matched {sorted(wanted)}", file=sys.stderr)
            return 2

    if args.dry_run:
        model, provider, requested = EchoModel(), "stub", "echo"
    else:
        names = PROVIDER_ENV[args.provider]
        if not any(os.environ.get(n) for n in names):
            print(f"No credentials for {args.provider}.", file=sys.stderr)
            print(f"Set one of: {', '.join(names)}", file=sys.stderr)
            print("\n(Set it in your shell — do not paste it into a chat.)",
                  file=sys.stderr)
            return 2
        try:
            model = load_model(args.provider, args.model)
        except ImportError as exc:
            print(exc, file=sys.stderr)
            return 2
        provider, requested = args.provider, args.model or model.name

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"runs/{provider}-{stamp}.jsonl")

    print(f"running {len(cases)} case(s) against {provider}/{requested}", flush=True)

    def progress(record):
        mark = "ERROR" if record.error else ("pass" if record.passed else "FAIL")
        print(f"  {mark:5s} {record.case_id}", flush=True)

    records = run_benchmark(
        model,
        provider=provider,
        requested_model=requested,
        cases=cases,
        seed=args.seed,
        iterations=args.iterations,
        on_result=progress,
    )

    write_jsonl(records, out)
    print(f"\nwrote {out}\n")
    print(render_report(records))

    return 0 if all(r.passed for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
