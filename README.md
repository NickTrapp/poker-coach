# poker-coach

An AI-native no-limit hold'em coaching system.

The central idea: **the model explains, it never calculates.** Every number a
coaching response cites — equity, pot odds, EV, MDF — is computed by a
deterministic layer and handed to the model as a given fact. A separate checker
then verifies the response didn't invent any figures. A right answer reached by
fabricating numbers is treated as a failure.

## Install

Requires Python 3.12+.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

The model clients are optional — the deterministic layers, the CLI, and the
whole test suite work without either:

```bash
.venv/bin/pip install -e ".[anthropic]"    # Claude
.venv/bin/pip install -e ".[gemini]"       # Gemini
```

Run the tests:

```bash
.venv/bin/python -m pytest -q
```

Two checks live outside the unit suite because they take ~20s each, and CI runs
both on every push:

```bash
.venv/bin/python scripts/verify_math.py    # all 2,598,960 hands + sampler bias
.venv/bin/python scripts/stress_hands.py   # 1200 simulated hands, all invariants
```

## Quickstart

```python
import random
from poker_coach.coaching import Coach
from poker_coach.models import EchoModel          # or AnthropicModel
from poker_coach.domain import *

state = HandState(
    players=[
        PlayerState(name="hero", position=Position.BTN, stack=97.0,
                    hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
        PlayerState(name="villain", position=Position.BB, stack=97.0),
    ],
    board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=6.0,
)
state = state.apply(
    Action(actor="villain", type=ActionType.BET, amount=6.0, street=Street.FLOP)
)

coach = Coach(model=EchoModel(), rng=random.Random(1))
advice = coach.advise(state, villain_range="22+, ATs+, KQs, AJo+")
print(advice.facts)
```

That prints the block the model receives — the coach's entire factual basis,
grouped by how much weight each figure can carry:

```
EXACT STATE FACTS
(True by observation.)
  Street: flop
  Board: Qs2s9c
  Pot: 12 [scope: includes the bet hero is facing]
  Hero stack behind: 97 [scope: chips hero has left right now, after villain's bet]
  ...

EXACT CALCULATIONS
  Flush draw: nut flush draw (s)
  Overcards to the board: 2 (AK)
  SPR: 8.08 [scope: effective start-of-street stack / current pot]
  Pot odds: 2.00:1 (need 33.3% equity to break even on a call)

SAMPLED CALCULATIONS
(Do not draw conclusions finer than the stated margin of error.)
  Hero equity at showdown: 55.92% ±0.97 [scope: NOT the share hero realises]

ASSUMPTION-CONDITIONED CALCULATIONS
  EV of calling vs folding, if equity were fully realised: +4.07 chips
    [assumes: the hand is checked down from here — NO further betting]
    [scope: compares CALLING with FOLDING only; it does not evaluate raising]

THEORETICAL REFERENCE VALUES
(Formula outputs, not strategy.)
  Minimum defence frequency for this bet size: 50.0%

QUESTIONS THESE FACTS DO NOT ANSWER
  Calling vs raising: not evaluated — no figure above compares them
  Equity realisation: not modelled — future streets may force hero off the hand
```

The groupings are load-bearing. A live run showed a model quoting every number
correctly while treating a checked-down estimate as settled value, so each fact
now states what it assumes and what it does not cover. See
[docs/semantics-audit.md](docs/semantics-audit.md).

## Choosing a model

Swap `EchoModel()` for a real client. Credentials come from the environment;
neither client accepts an API key as an argument, so one can't end up in a repr
or a log line.

```python
from poker_coach.models import AnthropicModel, GeminiModel

Coach(model=AnthropicModel())                      # ANTHROPIC_API_KEY
Coach(model=GeminiModel())                         # GEMINI_API_KEY / GOOGLE_API_KEY
Coach(model=GeminiModel(model="gemini-2.5-flash", temperature=0.2))
```

Nothing above `models/` knows which one it is. The contract is small enough to
write a third against in an afternoon:

```python
name: str
def complete(self, system: str, messages: Sequence[Message], *, max_tokens: int) -> ModelResponse
```

Two provider differences the adapters absorb, so callers never see them:

- **Sampling parameters.** Current Claude models *reject* `temperature`; Gemini
  accepts it. It is therefore an instance setting on `GeminiModel`, not a
  protocol argument.
- **Refusals.** A safety block is a successful response, not an exception, on
  both. Both clients normalise it to `stop_reason == "refusal"`.

Model ids move faster than this repo does — `poker_coach.models.gemini_client.list_models()`
shows what your key can actually reach.

## Command line

```bash
poker-coach eval AsKsQsJsTs2h3d           # best five-card hand
poker-coach equity AsKs 7h7d --board Qs2s9c4d
poker-coach equity AsAh "77+, AQs+" --iterations 20000 --seed 1
poker-coach odds 12 6 --equity 55         # pot odds, MDF, alpha, EV
poker-coach range "77+, AQs+, -55"        # expand range notation
poker-coach review examples/hands/thin-river-call.json --replay
poker-coach check "You have 71.8% equity." --hero AsKs --board Qs2s9c --pot 12 --to-call 6
```

`check` exits non-zero when a response cites a number that isn't in the facts,
so it works directly as a CI gate.

## Architecture

Layers are strict — lower layers never import higher ones.

| Layer | Role |
|---|---|
| `domain` | Cards, actions, hand state, recorded histories. Pure data and rules. |
| `calculations` | Hand evaluation, ranges, equity, pot odds. The source of truth. |
| `models` | The language-model seam: a protocol, test stubs, Claude and Gemini clients. |
| `coaching` | Turns state + numbers into explanation. Quotes, never computes. |
| `evaluation` | Verifies the quoting rule and benchmarks recommendations. |
| `players` | Rule-based opponent archetypes to practise against. |
| `tracing` | Span timing for "what did that call actually do". |
| `knowledge`, `solver` | Documented stubs — not yet built. |

### Why the split matters

`calculations` is deterministic and independently verified: the hand evaluator
is checked against the known 5-card hand distribution across all 2,598,960
hands, and the Monte Carlo sampler is checked against exact enumeration. Because
the coaching layer only ever *quotes* those results, a coaching bug can't
silently become a math bug.

Equity is computed exactly when enumeration is cheap and sampled otherwise, and
every result records which path ran — so the coach never presents a sampled
number as a certainty.

## Verifying groundedness

```python
from poker_coach.evaluation import check_grounding

report = check_grounding("Call. You have 88% equity.", advice.analysis)
report.is_grounded   # False
report.summary()     # "1 of 1 numeric claim(s) are not in the facts: 88%"
```

Known limits, stated precisely. A clean report means **every number quoted
appears somewhere in the facts** — not that it was attached to the right
quantity. With equity at 71.8% and MDF at 50%, "Hero has 50% equity" passes,
because 50 is a supplied percentage. Bare integers 0–10 are ignored as well
(too often prose — "two pair"), so a small invented count slips through, and
non-numeric claims are not checked at all. See the module docstring in
`evaluation/grounding.py` for why closing this needs fact-addressed generation
rather than a better regex.

## Benchmarking against a real model

```bash
python scripts/benchmark.py --dry-run                        # offline, no API call
python scripts/benchmark.py --provider gemini --model gemini-flash-latest
python scripts/benchmark.py --report runs/gemini-....jsonl   # re-render a past run
```

Nine cases spanning terminal and non-terminal EV, exact and sampled equity, and
pre-action versus action-conditioned ranges. Each run records the facts block,
the raw reply, the deterministic verdicts, token usage, latency, and the
**resolved** model version — never just the floating alias you asked for.

The report separates automatic flags (deterministic; they locate problems) from
a manual checklist (things like "is this principle actually useful?"), which is
left to a human because automating it would mean building a strategic judge
whose accuracy nobody has established.

## Hand histories

A `HandHistory` replays through the same state machine live play uses, so a
reviewed hand and a played hand cannot drift apart. Blinds and antes post
automatically from seat position, so a history file records only real decisions.

```python
from poker_coach.domain import HandHistory

hand = HandHistory.from_json_file("examples/hands/thin-river-call.json")
for point in hand.replay():
    print(point)          # "* [flop] hero bet 4"
hand.final_state().total_pot   # 94.0
```

## Opponents

Five rule-based archetypes, each exaggerating one recognisable leak:

```python
import random
from poker_coach.players import make_player

villain = make_player("villain", "station", rng=random.Random(1))
print(villain.act(state))    # "villain call 8"
```

Facing an 8-into-10 bet holding K-high on an ace-high board — a call needing
44.4% equity — the five styles split exactly where their names promise:

```
nit      -> fold      station  -> call
tag      -> fold      maniac   -> call
lag      -> fold
```

`nit`, `tag`, `lag`, `station`, `maniac`. They are **not** solver output and are
deliberately unbalanced — beating them proves nothing about GTO play. What they
are good for is practising against a known tendency, and their calling looseness
is strictly ordered so the difference is legible.

Every action they return is guaranteed legal for the given state.

### Playing hands out

`play_hand` deals and plays a full hand for two to nine seats, returning the
result plus a `HandHistory` that replays to the same final state — so a
simulated hand feeds straight into the review path:

```python
from poker_coach.players import Seat, make_player, play_hand
from poker_coach.domain import Position

rng = random.Random(1)
result = play_hand([
    Seat(make_player("sb", "lag", rng=rng), Position.SB, 100.0),
    Seat(make_player("bb", "station", rng=rng), Position.BB, 100.0),
], rng=rng, hero="bb")

print(result)                       # "bb wins 2.8 (no showdown)"
coach.review(result.history)        # critique every hero decision
```

Action order follows the seat ring: preflop opens left of the big blind and the
blinds close it; later streets open at the small blind and the button closes.
Heads-up is the usual exception — the small blind *is* the button, so it acts
first preflop and last afterwards.

**Side pots** are handled properly. When a short stack is all-in, chips it could
not match go to a separate pot it cannot win:

```
p4 all-in for 18 with two pair, kings and queens   -> wins  55.50 (main pot)
p3 covers, with two pair, kings and jacks          -> wins 164.00 (side pot)
```

The best hand won only the main pot, because that is all it paid for.

## Status

Built and tested: the domain model, all deterministic math, the coaching and
review paths, the model seam with a real Anthropic client, the grounding
checker, the opponent archetypes, and the table runner (2–9 seats, side pots).

Not built: `knowledge` (retrieval corpus) and `solver` (GTO baselines). Both
have docstrings describing their intended shape and why they were deferred —
each depends on how the coaching prompt actually performs against a real model,
which has not been measured yet.
