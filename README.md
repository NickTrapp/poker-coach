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

## Practice

```bash
poker-coach practice --hands 5 --opponent station
poker-coach practice --opponent lag --provider gemini --seed 7
```

Play hands against an archetype; every decision you make is critiqued. Feedback
is split by what justifies it, because those parts do not deserve equal
confidence:

```
VERIFIED CALCULATIONS
(Deterministically computed from a sampled equity estimate; respect the
 stated margin of error.)
  Showdown equity vs the assumed range: 32.25% (400 samples, ±4.58).
  Price: 0.5 to call into 1.5, so 25.0% equity breaks even.
  You raised. The only comparison computed here is calling against folding,
  which does not evaluate your raise. For reference: calling beats folding
  by 0.15 chips (an upper bound; it assumes the hand checks down and hero
  realises all of its equity).

EXPLOITATIVE INTERPRETATION
(The coach's reading. Not proven, and only as good as the assumed range.)
  ...

UNRESOLVED
(Nothing above settles these.)
  Calling vs raising: not evaluated — no figure above compares them
```

The first and third sections need **no model at all**, which is why
`--provider none` (the default) still teaches you something and costs nothing.

"Deterministic" is not "exact": when equity was sampled, everything derived
from it inherits that sampling error, and the heading says so. A terminal call
has an exact decision *tree* — no betting follows — but its EV is still an
estimate if the equity feeding it was one.

### The opponent's range moves with its actions

Normally, turning "this player is a station" into "here is what it holds having
bet the flop" needs a strategy model — so the range would have to stay a fixed
assumption. But the opponent here is not a person, it is an **explicit policy**,
so the coach can ask it directly: put each possible holding in its seat and see
whether it takes the action it took.

```
  villain's range: 194 of 1225 combos (check preflop, then bet on the flop)
  equity vs the assumed range: 14.5% (±2.4)
```

Ace-high on `6c8c3h` is worth 58% before the opponent acts and **14.5%** after
its flop bet is folded in. A fixed pre-action range would have quoted the first
number for a decision that only exists because of the second.

Each action narrows what the last one left, and the counts say how much:

```
  flop   194 of 1225 combos (check preflop, then bet on the flop)
  turn   101 of 1225 combos (..., then bet on the turn)
```

Three things this is **not**:

- It is a fact about the *policy*, not about a person. A real opponent is not
  this station, and the provenance line says so at every decision.
- It is a **distribution**, not a set. An opponent that bluffs 60% of the time
  holds its bluffs at less weight than its value hands, and the equity engine
  measures against the weighted posterior. Flattening it to "which combos are
  in" put a maniac's bluffs at 24.1% of its betting range instead of 16.0% —
  worth 2.4 equity points, twice the margin of error the figure carried.
- It is not exact. The likelihood is a *plug-in* estimate: the policy's branch
  is deterministic given an equity estimate, but it gets that estimate from
  Monte Carlo, so the equity ranking behind it is sampled, so a holding within sampling error of a threshold can fall either
  way. On the turn and river the equity itself *is* exact — the range is a
  finite combo list, so there is nothing to sample — but it is exact arithmetic
  on an uncertain input, and the heading says so rather than claiming the
  figure is deterministic.
- It is not always available. If no holding left in the range takes the action
  the opponent just took, conditioning stops, the last good range stands, and
  the fallback is reported rather than hidden.

```
Range assumption  : starts at random, narrowed by each action
Conditioning      : action-conditioned
Source            : the simulated station's own policy — true of it, not of a person
```

Pass `--fixed-range` for the older behaviour: one configured range, labelled
pre-action, never inferred from play.

If the coach cites a number it wasn't given, it gets one repair request quoting
the exact failed claims. If it fails again its prose is withheld and the
arithmetic shown alone — a noisy checker shouldn't silently delete useful
feedback, and it shouldn't launder invented figures either.

Every decision is logged to `practice/session-*.jsonl`: seed, full state, legal
actions, your action, the facts block, every raw model response including the
repair, the grounding verdict, resolved model id, and latency.

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
