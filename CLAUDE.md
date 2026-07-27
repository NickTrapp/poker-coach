# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A no-limit hold'em coaching system. See `README.md` for the user-facing tour.
This file records the invariants and conventions that are easy to violate
without noticing.

## Commands

```bash
.venv/bin/python -m pytest -q                    # full suite (~13s)
.venv/bin/python -m pytest tests/calculations -q # one layer
.venv/bin/poker-coach --help                     # CLI
```

The venv is at `.venv/`. `anthropic` and `google-genai` are **optional** extras
— never import either at module scope, and never make the test suite depend on
one. Both are resolved lazily through `models/__init__.py`'s `__getattr__`.

## Providers

Two clients sit behind `LanguageModel`: `AnthropicModel` and `GeminiModel`.
Keeping both is deliberate — it is what stops provider assumptions leaking
upward. Nothing outside `models/` may import a vendor SDK.

Differences the adapters absorb, so callers never see them:

- **Sampling parameters.** Current Claude models reject `temperature`/`top_p`/
  `top_k` with a 400; Gemini accepts them. So `temperature` is an *instance*
  setting on `GeminiModel`, never a protocol argument. Do not add it to
  `LanguageModel.complete`.
- **Role naming.** Gemini calls the assistant role `"model"` and rejects
  `"assistant"`. The conversion lives in `GeminiModel.build_contents`.
- **Refusals.** Both providers return a refusal as a *successful* response, not
  an exception. Both normalise it to `stop_reason == "refusal"` so the coaching
  layer has one signal. Preserve that when adding a provider.

Verify SDK bindings against the installed package before writing client code —
both clients were written that way, and their request-shape tests construct
real SDK objects so an SDK change fails a test rather than a production call.

## The one rule everything rests on

**The model quotes numbers; it never computes them.**

`calculations/` produces every figure. `coaching/analysis.py` gathers them into
a `SpotAnalysis`. `coaching/coach.py` renders them into the prompt.
`evaluation/grounding.py` verifies the response invented nothing.

If you add a quantity the coach can cite, you must add it in **three** places or
the grounding checker will produce false positives:

1. Compute it in `calculations/`.
2. Surface it on `SpotAnalysis` and as an `AnalysisFact`.
3. Add it to `grounded_values()` in `evaluation/grounding.py`.

Step 3 is the one that gets forgotten. The test
`test_facts_block_is_self_grounded` catches it — the facts block must always
ground itself. (It also catches stray percentages in *prose*: an assumption
string reading "realises 100% of equity" fails, because 100 is not a supplied
value. Write "all of its equity" instead.)

## Facts carry their own semantics

A bare number is not honest. Every fact is an `AnalysisFact` with a
`FactCategory`, and the prompt groups by category — see
`docs/semantics-audit.md` for why. When adding a fact, choose the category
truthfully:

| Category | Means |
|---|---|
| `STATE` | exact fact about the hand |
| `DERIVED` | exact calculation from state |
| `SAMPLED` | Monte Carlo, carries a margin of error |
| `ASSUMED` | correct only under a stated assumption |
| `REFERENCE` | a formula's output, not a prescription |
| `OPEN` | something the numbers do not settle |

Three distinctions that were once implicit and are now enforced by tests:

- **Call EV compares calling with folding, never with raising.** Nothing in
  this package evaluates raising. A positive figure means continuing beats
  folding — say exactly that.
- **Call EV is exact only on a terminal call** (river, or the call puts someone
  all-in). Elsewhere it assumes the hand checks down with full equity
  realisation, and is labelled an upper bound. `_is_terminal_call` decides.
- **A range must state its conditioning.** `pre-action` and
  `action-conditioned` are different distributions. Never silently convert one
  to the other; that needs a strategy model this package does not have.

## Layering

Strict; lower layers never import higher ones.

```
domain → calculations → models → coaching → evaluation
```

`domain` and `calculations` have no LLM, network, or I/O dependencies. Keep it
that way — it is what makes them cheap to test and safe to import anywhere.

## Conventions that bite

**Action amounts are "to" amounts.** `Action.amount` is the total street
commitment *after* the action ("raise to 30"), not the chips added. Chips added
is derived at apply time. See `domain/action.py`.

**`HandState.apply()` returns a new state.** It never mutates. `advance_street`
builds the successor in one construction because `validate_assignment` rejects a
half-updated state (new board with the old street).

**Pot accounting.** `HandState.pot` is dead money from *previous* streets;
`total_pot` adds the live street bets. Antes go to `pot`, blinds to
`committed_this_street` — antes are not a live bet and must not raise the amount
to call.

**Three different stack quantities, and they are not interchangeable.**
`stack_behind(name)` is chips left right now; `stack_at_street_start(name)` adds
back what that player already committed this street; `effective_stack()` is the
*minimum* start-of-street stack across players. SPR uses the last one — passing
a single name to `effective_stack` returns that player's own stack, which is not
an effective stack at all and once produced a 10x-wrong SPR.

**Monte Carlo takes an explicit `rng`.** Every sampling path accepts a
`random.Random` so tests can pin it. Never call the module-level `random`.

**A draw must be hero's own, and must have a card to come.** `features.py`
counts an out only when hero's resulting hand beats what the *community cards
alone* would make — a board of JsTs7c9c makes J-T-9-8-7 with any eight, and
calling that hero's gutshot is wrong twice over. Draws are also suppressed on a
complete board: four to a flush on the river is not a draw, nothing is coming.
Both shipped to a live session before being caught.

**Equity reports its own provenance.** `EquityResult.exact` says whether the
result was enumerated or sampled. The coach surfaces this; don't drop it.

**Prompts are hard-wrapped prose.** Assertions on prompt text must collapse
whitespace first (`" ".join(text.split())`) or a reflow silently breaks them.

## Verifying math changes

```bash
python scripts/verify_math.py        # ~22s: full enumeration + sampler check
python scripts/verify_math.py --quick # sampler only
python scripts/stress_hands.py       # ~40s: 1200 hands, all invariants
```

`verify_math.py` enumerates all 2,598,960 five-card hands against the published
category distribution, and checks the Monte Carlo sampler against exact
enumeration on turn spots. `stress_hands.py` plays hands across 2-9 seats with
mixed stacks and asserts chips conserved, awards summing to the pot, zero-sum
results, and strict replay. CI runs both.

Simulated play only breaks in volume: the bugs found so far needed 300, 500 and
1500 hands to surface. Run `stress_hands.py` after touching the runner or the
player policy — no single-hand test substitutes.

**Enumeration is gated on total deals, not on runouts.** A range is a finite
set of combos, so `combos x runouts` is the cost — and a river spot against a
range is *always* enumerable. Gating on runouts alone made those spots sample:
a margin of error quoted on a quantity that has none, and slower than the exact
answer (379 combos evaluate in 3ms where 6,000 samples take 168ms). Multiway,
assignments where two opponents hold the same card are skipped, never repaired
by re-drawing one opponent — that is the same non-uniformity `_draw_joint`
guards against on the sampling side.

Do not "fix" an equity number by adjusting a test expectation. Published
per-class preflop equities are rounded and suit-generic; exact enumeration is
the authority.

## When the model cites a number you did not give it

The reflex is to widen `grounded_values()`. Resist it. The one live case —
a model citing "a total pot of 35" from a pot of 20 and a call of 15 — was the
checker being *right*: the facts implied that number but never stated it, and
"the model quotes, never computes" makes computing it a violation.

Allowing arithmetic closure has no stopping point: pot after calling, then pot
after a raise, then final stack, then percentage differences, and the rule is
gone. Add the quantity as a named `AnalysisFact` instead. It is usually worth
saying out loud anyway — `Pot after hero calls` is more useful to a student
than leaving them to do the addition.

## Poker notation in text processing

Anything that parses response text must mask card and range tokens before
reading numbers. Two silent-corruption bugs have already come from getting this
wrong (`33.3%` → `3%`, and `77.40%` → `40%`). Patterns require a distinguishing
marker — suit, face rank, `+`/`s`/`o`, or a dash range — and literal masks need
boundary guards. See `evaluation/grounding.py`.

## Betting legality

`domain/betting.py` owns what is legal. `state.legal_actions(name)` is the
source of truth — anything not listed there is rejected by `apply()`. Never
re-derive legality at a call site; ask.

Enforcement has two tiers, and the split is deliberate:

- **Always**: the action's street tag, the minimum raise, and that a street
  cannot be left with a bet unmatched. All derivable from state alone.
- **`strict=True` only**: turn order, and nothing else. It needs the street's
  action history, which a directly-built `HandState` cannot supply. Replay and
  the table runner pass it; test fixtures generally cannot.

`strict` does **not** gate raise-reopening — that lives in `legal_actions`,
which `validate` always consults, and degrades to permissive without history.
So `replay(strict=False)` relaxes turn order but still rejects a forbidden
re-raise. Without history the min-raise floor also drops to the big blind:
permissive, never wrong.

An all-in is always legal regardless of size. An under-raise all-in does *not*
reopen the betting for players who already acted.

## Simulated play

`players/table.play_hand` handles 2–9 seats. Two things carry the complexity:

- **Action order** comes from `action_order()`, which walks the seat ring.
  Preflop opens left of the big blind; later streets open at the small blind.
  Heads-up inverts the postflop order because the small blind is the button.
  **Whose turn it is comes from `BettingRound.next_actor`, which resumes
  clockwise from the last actor** — never rescan from the top of the order, or
  a raise puts the wrong player in multiway. The runner and the legality check
  must call the same function; when they each had their own copy, they shared
  a bug and 1500 replayed hands agreed with it.
- **Side pots** are computed in `_settle()` by walking the distinct
  contribution levels. Folded players *fund* a layer but are never eligible to
  win it. Never award the whole pot to the best hand — a short stack must not
  scoop chips it never covered.

Two more rules the runner depends on, both of which have already caused bugs:

- **Heads-up, the small blind is the button** — it acts first preflop and last
  on every later street.
- **Owing nothing is not the same as an unbet pot.** On the big blind's option
  the player owes nothing while its own blind is still a live bet, so
  aggression there must be a `RAISE`; a `BET` is illegal and the state machine
  rejects it.

When changing the runner or the player policy, run a few hundred hands across
archetype pairings **with mixed stack sizes** (so side pots actually trigger)
and assert: chips are conserved, awards sum to the pot, the history replays to
the same pot, results are zero-sum, and nobody wins more than they could cover.
Single-hand tests miss all of these.

## Range conditioning

`players/conditioning.py` inverts an archetype: given the action it took, which
holdings would take it? `P(H|A,S) ∝ P(A|H,S)·P(H|S)`, and the likelihood — the
term nobody normally has — is *computable* here because the opponent is an
explicit policy rather than a person.

Four rules:

- **Ask the policy; never re-derive it.** `RuleBasedPlayer.decide` and
  `action_probability` exist so conditioning consults the same rules the
  simulator plays. A second copy of the policy would drift, and this project has
  the scars.
- **Likelihoods multiply; chain, do not restart.** Pass a `ConditionedRange`
  back in as the prior. Chaining through `to_range()` flattens the weights and
  silently promotes a bluffing branch to full weight.
- **The prior is "random".** The dealer hands the opponent any two cards, and
  the archetype's preflop range gate is part of the policy, so conditioning
  *recovers* the open range rather than assuming it. Assuming it can contradict
  it — `DEFAULT_RANGES["station"]` and `STATION.open_range` differ by `96s+`.
- **A posterior derived from a policy is a fact about the policy.** It says what
  this configured station would hold, not what a person would. Say so wherever
  it surfaces.

**The likelihood is a plug-in estimate, not the exact quantity.** The policy's
branch is deterministic *conditional on an equity estimate*, but the policy
gets that estimate from Monte Carlo and thresholds it — so unconditionally the
action is random over the policy's own sampling. `P(A|H,S)` is really
`P(Ê(H,S) lands on the side giving A)`; this takes one draw of `Ê` and reads
off 0 or 1. Do not describe it as exact. Combos within sampling error of a
threshold resolve arbitrarily, sometimes against the hand the opponent actually
holds. `tests/players/test_conditioning.py` checks structure exactly and
agreement tolerantly, on purpose.

**Weights are the posterior; the support is not.** `to_range()` keeps only
which combos survived and is lossy the moment the policy bluffs.
`to_weighted()` returns a `WeightedRange`, and that is what every calculation
must consume — `RangeAssumption.for_equity`, never `.range`. Flattening a
maniac's flop betting range put its bluffs at 24.1% of the range instead of
16.0% and moved hero's equity 2.4 points, twice the margin of error the figure
was quoted with. Three of the five archetypes bluff; a station does not, which
is exactly why the bug could ship unnoticed.

An emptied posterior is real, not a bug: the opponent can hold a hand outside
the prior, and the grid can disagree with the sample the policy decided from.
Fall back to the last good range and say conditioning stopped — an empty range
has no equity, and reverting quietly hides the disagreement.

`equity_grid()` is what makes this affordable: one villain hand and one runout
per iteration, shared across every hero combo. Around 1s for a 400-combo grid
against 5.3s for the naive loop.

## Practice mode

`practice/` orchestrates everything else. Two rules it must keep:

- **A human is just another `Player`.** `PracticeSession` drives
  `players.table.play_hand` with a `CallbackPlayer`; it does not re-implement a
  betting round. Anything else reintroduces the two-copies-drift bug.
- **The range is derived from the policy, or it is configuration — never a
  guess.** `players/conditioning.py` narrows the opponent's range by asking its
  own policy which holdings take the observed action. That is legitimate only
  because the opponent *is* an explicit policy; a posterior derived this way is
  a fact about that policy, not about a person, and the provenance line says so
  every time. `PracticeConfig.range_for()` still returns the fixed configured
  range when `condition_on_action=False`.

Feedback is three-way split: `verified` (deterministic calculations, no model),
`interpretation` (the model's read), `unresolved` (the OPEN facts). Only the
middle one needs a model, which is what makes the fallback honest. Never merge
them into one block — a student reads undifferentiated prose with uniform
confidence, which is the failure this project guards against on the model side.

**Deterministic is not exact, and the boundary keeps moving up.** The `verified`
heading depends on *two* flags, not one:

- `analysis.equity.exact` — sampled equity makes every figure derived from it an
  estimate, including a terminal call's EV. The tree being terminal says nothing
  about the precision of the equity feeding it.
- `range_assumption.sampled` — a *conditioned* range was itself narrowed by
  Monte Carlo, so a river enumeration against it is exact arithmetic on an
  uncertain input. `equity.exact` is `True` there and the figure is still not
  deterministic.

The pattern generalises: each time a layer consumes something inexact, the
confidence label has to be recomputed, never inherited. Never label a sampled
figure "exact" — that is the confidence-boundary error the fact categories exist
to prevent, and it is worse here because this layer faces the student.

**Conditioning and provenance are separate claims.** `RangeConditioning` says
*what the range is conditioned on*; `RangeAssumption.provenance` says *who
decided it*. A read the caller typed in and a posterior computed from a known
policy are both `action-conditioned` and deserve very different trust. The
`ACTION_CONDITIONED` caveat used to assert "an assumed read, not derived",
which made the two indistinguishable and became false the moment derivation
existed.

`AnalysisFact.render()` must keep emitting `provenance`. It carried the field
and dropped it on render for several releases, so "exact enumeration" and
"Monte Carlo, 6,000 samples" reached the model as the same string.

Grounding failure does **not** block. One repair request quoting the failed
claims, then withhold the prose and show the arithmetic with a warning. The
checker matches magnitudes, not meanings, so it is noisy in both directions and
must not be load-bearing.

## Not yet built

`knowledge/` and `solver/` are documented stubs. Each docstring states the
intended shape and why it was deferred. Don't fill one in speculatively — they
were left empty because the design depends on how the coaching prompt actually
performs.

## Known limits (stated, not bugs)

- **The grounding checker only sees numbers.** Confirmed in the first live run:
  a response passed with every figure traceable while also claiming a "nut
  flush draw with two overcards" — true, but derived by the model, not supplied
  in the facts. A wrong hand-read would pass identically. Closing this means
  putting draw/texture facts into `SpotAnalysis` so there is something to check
  against; do not attempt it by parsing prose.
- The grounding checker ignores bare integers 0–10, so an invented out-count
  ("you have 9 outs") is not caught.
- `evaluation.recommended_action` reads the first action word, so "don't fold —
  call" is misread. In scope only because the prompt requires the recommendation
  to lead.
- `analyse` costs roughly half a second at 20k iterations; the evaluator is pure
  Python. Fine for a coaching turn, not for a tight loop.
