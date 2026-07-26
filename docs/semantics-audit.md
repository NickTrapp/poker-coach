# Semantic audit of the facts contract

What every value handed to the model actually meant, as of the first live run,
and what it was implicitly claiming. Categories:

- **S** — exact state fact
- **D** — exact derived calculation
- **M** — sampled (Monte Carlo) calculation
- **A** — calculation conditioned on an explicit assumption
- **T** — simplified theoretical benchmark
- **J** — model-generated strategic judgment (not a supplied fact at all)

| Rendered as | True category | Problem |
|---|---|---|
| `Street`, `Hero`, `Board`, `Pot`, `To call` | S | none |
| `Hero's made hand` | D | none, but *only* the made hand — no draws |
| `Effective stack` | S | ambiguous name; it is the **start-of-street** stack (behind + already committed), not chips behind now |
| `SPR` | D | **wrong**: used hero's own stack, not the effective one; numerator is start-of-street, denominator includes the live bet |
| `Assumed villain range` | A | does not say whether it is pre-action or conditioned on the bet |
| `Hero equity vs that range` | M or D | correct, and provenance was already carried |
| `Pot odds` / `need X%` | D | none |
| `EV of calling` | **A**, presented as D | assumes free run-out to showdown and full equity realization; ignores future betting; compares call vs **fold only**, never call vs raise |
| `Minimum defence frequency` | T | reads as a prescription for hero |
| `Villain's bluff must work X%` | T | reads as a claim about villain's actual strategy |
| "nut flush draw with two overcards" | **J** | model-derived, unverifiable, absent from facts |

## The four defects

1. **`SPR` was numerically wrong.** `HandState.spr(name)` called
   `effective_stack(name)` with a single name, which returns that player's own
   stack. With hero 200 / villain 20 it reported 16.67 instead of 1.67. The
   live example hid it because both stacks were equal.

2. **`EV of calling` was mislabelled.** `equity * pot - (1 - equity) * to_call`
   is exact on a terminal river call. On the flop it silently assumes the hand
   is checked down with full equity realization. It was rendered identically in
   both cases.

3. **Range conditioning was unstated.** The same string could mean "villain's
   opening range" or "the range villain bets here", which are different
   distributions producing different equities.

4. **Only numbers were checkable.** Hand-reading and board-texture claims were
   entirely outside the facts, so the checker could not see them.

## Changes made

- `AnalysisFact` — a fact carries `category`, `assumptions`, `scope`,
  `provenance`. The prompt block groups by category.
- `HandState.spr` fixed to use the effective stack; `effective_stack` and
  `stack_behind` given unambiguous names and tests.
- `ev_call` split: `ev_call_terminal` (exact) vs
  `ev_call_if_realized` (assumption-carrying). Non-terminal streets label it
  and add the call-vs-raise caveat.
- `VillainRange` carries `conditioning`: `pre_action` or `action_conditioned`.
- `calculations/features.py` computes draws, texture, overcards, board pairing
  as deterministic facts.
- MDF and alpha relabelled as reference values with stated assumptions.
- Evaluator asserts "continuing beats folding", not "call is uniquely correct".

## First benchmark run (gemini-3.6-flash, prompt v2) — 8/9

The semantic fixes landed. Responses named the range conditioning, flagged
call-vs-raise as an open question in the exact words the facts use, and treated
sampled equity as approximate.

The single failure is the most instructive result in the run. On
`turn-draw-realization` the model needed an implied-odds figure, was not given
one, derived it, and cited it — "you would need to win over 100 additional
chips". The grounding checker flagged the bare `100` as ungrounded, which is
exactly right: the model computed rather than quoted.

**The model's number was sound.** The break-even is 95.3 extra chips against a
stack of 80, so its conclusion — that the chips are not there — holds. An
earlier check of mine put the figure at 65.3 by counting hero's own call as
winnings; that is wrong, and it would make a hopeless draw look playable. The
convention now has its own regression test.

The fix is therefore not "stop the model reasoning" but "supply the fact it
predictably reaches for": `implied_odds_breakeven` is now computed on
non-terminal streets and rendered with its assumptions.

## External review (post-push) — all six findings reproduced

An outside review of the pushed repository raised six defects. Each was
reproduced before being fixed; none had been caught by the 872-test suite,
because every one produced a plausible number rather than an error.

| # | Finding | Status |
|---|---|---|
| 1 | `apply()` enforces no turn order, street match, min-raise, or round completion | fixed |
| 2 | `analyze()` applies one range to a multiway pot | fixed — refuses |
| 3 | Multiway range sampling is sequential, not uniform-joint | fixed |
| 4 | One all-in opponent marks a call terminal | fixed |
| 5 | MDF/alpha derive from hero's call, wrong against raises | fixed |
| 6 | Grounding matches magnitudes, not meanings | **confirmed, documented** |

Measured before/after:

- **(3)** With `opp1 ∈ {AsKs, 2c3c}`, `opp2 ∈ {AsQs, AsJs, 2c4c}` — three legal
  joint assignments — the old sampler produced 0.496 / 0.252 / 0.252 where
  uniform is 1/3 each. It also made equity depend on the order ranges were
  passed. Now 0.334 / 0.330 / 0.336, order-invariant within the margin of error.
- **(5)** Pot 10, hero bets 5, villain raises to 15: reported alpha 33.3%, true
  50%. The aggressor's wager is their whole street commitment, not hero's call.
- **(6)** With equity 71.80% and MDF 50%, "Hero has 50% equity" passed.

### Betting legality — `domain/betting.py`

`BettingRound` is derived from `HandState` on demand, so it cannot fall out of
sync, and `state.legal_actions(name)` is the single source of truth. Two tiers,
because they are not equally knowable:

- **Always enforced** (derivable from state alone): the street tag on an action,
  the minimum raise, and that a street cannot be left with a bet unmatched.
- **`strict=True` only** (needs the street's action history): turn order and
  raise-reopening. Replay and the table runner pass it. A directly-built state
  has no history and no way to acquire one; without it the minimum-raise floor
  also degrades to the big blind, which is permissive rather than wrong.

Enabling it immediately found **two of the three bundled example hands were
illegal** — heads-up the big blind acts first from the flop on, and both had the
small blind leading. Same money, wrong order; they replayed silently for months.
Regenerated, with identical pots.

A 1200-hand run across 2–6 seats with mixed stacks then surfaced two more:

- Float underflow set a stack to `-7.1e-15`, which `validate_assignment`
  rejected. Clamped before assignment rather than after.
- The opponent policy proposed re-raises when an under-raise all-in had not
  reopened the betting. It now asks `legal_actions()` instead of assuming —
  which is the reviewer's point applied one layer up.

After both: 1200 hands, zero failures, every emitted history replaying strictly.

### Still open

**Grounding semantics (6).** Needs fact ids and units, model citations like
`[hero_equity_pct]`, and validation of the citation rather than the digits —
a change to how responses are produced, not a patch to the checker.


## Second review round — five more, all reproduced

| # | Finding | Status |
|---|---|---|
| 1 | `legal_actions()` was not actually the source of truth | fixed |
| 2 | Round completion never enforced (only unmatched bets) | fixed |
| 3 | MDF used the aggressor's total commitment, not their increment | fixed |
| 4 | Betting allowed into a pot no opponent could contest | fixed |
| 5 | `analyze()` accepted a folded hero and a finished hand | fixed |

**(1)** `validate()` returned early for every non-raise, and again for any
all-in raise — so a shove was accepted after an under-raise that never reopened
the betting, an action `legal_actions()` had already excluded. It now validates
against the listed options instead of a parallel set of hand-written checks.
That also closed sub-minimum bets, `RAISE` into an unopened pot, and
`POST_BLIND` mid-street. Writing it this way surfaced a bug of the opposite
kind: folding was not offered when checking was free, though folding is legal
whenever it is your turn.

**(2)** `unmatched()` is empty on an unbet street before anyone acts, so a flop
could advance to the turn with no action, or after one player of two had
checked. `advance_street` now requires `is_complete()` whenever the hand
carries any recorded action — `state.actions` spans the whole hand, so a fresh
street mid-hand is still covered; only a wholly synthetic state escapes.

**(3)** The wager is what the aggressor *added*, not their street total. A small
blind raising to 3 has posted 0.5, so it risked 2.5 to win 1.5: alpha 62.5%,
where the full commitment gave 75%. `BettingRound` now replays the street's
commitments to recover the increment and the pot before it. Where no aggression
is recorded and the blinds make inference unsound, MDF and alpha are withheld
rather than guessed.

**(4)** Aggression now requires an opponent who can still act, and a street with
fewer than two such players runs out without recording a check.

Enabling (2) then exposed an inconsistency between the runner and the rules: a
street with one player able to act was skipped by the runner but reported
incomplete by `is_complete`, failing 46 of 1500 hands. A round needs two players
able to act to be a betting round at all. After the fix: **1500 hands, 2-6
seats, mixed stacks, zero failures.**


## Third review round — two more, both reproduced

**Multiway action order broke after a raise.** `next_actor` scanned from the top
of the street order and returned the first player owing chips. Three-handed
(BTN, SB, BB): BTN calls, SB raises, and BTN owes again — so the scan returned
BTN when the action must pass to BB. Heads-up hid it entirely, since there is
only one candidate after the aggressor.

Turn selection now starts clockwise from whoever acted last. Rotation does not
change *whether* a candidate exists, only *which* one, so round completion was
unaffected — it was purely an identity bug.

The runner kept its own `acted` set and rescanned from the top after every
action, i.e. **the identical mistake**. That is why 1500 replayed hands stayed
green: the generator and the checker agreed with each other. `_play_street` now
calls `next_actor` — the same function `apply(strict=True)` validates against.
Two copies of a rule drift; one cannot.

The lesson generalises past this bug. A replay suite proves self-consistency,
not correctness. `tests/test_review_findings.py` now carries an **order oracle
written from the rules rather than derived from the implementation**, checked
across six seatings and all four streets.

**Folded players' chips vanished from the reconstructed pot.** The commitment
replay did `running[actor] = action.amount` for every action, but folds and
checks carry amount 0 — so folding after calling 3 erased that 3. BTN raises 3,
SB calls, BB raises 10, BTN folds, SB raises 30: the pot in front of SB is 16,
reconstructed as 13, giving alpha 67.5% instead of 62.79%. Only money-in actions
update the running commitment now, and `committed_when_acted` derives from it —
which also stops a big blind's check zeroing its posted chip.
