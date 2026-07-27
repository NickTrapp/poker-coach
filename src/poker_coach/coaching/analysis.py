"""Turn a `HandState` into the factual block a coach reasons from.

This is the bridge between the deterministic layer and the LLM layer. Every
quantity is computed by :mod:`poker_coach.calculations` and handed to the model
as a given fact — but *what kind* of fact each one is now travels with it (see
:mod:`poker_coach.coaching.facts`), because the first live run showed a model
treating an assumption-laden estimate as though it were exact.

Three semantics worth stating up front, all of them previously implicit:

**Call EV is a call-versus-fold comparison, never call-versus-raise.** Nothing
computed here evaluates raising. A positive number means continuing beats
folding; it says nothing about whether calling is the *best* continuation.

**Call EV is exact only on a terminal call.** On the river, or when the call
puts someone all-in, ``equity * pot - (1 - equity) * to_call`` is the real
number. Anywhere else it silently assumes the hand checks down and hero
realises every point of equity — no future bets faced, no folds, no
implied odds. Those spots are labelled and the assumption is printed.

**A range must say what it is conditioned on.** "Villain's opening range" and
"the range villain bets this flop with" are different distributions that
produce different equities. The caller states which one it supplied; nothing
here silently converts between them, because that conversion needs a strategy
model this package does not have.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from ..calculations.equity import EquityResult, equity
from ..calculations.features import HandFeatures, analyse_features
from ..calculations.hand_eval import HandRank, evaluate
from ..calculations.pot_odds import (
    PotOdds,
    bluff_success_threshold,
    ev_call,
    implied_odds_breakeven,
    minimum_defence_frequency,
    pot_odds,
)
from ..calculations.ranges import Range, WeightedRange
from ..domain.cards import cards_to_str
from ..domain.enums import Street
from ..domain.state import HandState
from .facts import AnalysisFact, FactCategory, render_fact_groups

__all__ = [
    "SpotAnalysis",
    "RangeAssumption",
    "RangeConditioning",
    "analyze",
    "DEFAULT_VILLAIN_RANGE",
]

DEFAULT_VILLAIN_RANGE = "random"


class RangeConditioning(str, Enum):
    """What the supplied range is conditioned on."""

    PRE_ACTION = "pre-action"
    """The range villain held *before* the action now being faced. Using it to
    price a bet overstates hero's equity whenever villain bets a subset."""

    ACTION_CONDITIONED = "action-conditioned"
    """The range villain is credited with *given* the observed action. Says
    nothing about where it came from — see :attr:`RangeAssumption.provenance`,
    which is a separate question and carries most of the trust."""

    UNSPECIFIED = "unspecified"
    """The caller did not say. Treated as pre-action, and flagged as such."""

    @property
    def caveat(self) -> str:
        return _CONDITIONING_CAVEATS[self]


_CONDITIONING_CAVEATS: dict[RangeConditioning, str] = {
    RangeConditioning.PRE_ACTION: (
        "this is villain's range BEFORE the action faced, not the range they "
        "take this action with"
    ),
    RangeConditioning.ACTION_CONDITIONED: (
        "villain's range GIVEN the observed action, not the range they held "
        "before it"
    ),
    RangeConditioning.UNSPECIFIED: (
        "conditioning not stated by the caller; treat as a pre-action range"
    ),
}


@dataclass(frozen=True, slots=True)
class RangeAssumption:
    """A villain range, what it is conditioned on, and where it came from."""

    notation: str
    conditioning: RangeConditioning = RangeConditioning.UNSPECIFIED
    #: A readable stand-in for :attr:`notation`, shown wherever the range is
    #: displayed. A posterior is an explicit list of hundreds of combos: exactly
    #: what the equity engine needs and exactly what nobody can read. Every
    #: calculation still uses ``notation``; only the prose uses this.
    #:
    #: It must contain no digits that are not part of a card or range token —
    #: it lands in prompt prose, which the grounding checker scans for numbers
    #: the model was never given.
    label: str | None = None
    #: *Who decided this range* — a different question from what it is
    #: conditioned on, and the one that carries most of the trust. A read the
    #: caller typed in and a posterior derived from a known policy can both be
    #: action-conditioned while deserving very different confidence.
    provenance: str = "supplied by the caller, not derived"
    #: How many combos survived conditioning, and how many were possible before
    #: it. Both or neither. Stated as a fact rather than left implicit in the
    #: notation, because "villain got narrower" is the whole point of
    #: conditioning and a student cannot see it from a range string.
    combos: int | None = None
    combos_before: int | None = None
    #: The opponent actions this range is conditioned on, in words
    #: ("check preflop, then bet on the flop"). Kept apart from :attr:`label` so
    #: a display can show the line without the surrounding phrasing. Countless,
    #: for the same reason as the label.
    observed_line: str | None = None
    #: The posterior with its probabilities intact. Kept apart from
    #: :attr:`notation`, which is only the *support*: a range string says which
    #: combos are in, never that one is four times likelier than another. Every
    #: equity figure measures against this when it is present.
    weighted: WeightedRange | None = None
    #: True when the range itself was derived by simulation, so it carries error
    #: of its own. Independent of whether the *equity* was enumerated: a river
    #: enumeration against a sampled range is exact arithmetic on an uncertain
    #: input, and calling the result deterministic is the same
    #: confidence-boundary mistake as calling a terminal call's EV exact when
    #: the equity feeding it was sampled.
    sampled: bool = False

    @property
    def narrowing(self) -> str | None:
        if self.combos is None or self.combos_before is None:
            return None
        return f"{self.combos} of {self.combos_before}"

    @property
    def range(self) -> Range:
        """The support, every combo equally likely. For display and blockers."""

        return Range(self.notation)

    @property
    def for_equity(self) -> Range | WeightedRange:
        """What a calculation must measure against.

        The weighted posterior when there is one. Using :attr:`range` instead
        silently flattens it — for a maniac that moved hero's equity by 2.4
        points, twice the margin of error the figure was quoted with.
        """

        return self.weighted if self.weighted is not None else self.range

    @property
    def display(self) -> str:
        """What a human should be shown. Never a raw combo dump."""

        return self.label or self.notation

    def __str__(self) -> str:
        return self.display


@dataclass(frozen=True, slots=True)
class SpotAnalysis:
    """Every deterministic fact about the decision hero faces, with semantics."""

    street: str
    hero_name: str
    hero_position: str
    hero_cards: str
    board: str
    made_hand: HandRank | None
    features: HandFeatures
    total_pot: float
    to_call: float
    effective_stack: float
    hero_stack_behind: float
    spr: float
    range_assumption: RangeAssumption
    equity: EquityResult
    odds: PotOdds
    ev_call_vs_fold: float
    ev_is_terminal: bool
    implied_odds_needed: float | None
    mdf: float | None
    alpha: float | None
    facts: list[AnalysisFact] = field(default_factory=list)

    # --------------------------------------------------------------- accessors

    @property
    def villain_range(self) -> str:
        """The range notation. See :attr:`range_assumption` for conditioning."""

        return self.range_assumption.notation

    @property
    def ev_of_calling(self) -> float:
        """Legacy alias for :attr:`ev_call_vs_fold`.

        Kept because the name appears throughout older call sites, but prefer
        the explicit one: this is a call-versus-*fold* comparison only.
        """

        return self.ev_call_vs_fold

    @property
    def facing_bet(self) -> bool:
        return self.to_call > 0

    @property
    def continuing_beats_folding(self) -> bool | None:
        """Whether calling is better than folding. None if nothing to call.

        Deliberately *not* named ``call_is_profitable``: a positive value does
        not establish that calling is the best action, only that folding is
        worse than continuing.
        """

        if not self.facing_bet:
            return None
        return self.ev_call_vs_fold > 0

    #: Retained alias; the name overstates what the number shows.
    call_is_profitable = continuing_beats_folding

    @property
    def pot_after_call(self) -> float:
        """The pot once hero calls — the denominator of the pot-odds figure.

        Supplied because the model predictably reaches for it: a live run cited
        "a total pot of 35" from a pot of 20 and a call of 15, and the checker
        flagged it as invented. It was arithmetic the facts implied but never
        stated. Widening the checker to accept derived combinations instead
        would have no principled stopping point — pot after a raise, final
        stack, percentage differences — and would quietly retire the rule that
        the model quotes rather than computes.
        """

        return self.total_pot + self.to_call

    @property
    def equity_surplus(self) -> float:
        """How much equity hero has above (or below) the price."""

        return self.equity.equity - self.odds.required_equity

    def to_prompt_block(self) -> str:
        """Render the facts, grouped by how certain each one is."""

        return render_fact_groups(self.facts)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.to_prompt_block()


def _is_terminal_call(state: HandState, hero: str, to_call: float) -> bool:
    """True when calling ends the betting, making the EV formula exact.

    Either the river (no street follows) or the call leaves someone with no
    chips behind, so no further betting is possible.
    """

    if state.street is Street.RIVER:
        return True
    if to_call <= 0:
        return False

    hero_left = state.stack_behind(hero) - to_call
    others_left = [
        state.stack_behind(p.name)
        for p in state.active_players
        if p.name != hero
    ]
    # `all`, not `any`: one opponent being all-in does not end the betting if
    # another still has chips behind. Using `min(...) <= 0` marked a call
    # terminal — and so promoted its EV to "exact" — whenever *any* opponent
    # was all-in, even with live money still to act.
    return hero_left <= 1e-9 or (
        bool(others_left) and all(stack <= 1e-9 for stack in others_left)
    )


def _build_facts(analysis_kwargs: dict) -> list[AnalysisFact]:
    """Assemble the fact list, each carrying its own semantics."""

    a = analysis_kwargs
    features: HandFeatures = a["features"]
    equity_result: EquityResult = a["equity"]
    odds: PotOdds = a["odds"]
    assumption: RangeAssumption = a["range_assumption"]
    facing_bet = a["to_call"] > 0

    facts: list[AnalysisFact] = [
        AnalysisFact("Street", a["street"], FactCategory.STATE),
        AnalysisFact(
            "Hero",
            f"{a['hero_name']} ({a['hero_position']}) with {a['hero_cards']}",
            FactCategory.STATE,
        ),
        AnalysisFact("Board", a["board"] or "(none)", FactCategory.STATE),
        AnalysisFact("Pot", f"{a['total_pot']:g}", FactCategory.STATE,
                     scope="includes the bet hero is facing"),
        AnalysisFact("To call", f"{a['to_call']:g}", FactCategory.STATE),
        AnalysisFact(
            "Hero stack behind",
            f"{a['hero_stack_behind']:g}",
            FactCategory.STATE,
            scope="chips hero has left right now, after villain's bet",
        ),
        AnalysisFact(
            "Effective stack (start of street)",
            f"{a['effective_stack']:g}",
            FactCategory.STATE,
            scope="smallest stack at the start of this street; the most either "
                  "player can lose",
        ),
    ]

    # --- deterministic hand and board features -----------------------------
    if features.made_description:
        facts.append(
            AnalysisFact("Hero's made hand", features.made_description,
                         FactCategory.DERIVED)
        )
    if features.flush_draw:
        facts.append(
            AnalysisFact("Flush draw", features.flush_draw.description,
                         FactCategory.DERIVED)
        )
    if features.straight_draw:
        facts.append(
            AnalysisFact("Straight draw", features.straight_draw.description,
                         FactCategory.DERIVED)
        )
    if features.overcards:
        names = "".join(r.symbol for r in features.overcards)
        facts.append(
            AnalysisFact("Overcards to the board", f"{len(features.overcards)} ({names})",
                         FactCategory.DERIVED)
        )
    if features.texture.high_card is not None:
        facts.append(
            AnalysisFact("Board texture", features.texture.description,
                         FactCategory.DERIVED)
        )
    if features.blocks_nut_flush:
        facts.append(
            AnalysisFact("Card removal", "hero holds an ace of a board suit "
                         "(blocks the nut flush)", FactCategory.DERIVED)
        )

    spr = a["spr"]
    facts.append(
        AnalysisFact(
            "SPR",
            f"{spr:.2f}" if spr != float("inf") else "n/a",
            FactCategory.DERIVED,
            scope="effective start-of-street stack divided by the current pot "
                  "(the pot already includes the bet faced)",
        )
    )

    if facing_bet:
        facts.append(
            AnalysisFact(
                "Pot after hero calls",
                f"{a['total_pot'] + a['to_call']:g}",
                FactCategory.DERIVED,
                scope="what hero is playing for; the denominator of the pot-odds "
                      "figure below",
            )
        )
        facts.append(
            AnalysisFact(
                "Pot odds",
                f"{odds.ratio:.2f}:1 (need {odds.required_equity_pct:.1f}% "
                "equity to break even on a call)",
                FactCategory.DERIVED,
            )
        )

    # --- equity, with its provenance and its range assumption ---------------
    equity_category = (
        FactCategory.SAMPLED if not equity_result.exact else FactCategory.ASSUMED
    )
    facts.append(
        AnalysisFact(
            "Hero equity at showdown",
            str(equity_result),
            equity_category,
            assumptions=(
                f"villain holds exactly: {assumption.display}",
                assumption.conditioning.caveat,
                assumption.provenance,
                "all remaining cards are dealt with no further betting",
            ),
            provenance="exact enumeration" if equity_result.exact
            else f"Monte Carlo, {equity_result.samples:,} samples",
            scope="share of the pot at showdown; NOT the share hero actually "
                  "realises once future betting is accounted for",
        )
    )

    facts.append(
        AnalysisFact(
            "Assumed villain range",
            f"{assumption.display} ({assumption.conditioning.value})",
            FactCategory.ASSUMED,
            assumptions=(assumption.conditioning.caveat,),
            provenance=assumption.provenance,
        )
    )

    if assumption.narrowing is not None:
        facts.append(
            AnalysisFact(
                "Villain combos consistent with this line",
                assumption.narrowing,
                FactCategory.ASSUMED,
                assumptions=(assumption.provenance,),
                scope="how much of the starting range the observed action "
                      "rules out; every equity figure above is against what "
                      "is left",
            )
        )

    # --- the EV figure, labelled by whether it is actually exact ------------
    if facing_bet:
        if a["ev_is_terminal"]:
            facts.append(
                AnalysisFact(
                    "EV of calling vs folding",
                    f"{a['ev_call_vs_fold']:+.2f} chips",
                    FactCategory.ASSUMED,
                    assumptions=(f"villain holds exactly: {assumption.display}",),
                    provenance="exact for a terminal call — no betting follows",
                    scope="compares CALLING with FOLDING only; it does not "
                          "evaluate raising",
                )
            )
        else:
            facts.append(
                AnalysisFact(
                    "EV of calling vs folding, if equity were fully realised",
                    f"{a['ev_call_vs_fold']:+.2f} chips",
                    FactCategory.ASSUMED,
                    assumptions=(
                        f"villain holds exactly: {assumption.display}",
                        "the hand is checked down from here — NO further betting",
                        # Deliberately no bare percentage here: the facts block
                        # must ground itself, and a stray "100%" reads as a
                        # numeric claim with no matching value.
                        "hero realises all of its showdown equity, never "
                        "folding a better hand and never paying another bet",
                    ),
                    provenance="upper bound; the real figure is lower whenever "
                               "hero can be bet off the hand",
                    scope="compares CALLING with FOLDING only; it does not "
                          "evaluate raising, and it is NOT decision-tree EV",
                )
            )

        # A losing call may still be right on implied odds. Supply the
        # break-even figure rather than leave the model to derive it — in the
        # first benchmark run one did, and got it badly wrong.
        needed = a["implied_odds_needed"]
        if needed is not None and needed > 0:
            facts.append(
                AnalysisFact(
                    "Extra chips needed on later streets to break even",
                    f"{needed:.1f}" if needed != float("inf") else "unreachable",
                    FactCategory.ASSUMED,
                    assumptions=(
                        "hero wins the current pot plus this much more whenever "
                        "the hand improves, and loses the call otherwise",
                    ),
                    scope="whether villain would actually pay this is unknown; "
                          "compare it against hero's stack behind, and note "
                          "that having the chips does not mean winning them",
                )
            )

        # --- reference values, explicitly not prescriptions -----------------
        if a["mdf"] is not None:
            facts.append(
                AnalysisFact(
                    "Minimum defence frequency for this bet size",
                    f"{100 * a['mdf']:.1f}%",
                    FactCategory.REFERENCE,
                    assumptions=(
                        "the figure at which a zero-equity bluff of this size "
                        "becomes break-even",
                    ),
                    scope="a property of the BET SIZE. It does not say hero "
                          "must defend this often with this hand, and hero's "
                          "own equity is a better guide than MDF here",
                )
            )
        if a["alpha"] is not None:
            facts.append(
                AnalysisFact(
                    "Break-even bluff frequency for this bet size",
                    f"{100 * a['alpha']:.1f}%",
                    FactCategory.REFERENCE,
                    assumptions=(
                        "a bluff with exactly zero equity when called",
                    ),
                    scope="how often such a bluff must win immediately to "
                          "break even. It is NOT a claim that villain bluffs "
                          "this often — villain's real frequency is unknown",
                )
            )
    else:
        facts.append(
            AnalysisFact(
                "Decision",
                "hero is not facing a bet (check or bet decision)",
                FactCategory.STATE,
            )
        )

    # --- what the numbers above do not settle ------------------------------
    open_questions = []
    if facing_bet:
        open_questions.append(
            AnalysisFact(
                "Calling vs raising",
                "not evaluated — no figure above compares them",
                FactCategory.OPEN,
                scope="a positive call EV shows only that continuing beats "
                      "folding",
            )
        )
    if facing_bet and not a["ev_is_terminal"]:
        open_questions.append(
            AnalysisFact(
                "Equity realisation",
                "not modelled — future streets may force hero off the hand",
                FactCategory.OPEN,
            )
        )
        open_questions.append(
            AnalysisFact(
                "Implied and reverse implied odds",
                "not modelled — chips won or lost on later streets are ignored",
                FactCategory.OPEN,
            )
        )
    if assumption.conditioning is not RangeConditioning.ACTION_CONDITIONED:
        open_questions.append(
            AnalysisFact(
                "Villain's actual betting range",
                "unknown — the supplied range is not conditioned on this action",
                FactCategory.OPEN,
            )
        )
    facts.extend(open_questions)

    return facts


def analyze(
    state: HandState,
    *,
    hero: str | None = None,
    villain_range: str | Range | RangeAssumption = DEFAULT_VILLAIN_RANGE,
    range_conditioning: RangeConditioning | str = RangeConditioning.UNSPECIFIED,
    iterations: int = 10_000,
    rng: random.Random | None = None,
) -> SpotAnalysis:
    """Compute the full factual picture of the hero's current decision.

    ``hero`` defaults to the player flagged ``is_hero``. ``villain_range`` is
    the assumption every equity figure is conditioned on; state what it is
    conditioned on via ``range_conditioning`` so the coach can say plainly
    whether it is a pre-action range or a read on this specific action.
    """

    player = state.player(hero) if hero else state.hero
    if player is None:
        raise ValueError(
            "no hero in this hand: pass hero=<name> or set is_hero on a player"
        )
    if player.hole_cards is None:
        raise ValueError(f"{player.name} has no hole cards to analyse")

    if player.has_folded:
        raise ValueError(
            f"{player.name} has folded; there is no decision left to analyse"
        )

    active = len(state.active_players)
    if active < 2:
        raise ValueError(
            f"the hand is over: {active} player(s) still active, so there is "
            "no opponent to hold a range"
        )
    if active > 2:
        # Refusing beats answering wrongly. One `villain_range` against a
        # three-way pot produces a heads-up equity figure sitting beside a
        # multiway pot and a table-wide effective stack — each number correct
        # in isolation, together describing a spot that does not exist.
        # Multiway needs one range per opponent, which is a real modelling
        # decision (position and action order imply different ranges), not a
        # default this function may invent.
        raise NotImplementedError(
            f"multiway analysis is not supported: {active} players are still "
            "active, and a single villain_range cannot describe them. "
            "Analyse a heads-up spot, or supply one range per opponent once "
            "that API exists."
        )

    rng_obj = rng or random.Random()

    if isinstance(villain_range, RangeAssumption):
        assumption = villain_range
    else:
        notation = (
            villain_range.notation
            if isinstance(villain_range, Range)
            else str(villain_range)
        )
        assumption = RangeAssumption(
            notation=notation, conditioning=RangeConditioning(range_conditioning)
        )

    to_call = state.amount_to_call(player.name)
    total_pot = state.total_pot

    eq = equity(
        player.hole_cards,
        assumption.for_equity,
        state.board,
        iterations=iterations,
        rng=rng_obj,
    )

    odds = pot_odds(total_pot, to_call)
    made_hand = (
        evaluate([*player.hole_cards, *state.board]) if len(state.board) >= 3 else None
    )
    features = analyse_features(player.hole_cards, state.board)

    mdf = alpha = implied_needed = None
    if to_call > 0:
        # MDF and alpha are defined by what the AGGRESSOR *added* and the pot
        # they were trying to win — not by hero's call, and not by their whole
        # street commitment either.
        #
        # A small blind raising to 3 has already posted 0.5, so it risked 2.5
        # to win the 1.5 in front of it: alpha is 62.5%, not the 75% that the
        # full commitment implies. Blinds, three-bets and postflop re-raises
        # all separate the two.
        #
        # When the increment cannot be recovered — a state with no recorded
        # aggression, where the blinds make any inference unsound — nothing is
        # reported. A plausible-but-possibly-false frequency is worse than a
        # missing one.
        betting = state.betting_round()
        if betting.last_wager is not None and betting.pot_before_aggression is not None:
            mdf = minimum_defence_frequency(
                betting.pot_before_aggression, betting.last_wager
            )
            alpha = bluff_success_threshold(
                betting.pot_before_aggression, betting.last_wager
            )

        # Only meaningful while chips can still change hands.
        terminal = _is_terminal_call(state, player.name, to_call)
        if not terminal:
            implied_needed = implied_odds_breakeven(total_pot, to_call, eq.equity)

    payload = dict(
        street=state.street.value,
        hero_name=player.name,
        hero_position=player.position.value,
        hero_cards=cards_to_str(player.hole_cards),
        board=cards_to_str(state.board),
        made_hand=made_hand,
        features=features,
        total_pot=total_pot,
        to_call=to_call,
        effective_stack=state.effective_stack(),
        hero_stack_behind=state.stack_behind(player.name),
        spr=state.spr(),
        range_assumption=assumption,
        equity=eq,
        odds=odds,
        ev_call_vs_fold=ev_call(total_pot, to_call, eq.equity),
        ev_is_terminal=_is_terminal_call(state, player.name, to_call),
        implied_odds_needed=implied_needed,
        mdf=mdf,
        alpha=alpha,
    )

    return SpotAnalysis(**payload, facts=_build_facts(payload))
