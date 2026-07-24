"""Pot odds, expected value, and the standard bluff/defence identities.

These are the numbers a coach quotes constantly. They are deliberately plain
functions over floats — no game state — so they can be unit-tested against
hand-checked values and reused from prompts, tests and the CLI alike.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PotOdds",
    "pot_odds",
    "required_equity",
    "ev_call",
    "ev_bluff",
    "implied_odds_breakeven",
    "minimum_defence_frequency",
    "bluff_success_threshold",
    "break_even_fold_equity",
    "bet_as_pot_fraction",
    "stack_to_pot_ratio",
    "outs_to_equity",
]


@dataclass(frozen=True, slots=True)
class PotOdds:
    """The price being laid on a call."""

    pot_before_call: float
    to_call: float

    @property
    def required_equity(self) -> float:
        """Break-even equity as a fraction in ``[0, 1]``."""

        total = self.pot_before_call + self.to_call
        if total <= 0:
            return 0.0
        return self.to_call / total

    @property
    def required_equity_pct(self) -> float:
        return 100.0 * self.required_equity

    @property
    def ratio(self) -> float:
        """Odds being laid, as ``pot : call`` expressed as a single number."""

        if self.to_call <= 0:
            return float("inf")
        return self.pot_before_call / self.to_call

    def __str__(self) -> str:
        return (
            f"{self.ratio:.2f}:1 — need {self.required_equity_pct:.1f}% "
            f"to call {self.to_call:g} into {self.pot_before_call:g}"
        )


def pot_odds(pot_before_call: float, to_call: float) -> PotOdds:
    """Build a :class:`PotOdds` from the pot *before* the call and the price.

    ``pot_before_call`` must already include the opponent's bet.
    """

    if pot_before_call < 0 or to_call < 0:
        raise ValueError("pot and call amounts must be non-negative")
    return PotOdds(pot_before_call=pot_before_call, to_call=to_call)


def required_equity(pot_before_call: float, to_call: float) -> float:
    """Break-even calling equity as a fraction in ``[0, 1]``."""

    return pot_odds(pot_before_call, to_call).required_equity


def ev_call(pot_before_call: float, to_call: float, equity: float) -> float:
    """EV of calling, in chips, relative to folding.

    Winning takes ``pot_before_call``; losing costs ``to_call``.
    """

    _check_equity(equity)
    return equity * pot_before_call - (1.0 - equity) * to_call


def ev_bluff(pot: float, bet: float, fold_equity: float, equity_when_called: float = 0.0) -> float:
    """EV of betting as a bluff, in chips, relative to checking/folding.

    ``fold_equity`` is the chance the opponent folds. When called, the bluff
    still realises ``equity_when_called`` of the resulting pot — set it to 0 for
    a pure bluff with no outs.
    """

    _check_equity(fold_equity, "fold_equity")
    _check_equity(equity_when_called, "equity_when_called")

    called_pot = pot + 2 * bet
    ev_when_called = equity_when_called * called_pot - bet
    return fold_equity * pot + (1.0 - fold_equity) * ev_when_called


def minimum_defence_frequency(pot: float, bet: float) -> float:
    """MDF: the share of their range a player must continue with facing ``bet``."""

    if bet <= 0:
        return 1.0
    if pot < 0:
        raise ValueError("pot must be non-negative")
    return pot / (pot + bet)


def bluff_success_threshold(pot: float, bet: float) -> float:
    """Alpha: how often a bluff must work to break even. ``1 - MDF``."""

    return 1.0 - minimum_defence_frequency(pot, bet)


#: Alias — the same quantity, named the way it usually gets asked about.
break_even_fold_equity = bluff_success_threshold


def bet_as_pot_fraction(pot: float, bet: float) -> float:
    """``bet / pot``, e.g. ``0.75`` for a three-quarter-pot bet."""

    if pot <= 0:
        raise ValueError("pot must be positive")
    return bet / pot


def stack_to_pot_ratio(effective_stack: float, pot: float) -> float:
    """SPR — low values commit stacks, high values leave room to manoeuvre."""

    if pot <= 0:
        return float("inf")
    return effective_stack / pot


def implied_odds_breakeven(
    pot_before_call: float, to_call: float, equity: float
) -> float:
    """Extra chips hero must win later to make a losing call break even.

    Added after a live run in which the model, given no such figure, derived
    one itself. A quantity the coach predictably reaches for is safer supplied
    than left to be invented.

    Convention matches :func:`ev_call`: winning takes ``pot_before_call`` (the
    pot as it stands, including the bet faced) — hero's own call is *not*
    counted as winnings, because hero put it there. So the break-even X solves
    ``equity * (pot_before_call + X) == (1 - equity) * to_call``. Counting the
    call as part of the win inflates equity and understates X.

    Returns 0.0 when the call already breaks even on pot odds alone. The figure
    says nothing about whether villain would actually pay that much.
    """

    _check_equity(equity)
    if equity <= 0:
        return float("inf")  # no draw can ever be paid off enough
    if ev_call(pot_before_call, to_call, equity) >= 0:
        return 0.0

    return ((1.0 - equity) * to_call) / equity - pot_before_call


def outs_to_equity(outs: int, streets: int = 1, unseen: int = 47) -> float:
    """Chance of hitting at least one of ``outs`` over ``streets`` cards.

    Exact hypergeometric complement, not the "rule of 2 and 4" shortcut. The
    default ``unseen=47`` is the post-flop count with two hole cards and three
    board cards known.
    """

    if outs < 0:
        raise ValueError("outs must be non-negative")
    if streets < 0:
        raise ValueError("streets must be non-negative")
    if outs > unseen:
        raise ValueError(f"cannot have {outs} outs among {unseen} unseen cards")
    if streets == 0:
        return 0.0

    miss = 1.0
    for i in range(streets):
        remaining = unseen - i
        if remaining <= 0:
            break
        miss *= (remaining - outs) / remaining
    return 1.0 - miss


def _check_equity(value: float, name: str = "equity") -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a fraction in [0, 1], got {value!r}")
