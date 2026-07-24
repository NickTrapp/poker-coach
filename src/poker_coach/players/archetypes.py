"""Recognisable opponent types.

Each style exaggerates one leak, because the point is to give a student
something specific to exploit:

============  ===========================================================
``NIT``       Plays too few hands and folds anything marginal.
``TAG``       Tight-aggressive. The closest to reasonable; a control case.
``LAG``       Loose-aggressive. Wide range, bets and bluffs constantly.
``STATION``   Calls far too much and almost never raises or bluffs.
``MANIAC``    Bets and raises with nearly anything.
============  ===========================================================

The numbers are chosen to make each tendency legible, not to model any real
population. `STATION` carries a negative ``call_margin``, which is exactly what
"calls at prices that lose money" means in this framework.
"""

from __future__ import annotations

import random

from .base import PlayerStyle, RuleBasedPlayer

__all__ = ["NIT", "TAG", "LAG", "STATION", "MANIAC", "STYLES", "make_player"]


NIT = PlayerStyle(
    label="nit",
    open_range="88+, ATs+, KQs, AQo+",
    raise_threshold=0.85,
    bet_threshold=0.72,
    call_margin=0.12,  # demands a big edge before putting money in
    bluff_frequency=0.0,
    bet_fraction=0.5,
)

TAG = PlayerStyle(
    label="tag",
    open_range="22+, A2s+, K9s+, QTs+, JTs, ATo+, KQo",
    raise_threshold=0.72,
    bet_threshold=0.58,
    call_margin=0.02,
    bluff_frequency=0.15,
    bet_fraction=0.66,
)

LAG = PlayerStyle(
    label="lag",
    open_range="22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, 86s+, A7o+, K9o+, QTo+, JTo",
    raise_threshold=0.62,
    bet_threshold=0.48,
    call_margin=-0.02,
    bluff_frequency=0.35,
    bet_fraction=0.75,
)

STATION = PlayerStyle(
    label="station",
    open_range="22+, A2s+, K2s+, Q6s+, J7s+, T7s+, 96s+, A2o+, K8o+, Q9o+, JTo",
    raise_threshold=0.90,  # almost never raises
    bet_threshold=0.70,
    call_margin=-0.15,  # calls well past the point it stops being profitable
    bluff_frequency=0.0,
    bet_fraction=0.4,
)

MANIAC = PlayerStyle(
    label="maniac",
    open_range="random",
    raise_threshold=0.50,
    bet_threshold=0.35,
    # Looser than the station's -0.15: at -0.10 a maniac folded 72o to a raise,
    # which is not what the name promises. This is the widest margin here.
    call_margin=-0.20,
    bluff_frequency=0.60,
    bet_fraction=1.0,
)

#: Lookup by label, for building an opponent from a config string.
STYLES: dict[str, PlayerStyle] = {
    style.label: style
    for style in (NIT, TAG, LAG, STATION, MANIAC)
}


def make_player(
    name: str,
    style: str | PlayerStyle = TAG,
    *,
    rng: random.Random | None = None,
    iterations: int = 400,
) -> RuleBasedPlayer:
    """Build an opponent by style name or style object."""

    if isinstance(style, str):
        try:
            style = STYLES[style.lower()]
        except KeyError:
            known = ", ".join(sorted(STYLES))
            raise ValueError(
                f"unknown style {style!r}; choose one of: {known}"
            ) from None

    return RuleBasedPlayer(
        name=name,
        style=style,
        rng=rng or random.Random(),
        iterations=iterations,
    )
