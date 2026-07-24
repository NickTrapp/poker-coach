"""Simulated opponents for practice hands.

Rule-based archetypes, not solver output — each exaggerates one recognisable
leak so a student has something specific to exploit. See `base.py` for the
policy and `archetypes.py` for the styles.

`table.play_hand` deals and plays a hand out between two of them, returning the
result plus a `HandHistory` that replays to the same final state. It is
heads-up only — multiway action order needs a seat ring and a rotating button,
and half-doing it would produce hands that look right and are subtly illegal.
"""

from .archetypes import LAG, MANIAC, NIT, STATION, STYLES, TAG, make_player
from .base import Player, PlayerStyle, RuleBasedPlayer
from .table import HandResult, Seat, play_hand

__all__ = [
    "HandResult",
    "LAG",
    "MANIAC",
    "NIT",
    "Player",
    "PlayerStyle",
    "RuleBasedPlayer",
    "STATION",
    "STYLES",
    "Seat",
    "TAG",
    "make_player",
    "play_hand",
]
