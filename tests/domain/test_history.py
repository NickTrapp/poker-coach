from pathlib import Path

import pytest

from poker_coach.domain import (
    Action,
    ActionType,
    HandHistory,
    Position,
    SeatRecord,
    Street,
    StreetRecord,
    parse_cards,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "hands"


def A(actor, type_, amount=0.0):
    return Action(actor=actor, type=type_, amount=amount)


def simple_history(**overrides) -> HandHistory:
    defaults = dict(
        hero="hero",
        small_blind=0.5,
        big_blind=1.0,
        seats=[
            SeatRecord(
                name="hero",
                position=Position.BTN,
                starting_stack=100.0,
                hole_cards=tuple(parse_cards("AsKs")),
            ),
            SeatRecord(
                name="villain",
                position=Position.BB,
                starting_stack=100.0,
                hole_cards=tuple(parse_cards("QhQd")),
            ),
        ],
        streets=[
            StreetRecord(
                street=Street.PREFLOP,
                actions=[
                    A("hero", ActionType.RAISE, 3.0),
                    A("villain", ActionType.CALL, 3.0),
                ],
            ),
            StreetRecord(
                street=Street.FLOP,
                cards=parse_cards("Qs2s9c"),
                actions=[
                    A("villain", ActionType.CHECK),
                    A("hero", ActionType.BET, 4.0),
                    A("villain", ActionType.CALL, 4.0),
                ],
            ),
        ],
    )
    defaults.update(overrides)
    return HandHistory(**defaults)


# ------------------------------------------------------------------ blinds


def test_blinds_are_posted_automatically():
    state = simple_history().initial_state()
    assert state.player("villain").committed_this_street == 1.0
    assert state.player("villain").stack == 99.0
    # The button posts nothing heads-up in this seating.
    assert state.player("hero").committed_this_street == 0.0
    assert state.total_pot == 1.0


def test_small_blind_is_posted_from_position():
    history = simple_history(
        seats=[
            SeatRecord(name="hero", position=Position.SB, starting_stack=100.0,
                       hole_cards=tuple(parse_cards("AsKs"))),
            SeatRecord(name="villain", position=Position.BB, starting_stack=100.0),
        ],
        streets=[],
    )
    state = history.initial_state()
    assert state.player("hero").committed_this_street == 0.5
    assert state.total_pot == 1.5


def test_antes_are_dead_money_not_a_live_bet():
    history = simple_history(ante=0.25, streets=[])
    state = history.initial_state()
    # Antes sit in the pot and do not raise the amount to call.
    assert state.pot == 0.5
    assert state.total_pot == 1.5
    assert state.amount_to_call("hero") == 1.0
    assert state.player("hero").committed_total == 0.25
    assert state.player("villain").committed_total == 1.25


def test_hero_flag_is_set_from_the_history():
    state = simple_history().initial_state()
    assert state.hero is not None
    assert state.hero.name == "hero"


def test_short_stack_cannot_post_more_than_it_has():
    history = simple_history(
        seats=[
            SeatRecord(name="hero", position=Position.BTN, starting_stack=100.0),
            SeatRecord(name="villain", position=Position.BB, starting_stack=0.4),
        ],
        streets=[],
    )
    state = history.initial_state()
    assert state.player("villain").stack == 0.0
    assert state.player("villain").is_all_in is True
    assert state.player("villain").committed_this_street == 0.4


# ------------------------------------------------------------------ replay


def test_replay_yields_every_action_in_order():
    points = list(simple_history().replay())
    assert [p.action.type for p in points] == [
        ActionType.RAISE,
        ActionType.CALL,
        ActionType.CHECK,
        ActionType.BET,
        ActionType.CALL,
    ]


def test_replay_state_precedes_its_action():
    points = list(simple_history().replay())
    # Before the hero's opening raise, nothing has gone in but the big blind.
    assert points[0].state.total_pot == 1.0
    # Before the villain's call, the raise is already live.
    assert points[1].state.total_pot == 4.0


def test_replay_stamps_the_street_onto_each_action():
    points = list(simple_history().replay())
    assert [p.action.street for p in points] == [
        Street.PREFLOP,
        Street.PREFLOP,
        Street.FLOP,
        Street.FLOP,
        Street.FLOP,
    ]
    assert [p.street for p in points][-1] is Street.FLOP


def test_replay_deals_the_board_between_streets():
    points = list(simple_history().replay())
    assert points[1].state.board == []
    assert points[2].state.board == parse_cards("Qs2s9c")


def test_hero_decisions_filters_to_the_hero():
    points = simple_history().hero_decisions()
    assert len(points) == 2
    assert all(p.is_hero for p in points)
    assert [p.action.type for p in points] == [ActionType.RAISE, ActionType.BET]


def test_final_state_matches_replaying_by_hand():
    history = simple_history()
    final = history.final_state()
    assert final.street is Street.FLOP
    assert final.total_pot == 14.0
    assert final.player("hero").stack == 93.0


def test_board_property_concatenates_street_cards():
    assert simple_history().board == parse_cards("Qs2s9c")


def test_replay_is_repeatable():
    history = simple_history()
    first = [str(p) for p in history.replay()]
    second = [str(p) for p in history.replay()]
    assert first == second


def test_illegal_recorded_action_surfaces_on_replay():
    history = simple_history(
        streets=[
            StreetRecord(
                street=Street.PREFLOP,
                actions=[
                    A("hero", ActionType.RAISE, 3.0),
                    # A check facing a raise is not legal.
                    A("villain", ActionType.CHECK),
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="cannot check"):
        history.final_state()


# -------------------------------------------------------------- validation


def test_hero_must_be_seated():
    with pytest.raises(ValueError, match="not seated"):
        simple_history(hero="nobody")


def test_unknown_actor_is_rejected():
    with pytest.raises(ValueError, match="unknown player"):
        simple_history(
            streets=[
                StreetRecord(
                    street=Street.PREFLOP,
                    actions=[A("ghost", ActionType.FOLD)],
                )
            ]
        )


def test_streets_must_be_in_order():
    with pytest.raises(ValueError, match="must run in order"):
        simple_history(
            streets=[
                StreetRecord(street=Street.FLOP, cards=parse_cards("Qs2s9c")),
            ]
        )


def test_streets_cannot_skip():
    with pytest.raises(ValueError, match="must run in order"):
        simple_history(
            streets=[
                StreetRecord(street=Street.PREFLOP),
                StreetRecord(street=Street.TURN, cards=parse_cards("Qs2s9c7h")),
            ]
        )


def test_preflop_cannot_deal_cards():
    with pytest.raises(ValueError, match="no cards are dealt"):
        simple_history(
            streets=[
                StreetRecord(street=Street.PREFLOP, cards=parse_cards("Qs")),
            ]
        )


def test_duplicate_seat_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate seat names"):
        simple_history(
            seats=[
                SeatRecord(name="x", position=Position.BTN, starting_stack=100.0),
                SeatRecord(name="x", position=Position.BB, starting_stack=100.0),
            ],
            hero="x",
            streets=[],
        )


def test_empty_seating_is_rejected():
    with pytest.raises(ValueError):
        simple_history(seats=[], streets=[])


# ------------------------------------------------------------- persistence


def test_json_round_trip_preserves_the_hand(tmp_path):
    history = simple_history()
    path = tmp_path / "hand.json"
    history.to_json_file(str(path))
    restored = HandHistory.from_json_file(str(path))
    assert restored == history
    assert restored.final_state().total_pot == history.final_state().total_pot


def test_saved_file_is_human_readable(tmp_path):
    path = tmp_path / "hand.json"
    simple_history().to_json_file(str(path))
    text = path.read_text()
    assert '"AsKs"' not in text  # cards serialise individually
    assert '"As"' in text
    assert text.endswith("\n")


@pytest.mark.parametrize(
    "name",
    [
        "thin-river-call",
        "dominated-ace-4bet-call",
        "river-fold-facing-shove",
    ],
)
def test_bundled_examples_load_and_replay_legally(name):
    history = HandHistory.from_json_file(str(EXAMPLES / f"{name}.json"))
    points = list(history.replay())
    assert points, "example hand recorded no actions"
    assert history.final_state().total_pot > 0
    assert history.hero_decisions()


def test_bundled_example_pot_math():
    history = HandHistory.from_json_file(str(EXAMPLES / "thin-river-call.json"))
    # 3+3 preflop, 4+4 flop, 10+10 turn, 30+30 river.
    assert history.final_state().total_pot == 94.0
