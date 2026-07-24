import pytest

from poker_coach.domain import (
    Action,
    ActionType,
    HandState,
    PlayerState,
    Position,
    Street,
    parse_cards,
)


def make_state(**overrides) -> HandState:
    """A standard 100bb heads-up pot, blinds already posted."""

    players = [
        PlayerState(
            name="hero",
            position=Position.BTN,
            stack=99.0,
            hole_cards=tuple(parse_cards("AsKd")),
            committed_this_street=1.0,
            committed_total=1.0,
            is_hero=True,
        ),
        PlayerState(
            name="villain",
            position=Position.BB,
            stack=99.0,
            hole_cards=tuple(parse_cards("7h7c")),
            committed_this_street=1.0,
            committed_total=1.0,
        ),
    ]
    defaults = dict(players=players, small_blind=0.5, big_blind=1.0)
    defaults.update(overrides)
    return HandState(**defaults)


def test_total_pot_includes_live_street_bets():
    assert make_state().total_pot == 2.0


def test_current_bet_is_the_highest_street_commitment():
    assert make_state().current_bet == 1.0


def test_amount_to_call_is_zero_when_bets_are_level():
    assert make_state().amount_to_call("hero") == 0.0


def test_amount_to_call_is_capped_by_stack():
    state = make_state()
    state = state.apply(Action(actor="hero", type=ActionType.RAISE, amount=100.0))
    assert state.player("hero").is_all_in
    assert state.amount_to_call("villain") == 99.0


def test_apply_returns_a_new_state_and_leaves_the_original_alone():
    state = make_state()
    after = state.apply(Action(actor="hero", type=ActionType.RAISE, amount=3.0))
    assert state.player("hero").stack == 99.0
    assert after.player("hero").stack == 97.0
    assert after.total_pot == 4.0


def test_raise_uses_to_amount_semantics():
    state = make_state().apply(
        Action(actor="hero", type=ActionType.RAISE, amount=3.0)
    )
    hero = state.player("hero")
    assert hero.committed_this_street == 3.0
    assert hero.committed_total == 3.0
    assert hero.stack == 97.0


def test_call_must_match_the_current_bet_exactly():
    state = make_state().apply(
        Action(actor="hero", type=ActionType.RAISE, amount=3.0)
    )
    with pytest.raises(ValueError, match="call must bring"):
        state.apply(Action(actor="villain", type=ActionType.CALL, amount=2.0))

    called = state.apply(Action(actor="villain", type=ActionType.CALL, amount=3.0))
    assert called.total_pot == 6.0


def test_cannot_check_facing_a_bet():
    state = make_state().apply(
        Action(actor="hero", type=ActionType.RAISE, amount=3.0)
    )
    with pytest.raises(ValueError, match="cannot check"):
        state.apply(Action(actor="villain", type=ActionType.CHECK))


def test_cannot_bet_when_facing_a_bet():
    state = make_state()
    with pytest.raises(ValueError, match="use raise"):
        state.apply(Action(actor="hero", type=ActionType.BET, amount=3.0))


def test_raise_must_exceed_the_current_bet():
    state = make_state().apply(
        Action(actor="hero", type=ActionType.RAISE, amount=5.0)
    )
    with pytest.raises(ValueError, match="does not exceed"):
        state.apply(Action(actor="villain", type=ActionType.RAISE, amount=4.0))


def test_cannot_exceed_stack():
    state = make_state()
    with pytest.raises(ValueError, match="cannot put in"):
        state.apply(Action(actor="hero", type=ActionType.RAISE, amount=500.0))


def test_folded_and_all_in_players_cannot_act():
    folded = make_state().apply(Action(actor="hero", type=ActionType.FOLD))
    assert folded.player("hero").has_folded
    assert [p.name for p in folded.active_players] == ["villain"]
    with pytest.raises(ValueError, match="already folded"):
        folded.apply(Action(actor="hero", type=ActionType.CHECK))

    shoved = make_state().apply(
        Action(actor="hero", type=ActionType.RAISE, amount=100.0)
    )
    with pytest.raises(ValueError, match="all-in"):
        shoved.apply(Action(actor="hero", type=ActionType.CHECK))


def test_actions_accumulate_in_order():
    state = make_state()
    state = state.apply(Action(actor="hero", type=ActionType.RAISE, amount=3.0))
    state = state.apply(Action(actor="villain", type=ActionType.CALL, amount=3.0))
    assert [a.type for a in state.actions] == [ActionType.RAISE, ActionType.CALL]


def test_advance_street_collects_bets_and_deals_the_board():
    state = make_state()
    state = state.apply(Action(actor="hero", type=ActionType.RAISE, amount=3.0))
    state = state.apply(Action(actor="villain", type=ActionType.CALL, amount=3.0))

    flop = state.advance_street(parse_cards("AhKh2d"))
    assert flop.street is Street.FLOP
    assert flop.pot == 6.0
    assert flop.total_pot == 6.0
    assert all(p.committed_this_street == 0.0 for p in flop.players)
    assert flop.player("hero").committed_total == 3.0


def test_advance_street_validates_board_size():
    state = make_state()
    with pytest.raises(ValueError, match="needs 3 board cards"):
        state.advance_street(parse_cards("AhKh"))


def test_board_size_must_match_street():
    with pytest.raises(ValueError, match="expects 3 board cards"):
        make_state(street=Street.FLOP, board=parse_cards("AhKh"))


def test_duplicate_cards_are_rejected():
    with pytest.raises(ValueError, match="duplicate card"):
        make_state(street=Street.FLOP, board=parse_cards("AsKh2d"))


def test_duplicate_player_names_are_rejected():
    players = [
        PlayerState(name="x", position=Position.BTN, stack=100),
        PlayerState(name="x", position=Position.BB, stack=100),
    ]
    with pytest.raises(ValueError, match="duplicate player names"):
        HandState(players=players)


def test_duplicate_positions_are_rejected():
    players = [
        PlayerState(name="a", position=Position.BTN, stack=100),
        PlayerState(name="b", position=Position.BTN, stack=100),
    ]
    with pytest.raises(ValueError, match="duplicate positions"):
        HandState(players=players)


def test_effective_stack_is_the_smaller_of_the_two():
    state = make_state()
    state.players[1].stack = 40.0
    assert state.effective_stack("hero", "villain") == 41.0


def test_spr_uses_the_total_pot():
    state = make_state()
    state = state.apply(Action(actor="hero", type=ActionType.RAISE, amount=3.0))
    state = state.apply(Action(actor="villain", type=ActionType.CALL, amount=3.0))
    flop = state.advance_street(parse_cards("AhKh2d"))
    assert flop.spr("hero") == pytest.approx(97.0 / 6.0)


def test_hero_lookup_and_missing_player():
    state = make_state()
    assert state.hero is not None and state.hero.name == "hero"
    with pytest.raises(KeyError):
        state.player("nobody")


def test_street_actions_filters_by_street():
    state = make_state()
    state = state.apply(
        Action(actor="hero", type=ActionType.RAISE, amount=3.0, street=Street.PREFLOP)
    )
    state = state.apply(
        Action(actor="villain", type=ActionType.CALL, amount=3.0, street=Street.PREFLOP)
    )
    flop = state.advance_street(parse_cards("AhKh2d"))
    assert len(flop.street_actions(Street.PREFLOP)) == 2
    assert flop.street_actions() == []


def test_action_amount_validation():
    with pytest.raises(ValueError):
        Action(actor="x", type=ActionType.CHECK, amount=5.0)
    with pytest.raises(ValueError):
        Action(actor="x", type=ActionType.BET, amount=0.0)


def test_hand_state_round_trips_through_json():
    state = make_state().apply(
        Action(actor="hero", type=ActionType.RAISE, amount=3.0)
    )
    restored = HandState.model_validate_json(state.model_dump_json())
    assert restored.total_pot == state.total_pot
    assert restored.player("hero").hole_cards == state.player("hero").hole_cards
