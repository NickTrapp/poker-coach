import io
import json
import random

import pytest

from poker_coach.domain import ActionType
from poker_coach.models.base import Message, ModelResponse, Usage
from poker_coach.practice import (
    DEFAULT_RANGES,
    CritiqueResult,
    DecisionRecord,
    HeroTurn,
    PracticeConfig,
    PracticeSession,
    critique,
    load_records,
    write_records,
)
from poker_coach.practice.cli import parse_action, render_turn, run_practice


class ScriptedModel:
    """Replies from a list; records what it was asked."""

    def __init__(self, replies, name="scripted"):
        self.replies = list(replies)
        self.name = name
        self.calls = []

    def complete(self, system, messages, *, max_tokens=1024):
        self.calls.append((system, list(messages)))
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return ModelResponse(
            text=text, model=self.name, stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=20),
        )


class ExplodingModel:
    name = "boom"

    def complete(self, system, messages, *, max_tokens=1024):
        raise RuntimeError("network is down")


def scripted_session(actions, **kwargs):
    """A session whose hero always plays the first legal action in `actions`."""

    chosen = iter(actions)

    def on_turn(turn: HeroTurn):
        want = next(chosen, ActionType.FOLD)
        options = [o for o in turn.legal if o.type is want]
        option = options[0] if options else turn.legal[0]
        from poker_coach.domain.action import Action

        return Action(actor="you", type=option.type,
                      amount=option.min_amount, street=turn.state.street)

    return PracticeSession(
        config=PracticeConfig(iterations=200, opponent_iterations=60, **kwargs),
        rng=random.Random(5), on_turn=on_turn,
        **({} if "model" not in kwargs else {}),
    )


def one_analysis():
    """An analysis from a real played turn, for critique tests."""

    captured = {}

    def on_turn(turn: HeroTurn):
        captured.setdefault("turn", turn)
        from poker_coach.domain.action import Action

        option = turn.legal[0]
        return Action(actor="you", type=option.type, amount=option.min_amount,
                      street=turn.state.street)

    session = PracticeSession(
        config=PracticeConfig(iterations=200, opponent_iterations=60),
        rng=random.Random(3), on_turn=on_turn,
    )
    session.play_hand(seed=11)
    return captured["turn"].analysis


# ------------------------------------------------------------- the session


def test_a_hand_plays_through_to_a_result():
    session = scripted_session([ActionType.CALL, ActionType.CHECK])
    result = session.play_hand(seed=7)
    assert result.pot > 0
    assert session.hands_played == 1


def test_every_hero_decision_is_recorded():
    session = scripted_session([ActionType.CALL, ActionType.CHECK])
    session.play_hand(seed=7)
    assert session.records
    assert all(r.hand_number == 1 for r in session.records)
    assert [r.decision_number for r in session.records] == list(
        range(1, len(session.records) + 1)
    )


def test_the_hero_only_ever_sees_legal_options():
    seen = []

    def on_turn(turn: HeroTurn):
        seen.append(turn)
        from poker_coach.domain.action import Action

        option = turn.legal[0]
        return Action(actor="you", type=option.type, amount=option.min_amount,
                      street=turn.state.street)

    session = PracticeSession(
        config=PracticeConfig(iterations=200, opponent_iterations=60),
        rng=random.Random(2), on_turn=on_turn,
    )
    session.play_hand(seed=4)
    for turn in seen:
        assert turn.legal == turn.state.legal_actions("you")
        assert turn.legal, "a turn was offered with no legal action"


def test_the_button_alternates_between_hands():
    session = scripted_session([ActionType.FOLD])
    session.play_hand(seed=1)
    first = session.records[0]
    session.play_hand(seed=2)
    second = [r for r in session.records if r.hand_number == 2][0]
    assert first.seed != second.seed or first.hero_cards != second.hero_cards


def test_an_unknown_opponent_is_rejected():
    with pytest.raises(ValueError, match="unknown opponent"):
        PracticeSession(config=PracticeConfig(opponent_style="genius"))


def test_a_session_without_a_callback_says_so():
    session = PracticeSession(config=PracticeConfig(iterations=100))
    with pytest.raises(ValueError, match="needs an on_turn callback"):
        session.play_hand(seed=1)


# ------------------------------------------------ range is configuration


def test_every_archetype_has_a_configured_range():
    from poker_coach.players import STYLES

    assert set(DEFAULT_RANGES) == set(STYLES)


def test_the_config_states_where_the_range_came_from():
    described = PracticeConfig(opponent_style="station").describe()
    assert "not inferred from play" in described
    assert "pre-action" in described


def test_the_record_carries_the_range_and_its_source():
    session = scripted_session([ActionType.CALL])
    session.play_hand(seed=7)
    record = session.records[0]
    assert record.villain_range == DEFAULT_RANGES["station"]
    assert record.range_source == "fixed practice configuration, not inferred"
    assert record.range_conditioning == "pre-action"


# ---------------------------------------------------------------- feedback


def test_feedback_without_a_model_still_has_arithmetic():
    result = critique(one_analysis(), "call to 1", opponent="station", model=None)
    assert result.feedback.verified
    assert not result.feedback.has_interpretation
    assert result.model_name is None
    rendered = result.feedback.render()
    assert "VERIFIED CALCULATIONS" in rendered
    assert "EXPLOITATIVE INTERPRETATION" not in rendered


def test_verified_lines_name_their_assumption_and_exactness():
    analysis = one_analysis()
    result = critique(analysis, "call", opponent="station", model=None)
    text = " ".join(result.feedback.verified)
    assert "assumed range" in text
    if not analysis.ev_is_terminal and analysis.facing_bet:
        assert "upper bound" in text


def test_unresolved_lines_come_from_the_open_facts():
    analysis = one_analysis()
    result = critique(analysis, "call", opponent="station", model=None)
    assert result.feedback.unresolved
    assert any("raising" in line for line in result.feedback.unresolved)


def test_a_grounded_reply_is_shown():
    analysis = one_analysis()
    grounded = (
        f"A station calls too wide, so your value hands get paid more often "
        f"than the {analysis.equity.equity_pct:.2f}% figure alone suggests."
    )
    result = critique(analysis, "call", opponent="station",
                      model=ScriptedModel([grounded]))
    assert result.feedback.has_interpretation
    assert not result.fell_back
    assert not result.repair_attempted
    assert "EXPLOITATIVE INTERPRETATION" in result.feedback.render()


def test_an_ungrounded_reply_triggers_exactly_one_repair():
    analysis = one_analysis()
    model = ScriptedModel([
        "You have 91.4% equity here.",                    # invented
        "A station calls far too wide; keep value betting.",  # clean
    ])
    result = critique(analysis, "call", opponent="station", model=model)
    assert result.repair_attempted
    assert len(model.calls) == 2
    assert result.feedback.has_interpretation
    assert not result.fell_back


def test_the_repair_request_quotes_the_failed_claims():
    analysis = one_analysis()
    model = ScriptedModel(["You have 91.4% equity.", "Fine now."])
    critique(analysis, "call", opponent="station", model=model)
    repair_prompt = model.calls[1][1][-1].content
    assert "91.4%" in repair_prompt
    assert "not in the FACTS block" in repair_prompt


def test_a_persistently_ungrounded_reply_is_withheld():
    analysis = one_analysis()
    model = ScriptedModel(["You have 91.4% equity.", "Still 91.4% equity."])
    result = critique(analysis, "call", opponent="station", model=model)

    assert result.repair_attempted
    assert result.fell_back
    assert not result.feedback.has_interpretation
    rendered = result.feedback.render()
    assert "withheld" in rendered
    assert "VERIFIED CALCULATIONS" in rendered   # the arithmetic survives


def test_a_model_failure_falls_back_rather_than_crashing():
    result = critique(one_analysis(), "call", opponent="station",
                      model=ExplodingModel())
    assert result.fell_back
    assert result.error and "RuntimeError" in result.error
    assert result.feedback.verified
    assert "could not be reached" in result.feedback.render()


def test_the_practice_prompt_forbids_restating_the_arithmetic():
    from poker_coach.practice import PRACTICE_SYSTEM_PROMPT

    flat = " ".join(PRACTICE_SYSTEM_PROMPT.lower().split())
    assert "do not restate" in flat
    assert "never state a number that is not in the facts block" in flat
    assert "not something the system inferred" in flat
    assert "do not claim an action is optimal" in flat


# ----------------------------------------------------------------- logging


def test_a_record_carries_everything_needed_to_reconstruct_the_decision():
    session = scripted_session([ActionType.CALL])
    session.play_hand(seed=7)
    record = session.records[0]

    for field in ("seed", "street", "hero_cards", "facts_block", "action_taken",
                  "legal_actions", "equity_pct", "ev_call_vs_fold",
                  "prompt_version", "latency_seconds"):
        assert getattr(record, field) is not None, field
    assert record.legal_actions
    assert "EXACT STATE FACTS" in record.facts_block


def test_records_round_trip_through_jsonl(tmp_path):
    session = scripted_session([ActionType.CALL, ActionType.CHECK])
    session.play_hand(seed=7)
    path = write_records(tmp_path / "s.jsonl", session.records)

    loaded = list(load_records(path))
    assert len(loaded) == len(session.records)
    assert loaded[0].facts_block == session.records[0].facts_block
    assert loaded[0].seed == session.records[0].seed


def test_the_log_is_valid_json_per_line(tmp_path):
    session = scripted_session([ActionType.CALL])
    session.play_hand(seed=7)
    path = write_records(tmp_path / "s.jsonl", session.records)
    for line in path.read_text().splitlines():
        json.loads(line)


def test_a_withheld_reply_is_still_logged_in_full():
    analysis = one_analysis()
    model = ScriptedModel(["You have 91.4% equity.", "Still 91.4% equity."])
    result = critique(analysis, "call", opponent="station", model=model)
    record = DecisionRecord.build(
        hand_number=1, decision_number=1, seed=1,
        config=PracticeConfig(), state=None, legal=[],
        action=_dummy_action(), analysis=analysis, critique=result,
    )
    assert record.interpretation is None      # not shown
    assert len(record.raw_responses) == 2     # but both attempts are kept
    assert record.ungrounded_claims
    assert record.repair_attempted and record.fell_back


def _dummy_action():
    from poker_coach.domain.action import Action

    return Action(actor="you", type=ActionType.CALL, amount=1.0)


# --------------------------------------------------------------------- CLI


def test_parse_action_reads_abbreviations():
    session = scripted_session([ActionType.CALL])
    captured = {}

    def on_turn(turn):
        captured["turn"] = turn
        from poker_coach.domain.action import Action

        return Action(actor="you", type=turn.legal[0].type,
                      amount=turn.legal[0].min_amount, street=turn.state.street)

    session.on_turn = on_turn
    session.play_hand(seed=7)
    turn = captured["turn"]

    for text, kind in [("f", ActionType.FOLD), ("fold", ActionType.FOLD)]:
        if any(o.type is kind for o in turn.legal):
            assert parse_action(text, turn, "you").type is kind


def test_parse_action_rejects_an_illegal_action():
    session = scripted_session([ActionType.CALL])
    captured = {}

    def on_turn(turn):
        captured["turn"] = turn
        from poker_coach.domain.action import Action

        return Action(actor="you", type=turn.legal[0].type,
                      amount=turn.legal[0].min_amount, street=turn.state.street)

    session.on_turn = on_turn
    session.play_hand(seed=7)
    turn = captured["turn"]

    illegal = next(
        (k for k in (ActionType.CHECK, ActionType.BET)
         if not any(o.type is k for o in turn.legal)), None
    )
    if illegal is not None:
        with pytest.raises(ValueError, match="cannot"):
            parse_action(illegal.value, turn, "you")


def test_parse_action_rejects_a_bad_size():
    session = scripted_session([ActionType.CALL])
    captured = {}

    def on_turn(turn):
        captured["turn"] = turn
        from poker_coach.domain.action import Action

        return Action(actor="you", type=turn.legal[0].type,
                      amount=turn.legal[0].min_amount, street=turn.state.street)

    session.on_turn = on_turn
    session.play_hand(seed=7)
    turn = captured["turn"]

    sized = [o for o in turn.legal if o.min_amount != o.max_amount]
    if sized:
        with pytest.raises(ValueError, match="must be between"):
            parse_action(f"{sized[0].type.value} 0.01", turn, "you")


def test_the_cli_plays_a_hand_from_scripted_input(tmp_path):
    out = io.StringIO()
    records = run_practice(
        hands=1, opponent="station", model=None, seed=7,
        stdin=io.StringIO("c\nk\nk\nk\nk\nk\n"), stdout=out,
        log_dir=str(tmp_path), iterations=200,
    )
    text = out.getvalue()
    assert "PRACTICE" in text
    assert "not inferred from play" in text
    assert "VERIFIED CALCULATIONS" in text
    assert "hand 1" in text
    assert records


def test_the_cli_writes_a_session_log(tmp_path):
    out = io.StringIO()
    run_practice(
        hands=1, opponent="nit", model=None, seed=3,
        stdin=io.StringIO("f\nf\nf\nf\n"), stdout=out,
        log_dir=str(tmp_path), iterations=200,
    )
    logs = list(tmp_path.glob("session-*.jsonl"))
    assert len(logs) == 1
    assert list(load_records(logs[0]))


def test_the_cli_survives_end_of_input():
    out = io.StringIO()
    run_practice(
        hands=1, opponent="station", model=None, seed=7,
        stdin=io.StringIO(""), stdout=out, log_dir=None, iterations=200,
    )
    assert "no input" in out.getvalue()


def test_render_turn_shows_the_price_and_the_options():
    captured = {}

    def on_turn(turn):
        captured["turn"] = turn
        from poker_coach.domain.action import Action

        return Action(actor="you", type=turn.legal[0].type,
                      amount=turn.legal[0].min_amount, street=turn.state.street)

    session = PracticeSession(
        config=PracticeConfig(iterations=200, opponent_iterations=60),
        rng=random.Random(5), on_turn=on_turn,
    )
    session.play_hand(seed=7)
    text = render_turn(captured["turn"], "you")
    assert "your options:" in text
    assert "equity vs the assumed range" in text


def test_a_hand_you_never_act_in_is_announced():
    """Heads-up the small blind acts first, so a big blind can be folded to
    without ever seeing a decision. Silently skipping the hand read as a bug."""

    out = io.StringIO()
    run_practice(
        hands=4, opponent="nit", model=None, seed=11,
        stdin=io.StringIO("f\n" * 12), stdout=out, log_dir=None,
        iterations=200,
    )
    text = out.getvalue()
    assert "no decision for you" in text
    assert "the action never reached you" in text


def test_the_summary_explains_why_decisions_trail_hands():
    out = io.StringIO()
    run_practice(
        hands=2, opponent="nit", model=None, seed=11,
        stdin=io.StringIO("f\nf\nf\n"), stdout=out, log_dir=None,
        iterations=200,
    )
    assert "reach you for none" in out.getvalue()


def test_feedback_after_a_raise_does_not_read_as_a_verdict_on_the_raise():
    """Telling someone who raised that "calling beats folding" answers a
    question they did not ask about an action they did not take."""

    analysis = one_analysis()
    from poker_coach.domain.enums import ActionType as AT

    raised = critique(analysis, "raise to 2", opponent="station",
                      action_type=AT.RAISE, model=None)
    text = " ".join(raised.feedback.verified)
    assert "You raised" in text
    assert "does not evaluate your raise" in text


def test_feedback_after_a_call_and_a_fold_each_name_the_action():
    analysis = one_analysis()
    from poker_coach.domain.enums import ActionType as AT

    called = critique(analysis, "call", opponent="station",
                      action_type=AT.CALL, model=None)
    folded = critique(analysis, "fold", opponent="station",
                      action_type=AT.FOLD, model=None)
    assert "You called." in " ".join(called.feedback.verified)
    assert "You folded" in " ".join(folded.feedback.verified)


def sampled_and_exact_analyses():
    """One sampled analysis, and one where equity enumerates exactly."""

    import random as _random

    from poker_coach.coaching.analysis import analyze
    from poker_coach.domain import (
        Action, HandState, PlayerState, Position, Street, parse_cards,
    )

    sampled = analyze(
        HandState(
            players=[
                PlayerState(name="you", position=Position.BTN, stack=97.0,
                            hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
                PlayerState(name="v", position=Position.BB, stack=97.0),
            ],
            board=parse_cards("Qs2s9c"), street=Street.FLOP, pot=6.0,
        ).apply(
            Action(actor="v", type=ActionType.BET, amount=6.0, street=Street.FLOP)
        ),
        villain_range="22+", iterations=400, rng=_random.Random(1),
    )

    # River, single-combo range: enumerates, and the call is terminal.
    exact = analyze(
        HandState(
            players=[
                PlayerState(name="you", position=Position.BTN, stack=97.0,
                            hole_cards=tuple(parse_cards("AsKs")), is_hero=True),
                PlayerState(name="v", position=Position.BB, stack=97.0),
            ],
            board=parse_cards("Qs2s9c4d7h"), street=Street.RIVER, pot=20.0,
        ).apply(
            Action(actor="v", type=ActionType.BET, amount=10.0, street=Street.RIVER)
        ),
        villain_range="QhQd", rng=_random.Random(1),
    )
    return sampled, exact


def test_sampled_feedback_is_not_labelled_exact():
    """The heading must not claim exactness the numbers underneath lack."""

    sampled, _ = sampled_and_exact_analyses()
    assert sampled.equity.exact is False

    rendered = critique(sampled, "call", opponent="station",
                        action_type=ActionType.CALL, model=None).feedback.render()
    assert "Exact given" not in rendered
    assert "sampled equity estimate" in rendered
    assert "margin of error" in rendered


def test_enumerated_feedback_may_say_it_is_deterministic():
    _, exact = sampled_and_exact_analyses()
    assert exact.equity.exact is True

    rendered = critique(exact, "call", opponent="station",
                        action_type=ActionType.CALL, model=None).feedback.render()
    assert "under the stated assumptions" in rendered
    assert "sampled equity estimate" not in rendered


def test_a_terminal_call_is_not_called_numerically_exact():
    """A terminal *tree* is not an exact *figure* when equity was sampled."""

    from poker_coach.practice.feedback import verified_lines

    sampled, _ = sampled_and_exact_analyses()
    terminal_sampled = critique(sampled, "call", opponent="station",
                                action_type=ActionType.CALL, model=None)
    assert not any("exact —" in line for line in terminal_sampled.feedback.verified)

    _, exact = sampled_and_exact_analyses()
    lines = verified_lines(exact, "call", ActionType.CALL)
    assert exact.ev_is_terminal
    assert any("terminal decision tree" in line for line in lines)
    assert not any("exact —" in line for line in lines)
