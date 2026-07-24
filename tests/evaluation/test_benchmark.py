import json

import pytest

from poker_coach.evaluation import (
    STANDARD_CASES,
    MANUAL_CHECKS,
    RunRecord,
    flag_record,
    load_jsonl,
    render_report,
    run_benchmark,
    run_case,
    write_jsonl,
)
from poker_coach.evaluation.cases import Case
from poker_coach.models.base import EchoModel, ScriptedModel


def case(name: str) -> Case:
    return next(c for c in STANDARD_CASES if c.name == name)


def scripted(*replies: str) -> ScriptedModel:
    return ScriptedModel(replies, name="scripted-1.2.3")


def run(name: str, reply: str, **kwargs) -> RunRecord:
    return run_case(
        case(name),
        scripted(reply),
        provider="test",
        requested_model="scripted-latest",
        iterations=kwargs.pop("iterations", 400),
        **kwargs,
    )


# ------------------------------------------------------------- recording


def test_a_record_captures_everything_needed_to_audit():
    record = run("river-drawing-dead", "Fold. Nothing here.")

    assert record.case_id == "river-drawing-dead"
    assert record.provider == "test"
    assert record.prompt_version
    assert record.seed
    assert record.iterations == 400
    assert record.facts_block
    assert record.response
    assert record.timestamp
    assert record.latency_seconds >= 0


def test_the_resolved_model_is_recorded_separately_from_the_request():
    """A floating alias must not be the only identity in the record."""

    record = run("river-drawing-dead", "Fold.")
    assert record.requested_model == "scripted-latest"
    assert record.resolved_model == "scripted-1.2.3"
    assert record.resolved_model != record.requested_model


def test_semantic_context_is_recorded():
    terminal = run("river-drawing-dead", "Fold.")
    non_terminal = run("flop-draw-future-action", "Call.")

    assert terminal.ev_is_terminal is True
    assert non_terminal.ev_is_terminal is False
    assert non_terminal.range_conditioning == "pre-action"


def test_exactness_of_equity_is_recorded():
    record = run("river-made-flush-exact", "Call.")
    assert record.equity_exact is True


def test_token_usage_and_stop_reason_are_recorded():
    record = run("river-drawing-dead", "Fold.")
    assert record.stop_reason == "end_turn"
    assert record.output_tokens >= 0


def test_a_correct_grounded_response_passes():
    record = run("river-drawing-dead", "Fold. You are drawing dead.")
    assert record.passed
    assert record.action_correct
    assert record.grounded


def test_an_invented_number_is_recorded_and_fails():
    record = run("river-drawing-dead", "Fold. You have 8.5% equity.")
    assert not record.passed
    assert record.action_correct        # right side of the line
    assert not record.grounded          # but dishonestly reached
    assert "8.5%" in record.ungrounded_claims


def test_a_raise_is_recorded_as_continuing():
    record = run("flop-raise-plausible", "Raise. Top set is ahead.")
    assert record.recommendation == "raise"
    assert record.verdict == "continue"
    assert record.action_correct


# ---------------------------------------------------------------- errors


def test_a_model_error_is_recorded_not_raised():
    class Exploding:
        name = "exploding"

        def complete(self, system, messages, *, max_tokens=1024):
            raise RuntimeError("provider exploded")

    record = run_case(
        case("river-drawing-dead"),
        Exploding(),
        provider="test",
        requested_model="exploding",
        iterations=400,
    )
    assert record.error is not None
    assert "provider exploded" in record.error
    assert not record.passed


def test_one_failing_case_does_not_abort_the_batch():
    class FailsOnce:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def complete(self, system, messages, *, max_tokens=1024):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            from poker_coach.models.base import ModelResponse

            return ModelResponse(text="Fold.", model="flaky")

    records = run_benchmark(
        FailsOnce(),
        provider="test",
        requested_model="flaky",
        cases=STANDARD_CASES[:3],
        iterations=300,
    )
    assert len(records) == 3
    assert records[0].error is not None
    assert all(r.error is None for r in records[1:])


# ------------------------------------------------------------ round trip


def test_records_round_trip_through_jsonl(tmp_path):
    records = run_benchmark(
        EchoModel(),
        provider="test",
        requested_model="echo",
        cases=STANDARD_CASES[:2],
        iterations=300,
    )
    path = write_jsonl(records, tmp_path / "run.jsonl")
    restored = load_jsonl(path)

    assert len(restored) == 2
    assert restored[0].case_id == records[0].case_id
    assert restored[0].facts_block == records[0].facts_block


def test_jsonl_is_one_valid_object_per_line(tmp_path):
    records = run_benchmark(
        EchoModel(), provider="test", requested_model="echo",
        cases=STANDARD_CASES[:2], iterations=300,
    )
    path = write_jsonl(records, tmp_path / "run.jsonl")
    for line in path.read_text().splitlines():
        assert json.loads(line)["case_id"]


def test_a_run_is_reproducible_for_the_deterministic_half(tmp_path):
    a = run("flop-draw-future-action", "Call.", seed=5)
    b = run("flop-draw-future-action", "Call.", seed=5)
    assert a.facts_block == b.facts_block


def test_a_different_seed_changes_the_sampled_facts():
    a = run("flop-draw-future-action", "Call.", seed=1)
    b = run("flop-draw-future-action", "Call.", seed=2)
    assert a.facts_block != b.facts_block


# ------------------------------------------------------------- reporting


def test_the_report_names_the_resolved_model():
    records = [run("river-drawing-dead", "Fold.")]
    text = render_report(records)
    assert "scripted-1.2.3" in text
    assert "scripted-latest" in text


def test_the_report_states_what_a_pass_does_not_mean():
    text = render_report([run("river-drawing-dead", "Fold.")])
    assert "does NOT mean the reasoning was good" in text


def test_the_report_lists_the_manual_checklist():
    text = render_report([run("river-drawing-dead", "Fold.")])
    for code, _ in MANUAL_CHECKS:
        assert code in text


def test_the_report_handles_an_empty_run():
    assert render_report([]) == "No records."


def test_the_report_includes_the_response_text():
    text = render_report([run("river-drawing-dead", "Fold. Drawing dead here.")])
    assert "Drawing dead here." in text


# ------------------------------------------------------ automatic flags


def test_invented_numbers_are_flagged():
    record = run("river-drawing-dead", "Fold. You have 8.5% equity vs that range.")
    codes = {f.code for f in flag_record(record)}
    assert "invented-numbers" in codes


def test_the_wrong_side_of_the_line_is_flagged():
    record = run("river-drawing-dead", "Call. Given the range you are fine.")
    codes = {f.code for f in flag_record(record)}
    assert "wrong-side-of-the-line" in codes


def test_a_non_terminal_ev_cited_without_realisation_is_flagged():
    record = run(
        "flop-draw-future-action",
        "Call. The EV of calling is +4.20 chips against that range.",
    )
    codes = {f.code for f in flag_record(record)}
    assert "ev-treated-as-realised" in codes


def test_mentioning_realisation_clears_that_flag():
    record = run(
        "flop-draw-future-action",
        "Call. EV is +4.20 chips against that range, though you will not "
        "realise all of it once future betting is accounted for.",
    )
    codes = {f.code for f in flag_record(record)}
    assert "ev-treated-as-realised" not in codes


def test_omitting_the_range_assumption_is_flagged():
    record = run("river-drawing-dead", "Fold. Nothing here.")
    codes = {f.code for f in flag_record(record)}
    assert "range-unmentioned" in codes


def test_an_unqualified_pre_action_range_is_flagged():
    record = run(
        "flop-draw-future-action",
        "Call. Your equity against that range clears the price.",
    )
    codes = {f.code for f in flag_record(record)}
    assert "pre-action-range-unqualified" in codes


def test_an_error_record_flags_only_the_error():
    record = RunRecord(
        case_id="x", case_description="", provider="p", requested_model="m",
        resolved_model=None, prompt_version="v", seed=1, iterations=1,
        error="Boom: bad",
    )
    flags = flag_record(record)
    assert [f.code for f in flags] == ["error"]


def test_flags_are_marked_automatic():
    record = run("river-drawing-dead", "Call. 99% equity.")
    assert all(f.automatic for f in flag_record(record))


def test_manual_checks_are_questions_not_assertions():
    """They must not read as verdicts — a human decides."""

    for _, question in MANUAL_CHECKS:
        assert question.strip().endswith("?")
