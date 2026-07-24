import pytest

from poker_coach.cli import main


def test_eval_prints_the_hand_and_best_five(capsys):
    assert main(["eval", "AsKsQsJsTs2h3d"]) == 0
    out = capsys.readouterr().out
    assert "royal flush" in out
    assert "AsKsQsJsTs" in out


def test_eval_names_a_pair(capsys):
    assert main(["eval", "AsAh9d5c2s"]) == 0
    assert "a pair of aces" in capsys.readouterr().out


def test_equity_reports_an_exact_river_result(capsys):
    assert main(["equity", "AsKs", "AhKh", "--board", "QsJs2s7d3c"]) == 0
    out = capsys.readouterr().out
    assert "exact" in out
    assert "100.00%" in out
    assert "win 100.00%" in out


def test_equity_against_a_range_is_seeded_and_reproducible(capsys):
    main(["equity", "AsAh", "77+", "--iterations", "500", "--seed", "1"])
    first = capsys.readouterr().out
    main(["equity", "AsAh", "77+", "--iterations", "500", "--seed", "1"])
    assert capsys.readouterr().out == first


def test_equity_multiway(capsys):
    assert main(["equity", "AsAh", "KsKh", "QcQd", "--iterations", "300", "--seed", "2"]) == 0
    assert "Equity:" in capsys.readouterr().out


def test_odds_reports_price_mdf_and_alpha(capsys):
    assert main(["odds", "12", "6"]) == 0
    out = capsys.readouterr().out
    assert "need 33.3%" in out
    assert "MDF: 50.0%" in out
    assert "Alpha: 50.0%" in out


def test_odds_with_equity_reports_ev(capsys):
    assert main(["odds", "12", "6", "--equity", "50"]) == 0
    out = capsys.readouterr().out
    assert "EV of calling at 50% equity: +3.00 chips" in out


def test_range_lists_classes(capsys):
    assert main(["range", "QQ+"]) == 0
    out = capsys.readouterr().out
    assert "18 combos" in out
    assert "AA KK QQ" in out


def test_range_lists_combos_when_asked(capsys):
    assert main(["range", "AA", "--combos"]) == 0
    out = capsys.readouterr().out
    assert "6 combos" in out
    assert "AsAh" in out or "AhAs" in out


EXAMPLE = "examples/hands/dominated-ace-4bet-call.json"


def test_review_replay_prints_the_action_sequence(capsys):
    assert main(["review", EXAMPLE, "--replay"]) == 0
    out = capsys.readouterr().out
    assert "[preflop] hero raise 6" in out
    assert "[turn] hero fold" in out
    assert "Final pot: 139" in out


def test_review_marks_hero_actions(capsys):
    main(["review", EXAMPLE, "--replay"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "]" in ln]
    hero_lines = [ln for ln in lines if ln.startswith("*")]
    assert len(hero_lines) == 6
    assert all("hero" in ln for ln in hero_lines)


def test_review_prints_decision_facts(capsys):
    assert main(["review", EXAMPLE, "--iterations", "400", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "DECISION 1 (preflop)" in out
    assert "Action taken:" in out
    assert "Hero equity" in out


def test_review_respects_the_villain_range_flag(capsys):
    main(["review", EXAMPLE, "--villain-range", "QQ+",
          "--iterations", "300", "--seed", "1"])
    assert "Assumed villain range: QQ+" in capsys.readouterr().out


def test_review_is_reproducible_under_a_seed(capsys):
    main(["review", EXAMPLE, "--iterations", "300", "--seed", "7"])
    first = capsys.readouterr().out
    main(["review", EXAMPLE, "--iterations", "300", "--seed", "7"])
    assert capsys.readouterr().out == first


CHECK_SPOT = [
    "--hero", "AsKs", "--board", "Qs2s9c",
    "--pot", "12", "--to-call", "6",
    "--iterations", "4000", "--seed", "11",
]


def test_check_passes_a_faithful_quote(capsys):
    text = "Call. You have 71.80% equity and need 33.3%, worth +6.92 chips."
    assert main(["check", text, *CHECK_SPOT]) == 0
    assert "All 3 numeric claim(s) trace to the facts" in capsys.readouterr().out


def test_check_fails_an_invented_figure(capsys):
    assert main(["check", "Call. You have 88% equity.", *CHECK_SPOT]) == 1
    out = capsys.readouterr().out
    assert "not in the facts" in out
    assert "88%" in out


def test_check_flags_an_invented_ratio(capsys):
    assert main(["check", "Call, you're a 4:1 favourite.", *CHECK_SPOT]) == 1
    assert "4:1" in capsys.readouterr().out


def test_check_accepts_a_response_with_no_numbers(capsys):
    assert main(["check", "Call. You're ahead.", *CHECK_SPOT]) == 0
    assert "No numeric claims" in capsys.readouterr().out


def test_check_exit_code_is_usable_as_a_gate():
    good = main(["check", "Call. 71.80% equity.", *CHECK_SPOT])
    bad = main(["check", "Call. 88% equity.", *CHECK_SPOT])
    assert (good, bad) == (0, 1)


def test_check_reads_stdin_when_given_a_dash(capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("Call. You have 88% equity."))
    assert main(["check", "-", *CHECK_SPOT]) == 1
    assert "88%" in capsys.readouterr().out


def test_check_rejects_an_impossible_board():
    with pytest.raises(ValueError, match="not a street"):
        main(["check", "Call.", "--hero", "AsKs", "--board", "Qs2s",
              "--pot", "12", "--to-call", "6"])


def test_check_works_preflop_with_no_board(capsys):
    assert main(["check", "Raise.", "--hero", "AsKs", "--pot", "3",
                 "--to-call", "1", "--iterations", "500", "--seed", "1"]) == 0
    assert "No numeric claims" in capsys.readouterr().out


def test_missing_command_exits_with_an_error():
    with pytest.raises(SystemExit):
        main([])


def test_bad_cards_surface_as_an_error():
    with pytest.raises(ValueError):
        main(["eval", "ZsKsQsJsTs"])
