import pytest

from poker_coach.tracing.trace import Trace


def test_span_records_name_and_duration():
    trace = Trace()
    with trace.span("work"):
        pass
    assert len(trace) == 1
    assert trace.spans[0].name == "work"
    assert trace.spans[0].duration_ms >= 0.0


def test_tags_are_recorded():
    trace = Trace()
    with trace.span("call", model="echo", tokens=12):
        pass
    assert trace.spans[0].tags == {"model": "echo", "tokens": 12}
    assert "model=echo" in str(trace.spans[0])


def test_nesting_depth_is_tracked():
    trace = Trace()
    with trace.span("outer"):
        with trace.span("inner"):
            pass
    assert [s.depth for s in trace.spans] == [0, 1]
    assert str(trace.spans[1]).startswith("  inner")


def test_total_ms_counts_top_level_spans_only():
    trace = Trace()
    with trace.span("outer"):
        with trace.span("inner"):
            pass
    assert trace.total_ms == pytest.approx(trace.spans[0].duration_ms)


def test_exceptions_are_recorded_and_re_raised():
    trace = Trace()
    with pytest.raises(ValueError, match="boom"):
        with trace.span("failing"):
            raise ValueError("boom")
    span = trace.spans[0]
    assert span.error == "ValueError: boom"
    assert span.end is not None
    assert "ERROR: ValueError: boom" in str(span)


def test_depth_recovers_after_an_exception():
    trace = Trace()
    with pytest.raises(ValueError):
        with trace.span("failing"):
            raise ValueError("boom")
    with trace.span("after"):
        pass
    assert trace.spans[1].depth == 0


def test_find_filters_by_name():
    trace = Trace()
    with trace.span("a"):
        pass
    with trace.span("b"):
        pass
    with trace.span("a"):
        pass
    assert len(trace.find("a")) == 2
    assert trace.find("missing") == []


def test_disabled_trace_records_nothing():
    trace = Trace(enabled=False)
    with trace.span("work"):
        pass
    assert len(trace) == 0
    assert trace.total_ms == 0.0


def test_clear_resets_spans_and_depth():
    trace = Trace()
    with trace.span("work"):
        pass
    trace.clear()
    assert len(trace) == 0
    with trace.span("again"):
        pass
    assert trace.spans[0].depth == 0


def test_render_lists_every_span():
    trace = Trace()
    with trace.span("outer"):
        with trace.span("inner"):
            pass
    rendered = trace.render()
    assert "outer" in rendered and "inner" in rendered
    assert len(rendered.splitlines()) == 2


def test_unfinished_span_reports_zero_duration():
    trace = Trace()
    cm = trace.span("open")
    span = cm.__enter__()
    assert span.duration_ms == 0.0
    cm.__exit__(None, None, None)
    assert span.duration_ms >= 0.0
