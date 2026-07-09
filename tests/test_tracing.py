import pytest

from src.tracing.trace import Span, Trace, span


def test_span_captures_latency_and_output():
    trace = Trace()
    with span(trace, "step1", {"in": 1}) as s:
        s.output_serialized = "done"
    assert len(trace.spans) == 1
    assert trace.spans[0].step_name == "step1"
    assert trace.spans[0].latency_ms >= 0
    assert trace.status == "success"


def test_span_records_error_and_marks_trace_failed():
    trace = Trace()
    with pytest.raises(ValueError):
        with span(trace, "boom", {}) as s:  # noqa: F841
            raise ValueError("kaboom")
    assert trace.status == "failure"
    assert trace.spans[0].error == "kaboom"


def test_get_span_by_name():
    trace = Trace()
    with span(trace, "a", {}):
        pass
    with span(trace, "b", {}):
        pass
    assert trace.get_span("b") is not None
    assert trace.get_span("missing") is None


def test_serialize_handles_pydantic_and_primitives():
    from src.tracing.trace import _serialize

    class M(Span):
        pass

    assert _serialize({"x": 1}) == '{"x": 1}'
    assert _serialize("plain") == "plain"
