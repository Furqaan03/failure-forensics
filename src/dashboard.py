"""Streamlit trace explorer: visualize spans, run diagnosis, flag failures."""
from __future__ import annotations

import streamlit as st

from src.analyzer.eval_dataset import add_eval_case, failure_analytics
from src.analyzer.root_cause import diagnose
from src.tracing.trace import list_traces, load_trace

st.set_page_config(page_title="Failure Forensics", layout="wide")
st.title("Failure Forensics — Trace Explorer")

_STATUS_COLOR = {"success": "🟢", "failure": "🔴", "degraded": "🟡"}

traces = list_traces()
if not traces:
    st.info("No traces yet. Run documents through POST /v1/process first.")
    st.stop()

selected = st.selectbox(
    "Select a trace",
    options=[t["trace_id"] for t in traces],
    format_func=lambda tid: f"{_STATUS_COLOR.get(next(t['status'] for t in traces if t['trace_id']==tid), '⚪')} {tid[:8]}",
)

trace = load_trace(selected)
st.subheader(f"Trace {trace.trace_id[:8]} — {trace.status}")

for span in trace.spans:
    conf = f" · confidence {span.confidence}/5" if span.confidence else ""
    marker = "🔴" if span.error else ("🟡" if (span.confidence or 5) <= 2 else "🟢")
    with st.expander(f"{marker} {span.step_name}{conf} · {span.latency_ms:.0f}ms"):
        st.text(f"Input: {span.input_serialized[:500]}")
        st.text(f"Output: {span.output_serialized[:500]}")
        if span.llm_prompt:
            st.caption("LLM prompt")
            st.code(span.llm_prompt[:800])
        if span.error:
            st.error(span.error)

if st.button("Diagnose root cause"):
    with st.spinner("Analyzing..."):
        d = diagnose(trace)
    if d.root_cause_step:
        st.error(f"Root cause: **{d.root_cause_step}** ({d.failure_type})")
        st.write(d.evidence)
    else:
        st.success("No step scored below the quality threshold.")
    for q in d.step_qualities:
        st.write(f"- {q.step_name}: {q.quality_score}/5 — {q.reasoning}")

st.divider()
st.subheader("Failure analytics")
st.json(failure_analytics())
