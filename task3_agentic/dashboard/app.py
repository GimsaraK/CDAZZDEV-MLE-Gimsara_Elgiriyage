"""Task 3 bonus: a Streamlit dashboard over agent_trace.jsonl and session_events.jsonl.

Run from the repository root:

    streamlit run task3_agentic/dashboard/app.py

Everything is read from the two log files; nothing calls an LLM or a data source.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)

import sys
from pathlib import Path

import altair as alt
import streamlit as st

# task3_agentic/dashboard/app.py -> repo root, so the package imports when launched by `streamlit run`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task3_agentic.src import config  # noqa: E402
from task3_agentic.src.dashboard_data import (  # noqa: E402
    filter_trace,
    load_events,
    load_trace,
    per_session,
    summarise,
    timeline,
)

# Rows given an expander with the full args and output; beyond this the table is enough.
MAX_EXPANDERS = 60

st.set_page_config(page_title="Agent Trace", layout="wide")
st.title("Agent trace dashboard")
st.caption("Task 3: every tool call from `agent_trace.jsonl`, plus cache and memory events from `session_events.jsonl`.")

with st.sidebar:
    st.header("Source")
    trace_path = Path(st.text_input("Trace file", str(config.TRACE_FILE)))
    events_path = Path(st.text_input("Events file", str(config.EVENTS_FILE)))

trace = load_trace(trace_path)
events = load_events(events_path)
if trace.empty:
    st.warning(f"No tool calls found in {trace_path}. Run the Task 3 notebook first.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    # dict.fromkeys de-duplicates while keeping first-seen order, so sessions are listed oldest first
    # (sorted() would order the random-looking session ids alphabetically instead).
    sessions = st.multiselect("Session", list(dict.fromkeys(trace["session_id"])))
    agents = st.multiselect("Agent", sorted(trace["agent"].dropna().unique()))
    tools = st.multiselect("Tool", sorted(trace["tool"].dropna().unique()))
    statuses = st.multiselect("Status", sorted(trace["status"].dropna().unique()))

view = filter_trace(trace, sessions, agents, tools, statuses)
kpi = summarise(view)
columns = st.columns(7)
for column, (label, key) in zip(
    columns,
    [("Tool calls", "calls"), ("Sessions", "sessions"), ("Errors", "errors"), ("Empty", "empty"), ("Denied", "denied"), ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms")],
):
    column.metric(label, kpi[key])

st.subheader("Sessions")
st.dataframe(per_session(view), hide_index=True)

st.subheader("Timeline")
session_ids = list(dict.fromkeys(view["session_id"]))
chosen = st.selectbox("Session for the timeline", session_ids) if session_ids else None
if chosen:
    chart_data = timeline(view[view["session_id"] == chosen])
    # A Gantt-style chart: each call is one horizontal bar from x (its start) to x2 (its end), on its own
    # row (y). sort=None keeps the rows in call order, and the height grows with the number of calls.
    # labelOverlap=False stops Vega-Lite hiding every other row label when rows are close together.
    x_title = "seconds since the session's first call"
    base = alt.Chart(chart_data).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=0)),
        color=alt.Color("agent:N"),
        tooltip=["tool", "agent", "status", "duration_ms", "args"],
    )
    bars = base.mark_bar().encode(x=alt.X("start_s:Q", title=x_title), x2="end_s:Q")
    # A tick at each call's start, so a call of a few milliseconds is still visible next to one of 30 s.
    starts = base.mark_tick(thickness=3).encode(x=alt.X("start_s:Q", title=x_title))
    chart = (bars + starts).properties(height=max(150, 36 * len(chart_data)))
    st.altair_chart(chart)

st.subheader("Tool calls")
st.dataframe(view[["timestamp", "session_id", "agent", "tool", "status", "duration_ms", "args"]], hide_index=True)
for _, row in view.tail(MAX_EXPANDERS).iterrows():
    with st.expander(f"{row['timestamp']:%H:%M:%S}  {row['agent']} -> {row['tool']}  [{row['status']}]  {row['duration_ms']:.0f} ms"):
        st.markdown("**Arguments**")
        st.code(row["args"], language="json")
        st.markdown(f"**Output** (first {config.TRACE_OUTPUT_MAX_CHARS} characters, as logged)")
        st.code(row["output"], language="json")
        # Calls without an error have NaN here (pandas fills the gap), and NaN is truthy, so test for real text.
        if isinstance(row["error"], str) and row["error"]:
            st.error(row["error"])

st.subheader("Session events")
if events.empty:
    st.info("No cache or memory events logged yet.")
else:
    # Events have no agent/tool/status columns, so only the session filter applies to them.
    shown = events if not sessions else events[events["session_id"].isin(sessions)]
    st.dataframe(shown, hide_index=True)
