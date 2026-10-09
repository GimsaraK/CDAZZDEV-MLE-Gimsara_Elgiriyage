"""Task 3 bonus: a Streamlit dashboard over agent_trace.jsonl and session_events.jsonl.

Run from the repository root:

    streamlit run task3_agentic/dashboard/app.py

Everything is read from the two log files; nothing calls an LLM or a data source.
"""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)

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
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("start_s:Q", title="seconds since the session's first call"),
            x2="end_s:Q",
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color("agent:N"),
            tooltip=["tool", "agent", "status", "duration_ms", "args"],
        )
        .properties(height=max(120, 28 * len(chart_data)))
    )
    st.altair_chart(chart)

st.subheader("Tool calls")
st.dataframe(view[["timestamp", "session_id", "agent", "tool", "status", "duration_ms", "args"]], hide_index=True)
for _, row in view.tail(MAX_EXPANDERS).iterrows():
    with st.expander(f"{row['timestamp']:%H:%M:%S}  {row['agent']} -> {row['tool']}  [{row['status']}]  {row['duration_ms']:.0f} ms"):
        st.markdown("**Arguments**")
        st.code(row["args"], language="json")
        st.markdown(f"**Output** (first {config.TRACE_OUTPUT_MAX_CHARS} characters, as logged)")
        st.code(row["output"], language="json")
        if row["error"]:
            st.error(row["error"])

st.subheader("Session events")
if events.empty:
    st.info("No cache or memory events logged yet.")
else:
    shown = events if not sessions else events[events["session_id"].isin(sessions)]
    st.dataframe(shown, hide_index=True)
