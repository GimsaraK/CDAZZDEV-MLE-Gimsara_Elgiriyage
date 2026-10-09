"""Shared ReAct loop: one LLM turn, then the tools it asked for, until it stops or runs out of budget.

    START -> agent --(tool calls)--> tools -> agent
               +--(no tool calls / budget spent / LLM down)--> END

Used by the 3A research agent (agent.py, which adds a report writer and a report check
around it) and by both 3B agents (multi_agent.py, one loop per agent with its own tools).
The LLM picks every tool call; routing only follows its decisions.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)

import operator
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypedDict, TypeVar

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from . import config
from .logging_utils import get_logger
from .tracing import safe_error

logger = get_logger("react")

NODE_AGENT = "agent"
NODE_TOOLS = "tools"

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LoopState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    budget_exhausted: bool
    llm_error: Optional[str]
    events: Annotated[List[str], operator.add]


# --------------------------------------------------------------------------- helpers
def count_tool_rounds(messages: Sequence[BaseMessage]) -> int:
    """Agent turns that called at least one tool."""
    return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)


def last_agent_notes(messages: Sequence[BaseMessage], limit: int = config.AGENT_NOTES_MAX_CHARS) -> str:
    """The agent's latest text without tool calls: the summary it hands on."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls and str(message.content).strip():
            return str(message.content)[:limit]
    return "(no notes: the agent stopped before writing a summary)"


def models_used(messages: Sequence[BaseMessage]) -> Dict[str, int]:
    """How many agent turns each model answered (read from the providers' response metadata)."""
    counts: Dict[str, int] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            name = message.response_metadata.get("model_name") or "unknown"
            counts[name] = counts.get(name, 0) + 1
    return counts


def unwrap_structured(output: Any, schema: Type[SchemaT]) -> Tuple[SchemaT, Optional[str]]:
    """Accept a schema instance, a plain dict, or with_structured_output(include_raw=True) output.

    Returns (instance, model name). Raises when the model's output did not parse.
    """
    if isinstance(output, schema):
        return output, None
    if isinstance(output, dict) and "parsed" in output:
        if output.get("parsing_error") is not None:
            raise output["parsing_error"]
        if output.get("parsed") is None:
            raise ValueError(f"the model returned no structured {schema.__name__}")
        raw = output.get("raw")
        model = raw.response_metadata.get("model_name") if raw is not None else None
        parsed = output["parsed"]
        return (parsed if isinstance(parsed, schema) else schema.model_validate(parsed)), model
    return schema.model_validate(output), None


# --------------------------------------------------------------------------- one agent turn
def agent_step(model: Runnable, system_prompt: str, messages: Sequence[BaseMessage], budget: int) -> Dict[str, Any]:
    """Ask the model for its next move. Never raises: a spent budget or a dead model becomes a state flag."""
    if count_tool_rounds(messages) >= budget:
        note = f"Tool budget of {budget} rounds reached; moving on with the evidence gathered"
        logger.info(note)
        return {"budget_exhausted": True, "events": [note]}
    try:
        reply = model.invoke([SystemMessage(content=system_prompt), *messages])
    except Exception as exc:  # noqa: BLE001 - every provider failed; degrade instead of crashing
        note = f"Agent LLM failed on every provider ({safe_error(exc, limit=600)})"
        logger.error(note)
        return {"llm_error": note, "events": [note]}
    return {"messages": [reply], "budget_exhausted": False, "llm_error": None}


def wants_tools(state: Dict[str, Any]) -> bool:
    """True when the last turn asked for tools and the loop may still run them."""
    if state.get("llm_error") or state.get("budget_exhausted"):
        return False
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    return isinstance(last, AIMessage) and bool(last.tool_calls)


def tool_node(tools: Sequence[BaseTool]) -> ToolNode:
    """The only tools this loop can execute. A call to any other name comes back as an error message."""
    # handle_tool_errors also catches arguments that fail a tool's schema.
    return ToolNode(list(tools), handle_tool_errors=True)


# --------------------------------------------------------------------------- the loop
def build_react_loop(model: Runnable, tools: Sequence[BaseTool], system_prompt: str, max_rounds: int):
    """Compile a standalone agent <-> tools loop with a fixed tool-round budget."""

    def agent_node(state: LoopState) -> Dict[str, Any]:
        return agent_step(model, system_prompt, state["messages"], max_rounds)

    graph = StateGraph(LoopState)
    graph.add_node(NODE_AGENT, agent_node)
    graph.add_node(NODE_TOOLS, tool_node(tools))
    graph.add_edge(START, NODE_AGENT)
    graph.add_conditional_edges(NODE_AGENT, lambda s: NODE_TOOLS if wants_tools(s) else END, [NODE_TOOLS, END])
    graph.add_edge(NODE_TOOLS, NODE_AGENT)
    return graph.compile()


def run_react_loop(
    loop: Any,
    messages: Sequence[BaseMessage],
    on_message: Optional[Callable[[BaseMessage], None]] = None,
    recursion_limit: int = config.RECURSION_LIMIT,
) -> Dict[str, Any]:
    """Run a compiled loop from `messages`. Returns the final state; never raises.

    `on_message` sees every message the loop adds, as it happens.
    """
    state: Dict[str, Any] = {"messages": list(messages), "events": []}
    # The messages passed in (the task, or an earlier conversation) are not re-printed; only new ones are.
    seen = len(messages)
    try:
        for state in loop.stream(state, config={"recursion_limit": recursion_limit}, stream_mode="values"):
            current = state.get("messages", [])
            if on_message is not None:
                for message in current[seen:]:
                    on_message(message)
            seen = len(current)
    except Exception as exc:  # noqa: BLE001 - keep the partial history and report the failure
        note = f"Agent loop stopped: {safe_error(exc)}"
        logger.error(note)
        # Return the last good state with the failure recorded, so the caller still gets the partial
        # conversation (and any tool results) instead of an exception.
        state = {**state, "llm_error": state.get("llm_error") or note, "events": [*state.get("events", []), note]}
    return state
