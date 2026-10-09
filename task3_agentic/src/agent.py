"""Task 3A research agent: a LangGraph ReAct loop with a report check that can send it back.

    START -> agent --(tool calls)--> tools -> agent            the model picks every tool call
               |
               +--(no tool calls / budget spent / LLM down)--> write_report -> check_report
                                                                   |-- passes ---------------> END
                                                                   |-- gaps, revisions left -> agent (replan)
                                                                   +-- gaps, none left ------> END (with warnings)

Nothing here fixes the order of tool calls. The agent node only asks the model what to
do next given the conversation so far, and routing follows the model's tool calls.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16): loop moved to react.py
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18): checkpointer thread and follow-up mode

import operator
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import config, prompts
from .llm import build_chat_models, with_structure, with_tools
from .logging_utils import get_logger
from .react import (  # noqa: F401 - count_tool_rounds, last_agent_notes, models_used are re-exported
    agent_step,
    count_tool_rounds,
    last_agent_notes,
    models_used,
    tool_node,
    unwrap_structured,
    wants_tools,
)
from .report import build_evidence_digest, check_report
from .schemas import FinalReport, ResearchReport
from .session import SessionContext
from .tools import build_tools
from .tracing import safe_error

logger = get_logger("agent")

NODE_AGENT = "agent"
NODE_TOOLS = "tools"
NODE_WRITE_REPORT = "write_report"
NODE_CHECK_REPORT = "check_report"


class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    revisions: int
    report: Optional[ResearchReport]
    report_model: Optional[str]
    report_error: Optional[str]
    report_issues: List[str]
    check_passed: bool
    send_back: bool
    budget_exhausted: bool
    llm_error: Optional[str]
    # Graph-level events (budget hit, LLM failure, replan) for the visible trace.
    events: Annotated[List[str], operator.add]
    # 3C short-term memory: "research" runs the full loop; "followup" answers a question on the same
    # thread and ends without writing a new report. followup_base_rounds = tool rounds before the question.
    mode: str
    followup_base_rounds: int


def default_runnables(tools: Sequence[BaseTool]) -> Dict[str, Runnable]:
    """Groq-first models: one with the tools bound (agent), one with the report schema (writer)."""
    # One skip set for both chains: a model that hit its daily limit is not retried by either.
    skipped: set = set()
    return {
        "agent": with_tools(build_chat_models(config.AGENT_MAX_TOKENS), tools, skipped),
        "report": with_structure(
            build_chat_models(config.REPORT_MAX_TOKENS, config.REPORT_REASONING_EFFORT), ResearchReport, skipped
        ),
    }


def unwrap_report(output: Any) -> Tuple[ResearchReport, Optional[str]]:
    """Structured-output result -> (ResearchReport, model name). Raises when it did not parse."""
    return unwrap_structured(output, ResearchReport)


def build_research_graph(
    ctx: SessionContext,
    agent_model: Runnable,
    report_model: Runnable,
    tools: Sequence[BaseTool],
    checkpointer: Optional[Any] = None,
):
    """Compile the graph. Models are passed in, so tests can use scripted fakes.

    With a checkpointer, every step is saved under the run's thread_id, so a later follow-up
    question on the same thread sees the whole conversation, tool results included (3C).
    """
    system_prompt = prompts.AGENT_SYSTEM.format(
        max_rounds=config.MAX_TOOL_ROUNDS, today=date.today().isoformat(), ticker=ctx.ticker
    )

    def followup_node(state: AgentState) -> Dict[str, Any]:
        """A follow-up may run FOLLOWUP_TOOL_ROUNDS tool rounds (memory comes first), then gets one turn to answer."""
        messages = state["messages"]
        tool_budget = state.get("followup_base_rounds", 0) + config.FOLLOWUP_TOOL_ROUNDS
        rounds = count_tool_rounds(messages)
        if rounds < tool_budget:
            return agent_step(agent_model, system_prompt, messages, tool_budget)
        # Tool budget spent: one last turn to answer. A tool request here is dropped, not added to the
        # thread, because an unanswered tool call would break the conversation for later turns.
        out = agent_step(agent_model, system_prompt + prompts.FOLLOWUP_ANSWER_NOW, messages, rounds + 1)
        replies = out.get("messages") or []
        if replies and isinstance(replies[-1], AIMessage) and replies[-1].tool_calls:
            note = "Follow-up tool budget spent and the agent asked for another tool; stopping without an answer"
            logger.info(note)
            return {"budget_exhausted": True, "events": [note]}
        return out

    def agent_node(state: AgentState) -> Dict[str, Any]:
        if state.get("mode") == config.MODE_FOLLOWUP:
            return followup_node(state)
        # Each revision the report check asks for buys the agent a few more tool rounds.
        budget = config.MAX_TOOL_ROUNDS + state.get("revisions", 0) * config.REVISION_TOOL_ROUNDS
        return agent_step(agent_model, system_prompt, state["messages"], budget)

    def route_after_agent(state: AgentState) -> str:
        if wants_tools(state):
            return NODE_TOOLS
        # A follow-up answer is the end of that turn; only a research run goes on to the report.
        return END if state.get("mode") == config.MODE_FOLLOWUP else NODE_WRITE_REPORT

    def write_report_node(state: AgentState) -> Dict[str, Any]:
        issues = state.get("report_issues") or []
        previous = (
            prompts.REPORT_PREVIOUS_ISSUES.format(issues="\n".join(f"- {i}" for i in issues)) if issues else ""
        )
        # The report writer is a separate, fresh call. It gets a compact digest of the session's successful
        # tool results instead of the whole agent conversation: smaller (fits the token budget) and
        # limited to facts the check can verify. messages[0] is always the original research question.
        request: List[BaseMessage] = [
            SystemMessage(content=prompts.REPORT_SYSTEM),
            HumanMessage(
                content=prompts.REPORT_USER.format(
                    query=state["messages"][0].content,
                    ticker=ctx.ticker,
                    notes=last_agent_notes(state["messages"]),
                    digest=build_evidence_digest(ctx),
                    previous_issues=previous,
                )
            ),
        ]
        error: Optional[str] = None
        for _attempt in range(1 + config.REPORT_REPAIR_ATTEMPTS):
            try:
                report, model = unwrap_report(report_model.invoke(request))
                return {"report": report, "report_model": model, "report_error": None}
            except Exception as exc:  # noqa: BLE001 - validation or provider failure; one repair, then degrade
                error = safe_error(exc)
                logger.warning("Report writer failed: %s", error)
                request = [*request, HumanMessage(content=prompts.REPORT_REPAIR.format(error=error))]
        return {"report": None, "report_error": error}

    def check_report_node(state: AgentState) -> Dict[str, Any]:
        report = state.get("report")
        if report is None:
            issues = [f"The report could not be produced ({state.get('report_error')})."]
        else:
            issues = check_report(report, ctx)
        if not issues:
            return {"report_issues": [], "check_passed": True, "send_back": False}
        revisions = state.get("revisions", 0)
        # Sending the agent back only helps if its model is still reachable.
        if revisions < config.MAX_REPORT_REVISIONS and not state.get("llm_error"):
            # This is the replan path: the failed checks go into the agent's own conversation as a new
            # user message, and route_after_check sends control back to the agent node, which can now
            # call more tools to fill the gaps.
            feedback = prompts.REPORT_FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))
            note = f"Report check failed ({len(issues)} issue(s)); sending the agent back (revision {revisions + 1})"
            logger.info(note)
            return {
                "messages": [HumanMessage(content=feedback)],
                "report_issues": issues,
                "revisions": revisions + 1,
                "check_passed": False,
                "send_back": True,
                "events": [note],
            }
        return {"report_issues": issues, "check_passed": False, "send_back": False}

    def route_after_check(state: AgentState) -> str:
        return NODE_AGENT if state.get("send_back") else END

    # Wiring: plain edges always go to the next node; conditional edges call the route function, whose
    # return value names the next node (the list gives the possible targets, for the graph drawing).
    graph = StateGraph(AgentState)
    graph.add_node(NODE_AGENT, agent_node)
    graph.add_node(NODE_TOOLS, tool_node(tools))
    graph.add_node(NODE_WRITE_REPORT, write_report_node)
    graph.add_node(NODE_CHECK_REPORT, check_report_node)
    graph.add_edge(START, NODE_AGENT)
    graph.add_conditional_edges(NODE_AGENT, route_after_agent, [NODE_TOOLS, NODE_WRITE_REPORT, END])
    graph.add_edge(NODE_TOOLS, NODE_AGENT)
    graph.add_edge(NODE_WRITE_REPORT, NODE_CHECK_REPORT)
    graph.add_conditional_edges(NODE_CHECK_REPORT, route_after_check, [NODE_AGENT, END])
    return graph.compile(checkpointer=checkpointer)


@dataclass
class AgentRun:
    """Everything a notebook cell needs after one research run."""

    final: FinalReport
    messages: List[BaseMessage]
    ctx: SessionContext
    events: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    # 3C: the compiled graph and its checkpointer thread, so a follow-up can continue the conversation.
    graph: Any = None
    thread_id: Optional[str] = None

    @property
    def tool_sequence(self) -> List[str]:
        return self.ctx.tool_sequence()


def run_research(
    ctx: SessionContext,
    query: Optional[str] = None,
    agent_model: Optional[Runnable] = None,
    report_model: Optional[Runnable] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    on_message: Optional[Callable[[BaseMessage], None]] = None,
    recursion_limit: int = config.RECURSION_LIMIT,
    checkpointer: Optional[Any] = None,
) -> AgentRun:
    """Run the agent to completion. Never raises: failures come back as a FinalReport with status 'failed'.

    `on_message` is called for every new message as it happens (the notebook prints the live trace).
    The conversation is saved in `checkpointer` (an InMemorySaver by default) under thread_id =
    session id, which is what memory.ask_followup continues.
    """
    query = query or config.RESEARCH_QUERY_TEMPLATE.format(ticker=ctx.ticker)
    started = time.perf_counter()
    state: Dict[str, Any] = {"messages": [HumanMessage(content=prompts.AGENT_USER.format(query=query))]}
    seen = 0
    error: Optional[str] = None
    graph = None
    thread_id = ctx.session_id
    try:
        tools = list(tools) if tools is not None else build_tools(ctx)
        if agent_model is None or report_model is None:
            defaults = default_runnables(tools)
            agent_model = agent_model or defaults["agent"]
            report_model = report_model or defaults["report"]
        graph = build_research_graph(ctx, agent_model, report_model, tools, checkpointer or InMemorySaver())
        initial = {**state, "revisions": 0, "events": [], "mode": config.MODE_RESEARCH}
        # recursion_limit caps the number of graph steps (a hard stop if routing ever loops);
        # thread_id tells the checkpointer which conversation to save this run under.
        run_config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": thread_id}}
        # stream_mode="values" yields the full state after every step, so a crash mid-run keeps the partial trace.
        for state in graph.stream(initial, config=run_config, stream_mode="values"):
            messages = state.get("messages", [])
            # `seen` counts messages already printed; only the new ones since the last step are sent to the printer.
            if on_message is not None:
                for message in messages[seen:]:
                    on_message(message)
            seen = len(messages)
    except Exception as exc:  # noqa: BLE001 - the notebook must always get a result object
        error = safe_error(exc)
        logger.error("Research run failed: %s", error)

    report = state.get("report")
    issues = state.get("report_issues") or []
    if report is None:
        status = config.REPORT_STATUS_FAILED
    elif state.get("check_passed"):
        status = config.REPORT_STATUS_VALIDATED
    else:
        status = config.REPORT_STATUS_UNVALIDATED
    final = FinalReport(
        ticker=ctx.ticker,
        query=query,
        session_id=ctx.session_id,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status=status,
        report=report,
        validation_warnings=issues if status != config.REPORT_STATUS_VALIDATED else [],
        tools_used=sorted(ctx.succeeded_tools()),
        revisions=state.get("revisions", 0),
        agent_models=models_used(state.get("messages", [])),
        report_model=state.get("report_model"),
        error=error or state.get("report_error") or state.get("llm_error"),
    )
    return AgentRun(
        final=final,
        messages=list(state.get("messages", [])),
        ctx=ctx,
        events=list(state.get("events", [])),
        duration_s=round(time.perf_counter() - started, 1),
        graph=graph,
        thread_id=thread_id,
    )
