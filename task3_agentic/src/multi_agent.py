"""Task 3B: a two-agent research pipeline with structured handoffs and one critique round.

    START -> analyst_initial   A (data_analyst): ReAct over get_price_data / calculate_volatility / llm_sentiment
                               -> DataBrief (numbers from A's tool results, interpretation from A's LLM)
          -> writer_research   B (research_writer): ReAct over web_search / get_news
                               -> exactly one ClarificationRequest, chosen by B's LLM
          -> analyst_clarify   A answers in its own conversation, with its own tools -> ClarificationResponse
          -> writer_final      B: ResearchReport from the brief, the answer and B's evidence,
                               checked by check_report + check_incorporation (one repair pass) -> END

Tool access is enforced in three layers, not by prompt:
  1. each agent's LLM is bound only to its own tool schemas (bind_tools);
  2. each agent's ToolNode holds only its own tools, so a forged call to another tool fails;
  3. each agent's ResearchTools has an `allowed` set checked in run_traced, so even code that
     calls the toolkit directly is refused, and the attempt is logged in agent_trace.jsonl.

Agent A has llm_sentiment but no news access. The headlines it scores arrive in Agent B's
clarification request: the agents need each other to produce the sentiment evidence.
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)

import json
import operator
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, Type, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from . import config, prompts
from .handoff import (
    AgentMessage,
    AnalystInterpretation,
    AnalystTask,
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    build_brief_numbers,
    build_response,
    check_incorporation,
    handoffs_to_json,
    merge_interpretation,
)
from .llm import build_chat_models, with_structure, with_tools
from .logging_utils import get_logger
from .react import build_react_loop, count_tool_rounds, last_agent_notes, models_used, run_react_loop, unwrap_structured
from .report import build_writer_digest, check_report
from .schemas import FinalReport, ResearchReport
from .session import SessionContext
from .tools import ResearchTools, build_tools
from .tracing import safe_error

logger = get_logger("multi_agent")

A = config.AGENT_A_NAME
B = config.AGENT_B_NAME
ORCHESTRATOR = config.ORCHESTRATOR_NAME

NODE_ANALYST_INITIAL = "analyst_initial"
NODE_WRITER_RESEARCH = "writer_research"
NODE_ANALYST_CLARIFY = "analyst_clarify"
NODE_WRITER_FINAL = "writer_final"


class PipelineState(TypedDict, total=False):
    # Separate conversations: each agent only ever sees its own history and the handoffs sent to it.
    analyst_messages: List[BaseMessage]
    writer_messages: List[BaseMessage]
    brief: Optional[DataBrief]
    request: Optional[ClarificationRequest]
    response: Optional[ClarificationResponse]
    report: Optional[ResearchReport]
    report_model: Optional[str]
    report_issues: List[str]
    report_error: Optional[str]
    report_revisions: int
    check_passed: bool
    handoffs: Annotated[List[AgentMessage], operator.add]
    events: Annotated[List[str], operator.add]


@dataclass
class PipelineModels:
    """The five model roles. Tests pass scripted fakes; default_pipeline_models builds the Groq-first chains."""

    analyst_agent: Runnable  # tool-calling, A's tools bound
    writer_agent: Runnable  # tool-calling, B's tools bound
    analyst_brief: Runnable  # structured output -> AnalystInterpretation
    writer_request: Runnable  # structured output -> ClarificationRequest
    writer_report: Runnable  # structured output -> ResearchReport


class PipelineObserver:
    """Receives everything that happens, as it happens. The notebook uses display.MultiAgentPrinter."""

    def on_message(self, agent: str, message: BaseMessage) -> None:  # an agent's LLM turn or tool result
        pass

    def on_handoff(self, message: AgentMessage) -> None:  # an inter-agent message
        pass

    def on_event(self, text: str) -> None:  # budget hit, model failure, fallback, repair pass
        pass


@dataclass
class MultiAgentRun:
    final: FinalReport
    ctx: SessionContext
    handoffs: List[AgentMessage]
    brief: Optional[DataBrief]
    request: Optional[ClarificationRequest]
    response: Optional[ClarificationResponse]
    analyst_messages: List[BaseMessage]
    writer_messages: List[BaseMessage]
    bound_tools: Dict[str, List[str]]
    events: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def tool_sequence(self, agent: Optional[str] = None) -> List[str]:
        return self.ctx.tool_sequence(agent)

    @property
    def critique_rounds(self) -> int:
        return sum(1 for h in self.handoffs if h.kind == "clarification_request")


def default_pipeline_models(tools_a: Sequence[BaseTool], tools_b: Sequence[BaseTool]) -> PipelineModels:
    """Same FallbackChain for both agents; one skip set, so a model at its daily limit is skipped by every role."""
    skipped: set = set()
    agent_models = build_chat_models(config.AGENT_MAX_TOKENS, config.AGENT_REASONING_EFFORT)
    writer_models = build_chat_models(config.REPORT_MAX_TOKENS, config.REPORT_REASONING_EFFORT)
    return PipelineModels(
        analyst_agent=with_tools(agent_models, tools_a, skipped),
        writer_agent=with_tools(agent_models, tools_b, skipped),
        analyst_brief=with_structure(writer_models, AnalystInterpretation, skipped),
        writer_request=with_structure(writer_models, ClarificationRequest, skipped),
        writer_report=with_structure(writer_models, ResearchReport, skipped),
    )


def call_structured(
    model: Runnable,
    messages: List[BaseMessage],
    schema: Type[BaseModel],
    repair_template: str,
    attempts: int,
) -> Tuple[Optional[BaseModel], Optional[str], Optional[str]]:
    """Invoke a structured-output model; on a validation or provider error send the error back and retry.

    Returns (instance or None, model name, last error). Never raises.
    """
    error: Optional[str] = None
    for _ in range(attempts):
        try:
            # unwrap_structured raises if the model's output did not parse into `schema`.
            instance, model_name = unwrap_structured(model.invoke(messages), schema)
            return instance, model_name, None
        except Exception as exc:  # noqa: BLE001 - repair once, then let the caller fall back
            error = safe_error(exc)
            logger.warning("%s step failed: %s", schema.__name__, error)
            # Next attempt sees the original request plus the error, so the model can correct itself.
            # A new list is built each time (not .append) so the caller's list is never modified.
            messages = [*messages, HumanMessage(content=repair_template.format(error=error))]
    return None, None, error


def _json(model: BaseModel) -> str:
    return model.model_dump_json(exclude_none=True)


def _last(handoffs: Sequence[AgentMessage], kind: str) -> Optional[AgentMessage]:
    # Each node reads its input from the handoff log (newest message of that kind), never from another
    # agent's state directly: every piece of data an agent receives came through a validated message.
    for message in reversed(handoffs):
        if message.kind == kind:
            return message
    return None


def writer_evidence(ctx: SessionContext) -> Dict[str, Any]:
    """B's own findings, compact: headline titles and search hits (what B can offer A, or cite)."""
    evidence: Dict[str, Any] = {}
    news = ctx.results_for("get_news", agent=B)
    if news:
        evidence["headlines"] = [h.title for h in news[0].result.data.headlines]
    # Flatten the hits of B's latest few searches into one list (nested comprehension: records, then hits).
    hits = [hit for record in ctx.results_for("web_search", agent=B)[: config.DIGEST_RESULTS_PER_TOOL] for hit in record.result.data.hits]
    if hits:
        evidence["web_results"] = [{"title": h.title, "snippet": h.snippet, "url": h.url} for h in hits]
    return evidence


def fallback_request(ctx: SessionContext, brief: Optional[DataBrief]) -> ClarificationRequest:
    """Used only if B's LLM cannot produce a valid request: ask A to score B's headlines, or for a second
    volatility window (one the brief does not already have)."""
    evidence = writer_evidence(ctx)
    titles = evidence.get("headlines") or [hit["title"] for hit in evidence.get("web_results", [])]
    if titles:
        return ClarificationRequest(
            request_kind="score_headlines",
            question="What is the confidence-weighted sentiment score of these recent headlines?",
            why_needed="The brief has no sentiment score, and the market-sentiment section needs one.",
            headlines=titles[: config.MAX_SENTIMENT_HEADLINES],
        )
    # No headlines to offer: ask for a volatility window the brief does not have yet (first of 20/60/252
    # that differs from the brief's), so the answer adds information.
    have = brief.volatility.window if brief and brief.volatility else None
    window = next(w for w in config.VOL_COMPARISON_WINDOWS if w != have)
    return ClarificationRequest(
        request_kind="volatility_window",
        question=f"What is the annualised volatility over the last {window} trading days?",
        why_needed="A second window shows whether the current volatility regime is stable before sizing the hedge.",
        window=window,
    )


def fulfil_for_analyst(toolkit: ResearchTools, request: ClarificationRequest) -> None:
    """If A's LLM did not run the tool the request needs, run it with A's own toolkit (same access rules)."""
    ticker = toolkit.ctx.ticker
    if request.request_kind == "score_headlines":
        toolkit.llm_sentiment(request.headlines)
    elif request.request_kind == "volatility_window":
        toolkit.calculate_volatility(ticker, request.window)
    elif request.request_kind == "price_period":
        toolkit.get_price_data(ticker, request.period)
    else:
        toolkit.get_price_data(ticker, config.DEFAULT_PERIOD)


def _brief_summary(brief: DataBrief) -> str:
    parts = []
    if brief.price:
        parts.append(f"price {brief.price.current_price}, {brief.price.period} return {brief.price.period_return_pct}%")
    if brief.volatility:
        parts.append(f"{brief.volatility.window}d vol {brief.volatility.annualised_vol_pct}% ({brief.volatility.regime})")
    parts.append("sentiment " + (f"{brief.sentiment.score}" if brief.sentiment else "not available (no news access)"))
    return "; ".join(parts) + f"; {len(brief.key_findings)} findings, {len(brief.quant_risk_flags)} risk flags"


def _response_summary(response: ClarificationResponse) -> str:
    if response.sentiment is not None:
        value = f"sentiment {response.sentiment.score} ({response.sentiment.label}) from {response.sentiment.n_scored} headlines"
    elif response.volatility is not None:
        value = f"{response.volatility.window}d vol {response.volatility.annualised_vol_pct}%"
    elif response.period_return is not None:
        value = f"{response.period_return.period} return {response.period_return.period_return_pct}%"
    elif response.metric_value is not None:
        value = f"{response.metric_name} = {response.metric_value}"
    else:
        value = response.error or "no data"
    return f"{response.status}: {value} (fulfilled by {response.fulfilled_by})"


# --------------------------------------------------------------------------- the graph
def build_pipeline_graph(
    ctx: SessionContext,
    models: PipelineModels,
    toolkit_a: ResearchTools,
    tools_a: Sequence[BaseTool],
    tools_b: Sequence[BaseTool],
    query: str,
    observer: PipelineObserver,
):
    today = date.today().isoformat()
    analyst_system = prompts.ANALYST_SYSTEM.format(today=today, ticker=ctx.ticker)
    writer_system = prompts.WRITER_SYSTEM.format(today=today, ticker=ctx.ticker)

    # The four node functions below are closures: they share ctx, models, toolkit_a, query and observer
    # from this enclosing function, so the graph state only carries what changes between steps.

    def emit(message: AgentMessage) -> AgentMessage:
        # Show the handoff live (notebook printer), then return it so the caller can store it in the state.
        observer.on_handoff(message)
        return message

    def note(text: str) -> str:
        observer.on_event(text)
        return text

    def analyst_initial(state: PipelineState) -> Dict[str, Any]:
        task = AnalystTask(
            ticker=ctx.ticker,
            query=query,
            focus="Quantitative evidence: trend, indicators, fundamentals, and volatility for sizing a 90-day hedge.",
            tools=list(config.AGENT_A_TOOLS),
        )
        task_message = emit(AgentMessage.wrap(ORCHESTRATOR, A, "task", task, summary=f"Research {ctx.ticker}: quantitative part"))
        # The receiving side re-validates the payload, exactly as it would for a message from another process.
        received = task_message.open(AnalystTask)
        history = [HumanMessage(content=prompts.ANALYST_TASK.format(task_json=_json(received)))]
        # A's own ReAct loop with only A's three tools; its model decides which to call and when to stop.
        loop = build_react_loop(models.analyst_agent, tools_a, analyst_system, config.ANALYST_MAX_ROUNDS)
        # on_message prints each of A's turns live, labelled with A's name.
        out = run_react_loop(loop, history, on_message=lambda m: observer.on_message(A, m))
        events = [note(f"{A}: {e}") for e in out.get("events", [])]

        # The brief's numbers come straight from A's successful tool results (code, not the LLM), then
        # A's LLM writes only the interpretation of those numbers.
        numbers = build_brief_numbers(ctx, A)
        interpretation, model_name, error = call_structured(
            models.analyst_brief,
            [
                SystemMessage(content=prompts.ANALYST_BRIEF_SYSTEM),
                HumanMessage(
                    content=prompts.ANALYST_BRIEF_USER.format(brief_json=_json(numbers), notes=last_agent_notes(out["messages"]))
                ),
            ],
            AnalystInterpretation,
            prompts.STRUCTURED_REPAIR,
            1 + config.REQUEST_REPAIR_ATTEMPTS,
        )
        if error:
            events.append(note(f"{A}: brief interpretation failed ({error}); sending numbers only"))
        brief = merge_interpretation(numbers, interpretation, model_name)
        brief_message = emit(AgentMessage.wrap(A, B, "data_brief", brief, summary=_brief_summary(brief)))
        # The returned dict is merged into the graph state. "handoffs" and "events" have an add reducer,
        # so these lists are appended to the log; the other keys simply overwrite.
        return {
            "analyst_messages": out["messages"],
            "brief": brief,
            "handoffs": [task_message, brief_message],
            "events": events,
        }

    def writer_research(state: PipelineState) -> Dict[str, Any]:
        brief = _last(state["handoffs"], "data_brief").open(DataBrief)
        # B starts a fresh conversation of its own: the query plus A's brief. It never sees A's messages.
        history = [HumanMessage(content=prompts.WRITER_TASK.format(query=query, brief_json=_json(brief)))]
        loop = build_react_loop(models.writer_agent, tools_b, writer_system, config.WRITER_MAX_ROUNDS)
        out = run_react_loop(loop, history, on_message=lambda m: observer.on_message(B, m))
        events = [note(f"{B}: {e}") for e in out.get("events", [])]

        # The critique step: B's LLM must produce exactly one ClarificationRequest (structured output),
        # choosing what to ask from the gaps it sees in the brief and its own research.
        request, _, error = call_structured(
            models.writer_request,
            [
                SystemMessage(content=prompts.WRITER_REQUEST_SYSTEM),
                HumanMessage(
                    content=prompts.WRITER_REQUEST_USER.format(
                        brief_json=_json(brief),
                        evidence_json=json.dumps(writer_evidence(ctx), ensure_ascii=False),
                        notes=last_agent_notes(out["messages"]),
                    )
                ),
            ],
            ClarificationRequest,
            prompts.REQUEST_REPAIR,
            1 + config.REQUEST_REPAIR_ATTEMPTS,
        )
        fallback = request is None
        if fallback:
            request = fallback_request(ctx, brief)
            events.append(note(f"{B}: request step failed ({error}); sending the fallback request"))
        message = emit(
            AgentMessage.wrap(B, A, "clarification_request", request, summary=f"{request.request_kind}: {request.question}", fallback=fallback)
        )
        return {"writer_messages": out["messages"], "request": request, "handoffs": [message], "events": events}

    def analyst_clarify(state: PipelineState) -> Dict[str, Any]:
        request = _last(state["handoffs"], "clarification_request").open(ClarificationRequest)
        # Remember where the tool history stands now: only tool calls made *after* this point count as
        # A's answer to the request (build_response(..., since) looks only at those).
        since = ctx.history_length()
        # A continues its own earlier conversation, so it still has its first-phase tool results in context.
        previous = list(state.get("analyst_messages") or [])
        history = [*previous, HumanMessage(content=prompts.ANALYST_CLARIFY.format(request_json=_json(request)))]
        # The budget counts A's earlier rounds too, so this phase gets ANALYST_CLARIFY_ROUNDS new ones.
        budget = count_tool_rounds(previous) + config.ANALYST_CLARIFY_ROUNDS
        loop = build_react_loop(models.analyst_agent, tools_a, analyst_system, budget)
        out = run_react_loop(loop, history, on_message=lambda m: observer.on_message(A, m))
        events = [note(f"{A}: {e}") for e in out.get("events", [])]

        # A's text answer is its last reply in this phase (messages after the request); the typed values
        # in the response are filled from its new tool results, not from that text.
        answer = last_agent_notes(out["messages"][len(history):], config.CLARIFICATION_ANSWER_MAX_CHARS)
        response = build_response(ctx, request, answer, since, A)
        if response.status == "failed":
            # A's LLM did not run the needed tool (or its model is down). The data still comes from A's toolkit.
            events.append(note(f"{A}: did not produce the requested data; running its tool on its behalf"))
            fulfil_for_analyst(toolkit_a, request)
            response = build_response(
                ctx, request, f"(Orchestrator ran the tool with Agent A's toolkit.) {answer}", since, A, fulfilled_by="fallback"
            )
        message = emit(AgentMessage.wrap(A, B, "clarification_response", response, summary=_response_summary(response)))
        return {"analyst_messages": out["messages"], "response": response, "handoffs": [message], "events": events}

    def writer_final(state: PipelineState) -> Dict[str, Any]:
        handoffs = state["handoffs"]
        brief = _last(handoffs, "data_brief").open(DataBrief)
        request = _last(handoffs, "clarification_request").open(ClarificationRequest)
        response = _last(handoffs, "clarification_response").open(ClarificationResponse)
        digest = build_writer_digest(ctx, brief, request, response, B)
        notes = last_agent_notes(state.get("writer_messages") or [])
        events: List[str] = []
        report: Optional[ResearchReport] = None
        model_name: Optional[str] = None
        error: Optional[str] = None
        issues: List[str] = []
        revisions = 0
        # Write -> check -> (if issues) write again with the issues listed in the prompt. Two separate repair
        # mechanisms are in play: call_structured repairs invalid JSON, this loop repairs a valid report
        # that fails the content checks (evidence, hedge numbers, use of A's answer).
        for attempt in range(1 + config.FINAL_REPORT_REPAIR_ATTEMPTS):
            previous = prompts.REPORT_PREVIOUS_ISSUES.format(issues="\n".join(f"- {i}" for i in issues)) if issues else ""
            candidate, candidate_model, error = call_structured(
                models.writer_report,
                [
                    SystemMessage(content=prompts.WRITER_REPORT_SYSTEM),
                    HumanMessage(
                        content=prompts.WRITER_REPORT_USER.format(
                            query=query, ticker=ctx.ticker, digest=digest, notes=notes, previous_issues=previous
                        )
                    ),
                ],
                ResearchReport,
                prompts.REPORT_REPAIR,
                1 + config.REPORT_REPAIR_ATTEMPTS,
            )
            if candidate is None:
                # No usable report this round. If an earlier round produced one, keep it (with its issues).
                if report is None:
                    issues = [f"The report could not be produced ({error})."]
                break
            report, model_name = candidate, candidate_model
            # 3A's evidence checks plus the 3B check that B actually used the value A sent back.
            issues = check_report(report, ctx) + check_incorporation(report, response)
            if not issues:
                break
            if attempt < config.FINAL_REPORT_REPAIR_ATTEMPTS:
                revisions += 1
                events.append(note(f"{B}: final report check found {len(issues)} issue(s); repair pass {revisions}"))
        update: Dict[str, Any] = {
            "report": report,
            "report_model": model_name,
            "report_issues": issues,
            "report_error": error if report is None else None,
            "report_revisions": revisions,
            "check_passed": report is not None and not issues,
            "events": events,
        }
        if report is not None:
            summary = f"report on {report.ticker}: " + "; ".join(r.name for r in report.risks)
            update["handoffs"] = [emit(AgentMessage.wrap(B, ORCHESTRATOR, "final_report", report, summary=summary))]
        return update

    graph = StateGraph(PipelineState)
    graph.add_node(NODE_ANALYST_INITIAL, analyst_initial)
    graph.add_node(NODE_WRITER_RESEARCH, writer_research)
    graph.add_node(NODE_ANALYST_CLARIFY, analyst_clarify)
    graph.add_node(NODE_WRITER_FINAL, writer_final)
    graph.add_edge(START, NODE_ANALYST_INITIAL)
    graph.add_edge(NODE_ANALYST_INITIAL, NODE_WRITER_RESEARCH)
    graph.add_edge(NODE_WRITER_RESEARCH, NODE_ANALYST_CLARIFY)
    graph.add_edge(NODE_ANALYST_CLARIFY, NODE_WRITER_FINAL)
    graph.add_edge(NODE_WRITER_FINAL, END)
    return graph.compile()


def build_agent_toolkits(ctx: SessionContext) -> Dict[str, Tuple[ResearchTools, List[BaseTool]]]:
    """Each agent's toolkit (layer 3: `allowed`) and its LangChain tools (layers 1 and 2: only its own)."""
    toolkits = {}
    for agent, names in ((A, config.AGENT_A_TOOLS), (B, config.AGENT_B_TOOLS)):
        toolkit = ResearchTools(ctx, agent_name=agent, allowed=names)
        toolkits[agent] = (toolkit, build_tools(toolkit, names))
    return toolkits


def run_multi_agent(
    ctx: SessionContext,
    query: Optional[str] = None,
    models: Optional[PipelineModels] = None,
    observer: Optional[PipelineObserver] = None,
) -> MultiAgentRun:
    """Run the whole pipeline: query in, report out, no manual step. Never raises."""
    query = query or config.RESEARCH_QUERY_TEMPLATE.format(ticker=ctx.ticker)
    observer = observer or PipelineObserver()
    started = time.perf_counter()
    toolkits = build_agent_toolkits(ctx)
    toolkit_a, tools_a = toolkits[A]
    _, tools_b = toolkits[B]
    # Recorded for the notebook: the exact tool names each agent's model was bound to (enforcement layer 1).
    bound = {A: [t.name for t in tools_a], B: [t.name for t in tools_b]}
    state: Dict[str, Any] = {"handoffs": [], "events": []}
    error: Optional[str] = None
    try:
        models = models or default_pipeline_models(tools_a, tools_b)
        graph = build_pipeline_graph(ctx, models, toolkit_a, tools_a, tools_b, query, observer)
        # stream(..., "values") yields the full state after every node. Keeping only the latest one means
        # that if a later node crashes, `state` still holds everything produced up to that point.
        for state in graph.stream({"handoffs": [], "events": []}, stream_mode="values"):
            pass
    except Exception as exc:  # noqa: BLE001 - the notebook must always get a result object
        error = safe_error(exc)
        logger.error("Pipeline failed: %s", error)
        observer.on_event(f"pipeline stopped: {error}")

    report = state.get("report")
    if report is None:
        status = config.MULTI_AGENT_STATUS_FAILED
    elif state.get("check_passed"):
        status = config.REPORT_STATUS_VALIDATED
    else:
        status = config.REPORT_STATUS_UNVALIDATED
    analyst_messages = list(state.get("analyst_messages") or [])
    writer_messages = list(state.get("writer_messages") or [])
    agent_models = models_used(analyst_messages)
    for name, count in models_used(writer_messages).items():
        agent_models[name] = agent_models.get(name, 0) + count
    final = FinalReport(
        ticker=ctx.ticker,
        query=query,
        session_id=ctx.session_id,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status=status,
        report=report,
        validation_warnings=(state.get("report_issues") or []) if status != config.REPORT_STATUS_VALIDATED else [],
        tools_used=sorted(ctx.succeeded_tools()),
        revisions=state.get("report_revisions", 0),
        agent_models=agent_models,
        report_model=state.get("report_model"),
        agent_tools={A: sorted(ctx.succeeded_tools(A)), B: sorted(ctx.succeeded_tools(B))},
        error=error or state.get("report_error"),
    )
    return MultiAgentRun(
        final=final,
        ctx=ctx,
        handoffs=list(state.get("handoffs") or []),
        brief=state.get("brief"),
        request=state.get("request"),
        response=state.get("response"),
        analyst_messages=analyst_messages,
        writer_messages=writer_messages,
        bound_tools=bound,
        events=list(state.get("events") or []),
        duration_s=round(time.perf_counter() - started, 1),
    )


def save_handoffs(run: MultiAgentRun, output_dir: Optional[Path] = None) -> Path:
    """The inter-agent message log as JSON, next to the reports (config.OUTPUTS_DIR, read at call time)."""
    output_dir = output_dir or config.OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / config.HANDOFFS_FILE_TEMPLATE.format(ticker=run.final.ticker, date=run.final.generated_at[:10])
    path.write_text(handoffs_to_json(run.handoffs), encoding="utf-8")
    return path
