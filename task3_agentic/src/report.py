"""Evidence digest for the report writer, deterministic report checks, Markdown rendering and saving.

check_report() is the guard against a report that sounds right but is not backed by
this session's data: every risk must cite a tool that actually succeeded, and the
hedge's volatility numbers must match a calculate_volatility result.
"""
# AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .schemas import TOOL_NAMES, FinalReport, ResearchReport
from .session import SessionContext

# Fields left out of the digest: the report never needs raw daily bars.
_DIGEST_EXCLUDE = {"data": {"recent_bars", "recent_bars_columns"}}


# --------------------------------------------------------------------------- digest
def build_evidence_digest(ctx: SessionContext) -> str:
    """Compact JSON of the newest successful results per tool, plus the tools that only failed."""
    succeeded: Dict[str, List[Dict[str, Any]]] = {}
    for tool in TOOL_NAMES:
        records = ctx.results_for(tool)[: config.DIGEST_RESULTS_PER_TOOL]
        if records:
            succeeded[tool] = [
                {
                    "args": record.args,
                    "result": record.result.model_dump(mode="json", exclude_none=True, exclude=_DIGEST_EXCLUDE),
                }
                for record in records
            ]
    failed = {}
    for tool in sorted(ctx.failed_tools()):
        last = ctx.results_for(tool, ok_only=False)[0].result
        failed[tool] = {"status": last.status, "error": last.error}
    return json.dumps({"succeeded": succeeded, "failed": failed}, separators=(",", ":"), default=str)


def build_writer_digest(ctx: SessionContext, brief: Any, request: Any, response: Any, agent: str) -> str:
    """3B: what Agent B may use for the final report. A's data reaches B only through the brief and the answer.

    AI-ASSISTED: Claude Code (claude-opus-5-5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)
    """
    own: Dict[str, List[Dict[str, Any]]] = {}
    failed: Dict[str, Any] = {}
    for tool in config.AGENT_B_TOOLS:
        records = ctx.results_for(tool, agent=agent)[: config.DIGEST_RESULTS_PER_TOOL]
        if records:
            own[tool] = [
                {"args": r.args, "result": r.result.model_dump(mode="json", exclude_none=True)} for r in records
            ]
        else:
            attempts = ctx.results_for(tool, ok_only=False, agent=agent)
            if attempts:
                failed[tool] = {"status": attempts[0].result.status, "error": attempts[0].result.error}
    digest = {
        "data_brief_from_agent_a": brief.model_dump(mode="json", exclude_none=True) if brief else None,
        "clarification": {
            "request_to_agent_a": request.model_dump(mode="json", exclude_none=True) if request else None,
            "answer_from_agent_a": response.model_dump(mode="json", exclude_none=True) if response else None,
        },
        "writer_tool_results": {"succeeded": own, "failed": failed},
    }
    return json.dumps(digest, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------- checks
def check_report(report: Optional[ResearchReport], ctx: SessionContext) -> List[str]:
    """Problems with a report, as instructions the agent can act on. An empty list means it passed."""
    if report is None:
        return ["No valid report was produced."]
    issues: List[str] = []
    succeeded = ctx.succeeded_tools()

    if report.ticker != ctx.ticker:
        issues.append(f"The report is about {report.ticker}, but the session ticker is {ctx.ticker}.")

    # Financial health must rest on price data.
    health = report.financial_health
    if "get_price_data" not in succeeded:
        issues.append(
            "Financial health has no price data: get_price_data has not succeeded this session. "
            "Call it (retry once if it failed)."
        )
    elif not any(m.source_tool == "get_price_data" for m in health.key_metrics):
        issues.append("financial_health.key_metrics must include values from get_price_data.")
    if len(health.key_metrics) < config.MIN_KEY_METRICS:
        issues.append(f"financial_health.key_metrics needs at least {config.MIN_KEY_METRICS} metrics.")
    unbacked_metrics = sorted({m.source_tool for m in health.key_metrics} - succeeded)
    if unbacked_metrics:
        issues.append(f"key_metrics cite {unbacked_metrics}, which returned no usable data this session.")

    # Market sentiment must come from the sentiment tool.
    if "llm_sentiment" not in succeeded:
        issues.append(
            "Market sentiment has no llm_sentiment score. Call get_news, then llm_sentiment on the headline "
            "titles; if llm_sentiment keeps failing, say so in data_gaps."
        )

    # Every risk needs evidence from a tool that worked.
    for number, risk in enumerate(report.risks, start=1):
        cited = {e.source_tool for e in risk.evidence}
        if not cited & succeeded:
            issues.append(
                f"Risk {number} ('{risk.name}') has no evidence from a tool that succeeded this session "
                f"(cited {sorted(cited)}). Back it with data from a working tool."
            )
        elif cited - succeeded:
            issues.append(
                f"Risk {number} ('{risk.name}') cites {sorted(cited - succeeded)}, which returned no usable data; "
                "replace that evidence."
            )

    # A tool that only failed is a gap the reader must be told about.
    gap_text = " ".join(report.data_gaps).lower()
    for tool in sorted(ctx.failed_tools()):
        if tool not in gap_text:
            issues.append(
                f"{tool} failed this session but data_gaps does not mention it. Retry it or use an alternative, "
                f"and record in data_gaps that {tool} failed and how you worked around it."
            )

    # The hedge must use computed volatility, not remembered or invented numbers.
    vol_records = ctx.results_for("calculate_volatility")
    if not vol_records:
        issues.append(
            "The hedge is not data-driven: calculate_volatility has not succeeded this session. Call it."
        )
    else:
        hedge = report.hedge
        tolerance = config.VOL_CHECK_TOLERANCE_PCT_POINTS
        computed = [(r.result.data.annualised_vol_pct, r.result.data.expected_move_pct) for r in vol_records]
        matches = any(
            abs(hedge.annualised_vol_pct - vol_pct) <= tolerance and abs(hedge.expected_move_90d_pct - move_pct) <= tolerance
            for vol_pct, move_pct in computed
        )
        if not matches:
            issues.append(
                f"The hedge uses annualised vol {hedge.annualised_vol_pct}% and 90-day move "
                f"{hedge.expected_move_90d_pct}%, which match no calculate_volatility result "
                f"(computed (vol, move) pairs: {computed}). Use the computed values."
            )
    return issues


# --------------------------------------------------------------------------- render and save
def _cell(text: Any) -> str:
    """Make a value safe for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(final: FinalReport, company_name: Optional[str] = None) -> str:
    title = f"{company_name} ({final.ticker})" if company_name else final.ticker
    lines = [
        f"# {title} - Agent Research Report",
        "",
        f"*Generated {final.generated_at} | session `{final.session_id}` | status: **{final.status}** | "
        f"report revisions: {final.revisions} | tools used: {', '.join(final.tools_used) or 'none'}*",
        "",
        f"*Agent turns by model: {', '.join(f'{m} x{n}' for m, n in final.agent_models.items()) or 'n/a'} | "
        f"report written by: {final.report_model or 'n/a'}*",
        "",
    ]
    if final.agent_tools:
        # 3B provenance: which agent gathered what.
        lines += [
            "*Pipeline: " + " | ".join(f"{agent}: {', '.join(tools) or 'no tools'}" for agent, tools in final.agent_tools.items()) + "*",
            "",
        ]
    lines += [f"> {final.query}", ""]
    report = final.report
    if report is None:
        lines += ["## Report unavailable", "", final.error or "The agent did not produce a report.", ""]
    else:
        health = report.financial_health
        lines += ["## 1. Financial Health Summary", "", health.summary, "", f"**Market sentiment.** {health.market_sentiment}", ""]
        lines += ["| Metric | Value | Source tool |", "|---|---|---|"]
        lines += [f"| {_cell(m.name)} | {_cell(m.value)} | `{m.source_tool}` |" for m in health.key_metrics]
        lines += ["", "## 2. Top Three Risks (next 90 days)", ""]
        for number, risk in enumerate(report.risks, start=1):
            lines += [f"### Risk {number}: {risk.name}", "", risk.description, "", "Evidence:"]
            for item in risk.evidence:
                link = f" ([source]({item.url}))" if item.url else ""
                lines.append(f"- `{item.source_tool}`: {item.detail}{link}")
            lines.append("")
        hedge = report.hedge
        lines += [
            "## 3. Hedge Strategy Recommendation",
            "",
            f"**Strategy:** {hedge.strategy}  ",
            f"**Instrument:** {hedge.instrument}  ",
            f"**Volatility basis:** annualised {hedge.annualised_vol_pct}%, 90-day 1-sigma move {hedge.expected_move_90d_pct}%",
            "",
            f"**Rationale.** {hedge.rationale}",
            "",
            f"**Sizing.** {hedge.sizing}",
            "",
            f"**Trade-offs.** {hedge.trade_offs}",
            "",
        ]
        if report.clarification_used:
            lines += ["## Clarification from the Data Analyst", "", report.clarification_used, ""]
        if report.data_gaps:
            lines += ["## Data gaps", ""] + [f"- {gap}" for gap in report.data_gaps] + [""]
    if final.validation_warnings:
        lines += ["## Validation warnings", ""] + [f"- {w}" for w in final.validation_warnings] + [""]
    lines += ["---", "", f"*{config.RISK_DISCLAIMER}*", ""]
    return "\n".join(lines)


def save_report(
    final: FinalReport,
    company_name: Optional[str] = None,
    suffix: str = "",
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Write the report as Markdown and JSON. Returns (markdown_path, json_path).

    The folder defaults to config.OUTPUTS_DIR read at call time, so it can be redirected.
    """
    output_dir = output_dir or config.OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = config.REPORT_FILE_TEMPLATE.format(ticker=final.ticker, date=final.generated_at[:10], suffix=suffix)
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    md_path.write_text(render_markdown(final, company_name), encoding="utf-8")
    json_path.write_text(final.model_dump_json(indent=2), encoding="utf-8")
    return md_path, json_path
