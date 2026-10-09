"""Task 1 bonus: one-page equity research brief.

Combines the Part 1A summary and chart with the Part 1B sentiment and signal
into a Markdown file and a styled HTML page. Headline ranking, the technical
paragraph, and the risk disclaimer are computed here. The Buy/Hold/Sell
justification is copied from the model and is not rewritten.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Task 1 bonus research brief', Date: 2026-10-08

import base64
import html
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import markdown
from jinja2 import Template

from . import config
from .llm import HeadlineSentiment, SentimentResult, SignalRecommendation
from .summary import MomentumComponent, StockSummary

# Punctuation the model sometimes emits. The brief files stay plain ASCII.
_ASCII_PUNCTUATION = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
    "\u2026": "...",
}

_HTML_PAGE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  @page { size: A4; margin: 10mm; }
  body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.32;
    color: #1c1c1c;
    max-width: 190mm;
    margin: 8mm auto;
  }
  h1 { font-size: 16pt; margin: 0 0 1mm; }
  h2 { font-size: 11.5pt; margin: 3mm 0 1mm; border-bottom: 1px solid #cccccc; }
  p, li { margin: 0.6mm 0; }
  ul { margin: 0; padding-left: 4.5mm; }
  img { display: block; width: 100%; max-height: 58mm; object-fit: contain; margin: 1mm 0 2mm; }
  .meta { color: #444444; margin: 0 0 2mm; }
</style>
</head>
<body>
{{ body }}
</body>
</html>
""")


def _ascii(text: str) -> str:
    """Replace common punctuation, then drop any character that is still not ASCII."""
    converted = text or ""
    for source, target in _ASCII_PUNCTUATION.items():
        converted = converted.replace(source, target)
    # Angle brackets would be raw HTML once the Markdown is rendered.
    converted = converted.replace("<", "(").replace(">", ")")
    return "".join(ch for ch in converted if ord(ch) < 128)


def _number(value: Optional[float], digits: int = 2) -> str:
    """Fixed-point text, or 'n/a' when the pipeline left the value missing."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _component(summary: StockSummary, name: str) -> Optional[MomentumComponent]:
    """The named momentum vote, if that rule ran."""
    for component in summary.momentum.components:
        if component.name == name:
            return component
    return None


def headline_contribution(item: HeadlineSentiment) -> float:
    """How much this headline can move the overall score: abs(vote) times confidence.

    Neutral is vote 0, so a confident neutral contributes nothing. Positive and
    negative use the same +1 / -1 votes as aggregate_sentiment.
    """
    return abs(config.SENTIMENT_VOTE[item.sentiment]) * item.confidence


def select_top_headlines(
    items: Sequence[HeadlineSentiment],
    n: int = config.BRIEF_TOP_HEADLINES,
) -> List[HeadlineSentiment]:
    """The `n` headlines with the largest absolute contribution.

    Ties go to the higher confidence, then to the earlier headline, so the
    order does not depend on dict ordering.
    """
    # enumerate() keeps each headline's original position. The sort key is a tuple compared left to right:
    # largest contribution first (negated, because sorted() is ascending), then higher confidence,
    # then the earlier position, so ties always resolve the same way.
    ranked = sorted(
        enumerate(items),
        key=lambda pair: (-headline_contribution(pair[1]), -pair[1].confidence, pair[0]),
    )
    return [item for _, item in ranked[:n]]


def technical_outlook(summary: StockSummary) -> str:
    """A short reading of the trend, MACD, RSI, and Bollinger votes. No new model call."""
    indicators = summary.latest_indicators
    price = summary.current_price
    sma_50 = indicators.get("SMA_50")
    sma_200 = indicators.get("SMA_200")
    sentences: List[str] = []

    # Trend: the three price/SMA votes, stated as positions rather than as a table.
    if price is not None and sma_50 is not None and sma_200 is not None:
        versus_50 = "above" if price >= sma_50 else "below"
        versus_200 = "above" if price >= sma_200 else "below"
        regime = "above" if sma_50 >= sma_200 else "below"
        sentences.append(
            f"Price ({_number(price)}) is {versus_50} the 50-day SMA ({_number(sma_50)}) "
            f"and {versus_200} the 200-day SMA ({_number(sma_200)}), and the 50-day average "
            f"sits {regime} the 200-day average."
        )

    # MACD level plus whether the histogram is rising or falling.
    macd = indicators.get("MACD")
    macd_signal = indicators.get("MACD_Signal")
    if macd is not None and macd_signal is not None:
        relation = "above" if macd >= macd_signal else "below"
        histogram = indicators.get("MACD_Hist")
        histogram_vote = _component(summary, "macd_histogram_direction")
        extra = ""
        if histogram is not None and histogram_vote is not None and histogram_vote.vote is not None:
            direction = {1: "rising", -1: "falling", 0: "flat"}.get(histogram_vote.vote, "unchanged")
            extra = f" The histogram is {_number(histogram)} and {direction} over the last 5 sessions."
        sentences.append(
            f"MACD ({_number(macd)}) is {relation} its signal line ({_number(macd_signal)}).{extra}"
        )

    # RSI and %B, plus any stretched-condition flags the momentum score recorded.
    detail: List[str] = []
    rsi = indicators.get("RSI_14")
    percent_b = indicators.get("BB_PctB")
    if rsi is not None:
        detail.append(f"RSI(14) is {_number(rsi, 1)}")
    if percent_b is not None:
        half = "upper" if percent_b >= 0.5 else "lower"
        detail.append(f"Bollinger %B is {_number(percent_b)} ({half} half of the band)")
    if detail:
        flag_text = ""
        if summary.momentum.flags:
            flag_text = " Flags: " + "; ".join(_ascii(flag) for flag in summary.momentum.flags) + "."
        sentences.append("; ".join(detail) + "." + flag_text)

    if summary.momentum.score is not None:
        sentences.append(
            f"The momentum score is {_number(summary.momentum.score, 4)} "
            f"({summary.momentum.label}), the mean of the indicator votes."
        )
    if not sentences:
        return "Technical indicators were not available for this run."
    return " ".join(sentences)


def _snapshot_lines(summary: StockSummary) -> List[str]:
    """The company-snapshot bullets. Missing fields stay visible as n/a plus the pipeline's reason."""
    name = _ascii(summary.company_name or summary.ticker)
    currency = _ascii(summary.currency or "USD")
    if summary.pe_ratio is None:
        pe_text = "n/a"
        if summary.pe_unavailable_reason:
            pe_text += f" ({_ascii(summary.pe_unavailable_reason)})"
    else:
        pe_text = _number(summary.pe_ratio)
        if summary.pe_source:
            pe_text += f" (source {_ascii(summary.pe_source)})"
    if summary.forward_pe is not None:
        pe_text += f"; forward P/E {_number(summary.forward_pe)}"

    # Prefer the stored percentage; fall back to converting the fraction; otherwise show why it is missing.
    # ":+.2f" always prints a sign, so a gain reads "+12.34%" and a loss "-3.10%".
    if summary.ytd_return_pct is not None:
        ytd_text = f"{summary.ytd_return_pct:+.2f}%"
    elif summary.ytd_return is not None:
        ytd_text = f"{summary.ytd_return * 100:+.2f}%"
    else:
        ytd_text = "n/a"
        if summary.ytd_unavailable_reason:
            ytd_text += f" ({_ascii(summary.ytd_unavailable_reason)})"
    if summary.ytd_base_price is not None and summary.ytd_base_date:
        ytd_text += f" (base {_number(summary.ytd_base_price)} on {summary.ytd_base_date})"

    high = _number(summary.week_52_high)
    low = _number(summary.week_52_low)
    if summary.week_52_high_date:
        high += f" on {summary.week_52_high_date}"
    if summary.week_52_low_date:
        low += f" on {summary.week_52_low_date}"

    score = _number(summary.momentum.score, 4) if summary.momentum.score is not None else "n/a"
    return [
        f"- Company: {name} ({summary.ticker})",
        f"- Price: {_number(summary.current_price)} {currency} as of {summary.as_of_date or 'n/a'}",
        f"- 52-week range: {low} to {high}",
        f"- Trailing P/E: {pe_text}",
        f"- Year-to-date return: {ytd_text}",
        f"- Momentum: {summary.momentum.label} (score {score})",
    ]


def _sentiment_lines(sentiment: SentimentResult) -> List[str]:
    """Overall score, then the top headlines. Unscored headlines are named but not ranked."""
    if sentiment.score is None:
        reason = _ascii(sentiment.unavailable_reason or "no scored headlines")
        return [f"News sentiment was not available ({reason})."]
    lines = [
        (
            f"Overall sentiment is {sentiment.label} "
            f"(score {_number(sentiment.score, 4)}, {sentiment.n_scored} scored, "
            f"{len(sentiment.unscored)} unscored). "
            "The score is a confidence-weighted mean: positive = +1, negative = -1, neutral = 0."
        ),
        "",
        "Top headlines by absolute contribution (confidence times the vote):",
        "",
    ]
    top = select_top_headlines(sentiment.items)
    if not top:
        lines.append("No headline was scored.")
        return lines
    for index, item in enumerate(top, start=1):
        lines.append(
            f"{index}. **{item.sentiment}** (confidence {_number(item.confidence, 2)}, "
            f"contribution {_number(headline_contribution(item), 2)}) - {_ascii(item.headline)}. "
            f"{_ascii(item.brief_reason)}"
        )
    if sentiment.unscored:
        lines.append("")
        lines.append("Unscored (excluded from the average): " + "; ".join(_ascii(t) for t in sentiment.unscored) + ".")
    return lines


def _recommendation_lines(signal: SignalRecommendation) -> List[str]:
    """The model's signal, or an explicit note when Hold is only the failure fallback."""
    if signal.fallback:
        reason = _ascii(signal.unavailable_reason or "the model did not return a usable justification").rstrip(".")
        return [
            f"**{signal.signal}** is a pipeline fallback, not the model's recommendation. {reason}."
        ]
    provider = f" Provider: {_ascii(signal.provider)}." if signal.provider else ""
    return [
        f"**{signal.signal}**.{provider}",
        "",
        _ascii(signal.justification),
    ]


def build_brief_markdown(
    summary: StockSummary,
    sentiment: SentimentResult,
    signal: SignalRecommendation,
    chart_filename: Optional[str],
) -> str:
    """The one-page brief. Section titles match the bonus rubric."""
    title = _ascii(summary.company_name or summary.ticker)
    lines = [
        f"# {title} ({summary.ticker}) - Equity Research Brief",
        "",
        f"As of {summary.as_of_date or 'n/a'}. Figures come from this pipeline run.",
        "",
        "## Company snapshot",
        "",
        *_snapshot_lines(summary),
        "",
        "## Technical outlook",
        "",
        technical_outlook(summary),
        "",
    ]
    # The image sits next to the Markdown file in outputs/.
    if chart_filename:
        lines.extend([
            f"![{summary.ticker} price, moving averages, RSI, MACD and Bollinger Bands]({chart_filename})",
            "",
        ])
    lines.extend([
        "## News sentiment",
        "",
        *_sentiment_lines(sentiment),
        "",
        "## Recommendation",
        "",
        *_recommendation_lines(signal),
        "",
        "## Risk disclaimer",
        "",
        config.RISK_DISCLAIMER,
        "",
    ])
    return "\n".join(lines)


def build_brief_html(markdown_text: str, title: str, chart_path: Optional[Path]) -> str:
    """Render the Markdown and inline the chart so the HTML file stands alone."""
    body = markdown.markdown(markdown_text)
    # Swap the relative image for a data URI. The Markdown file keeps the filename.
    if chart_path is not None and chart_path.is_file():
        # The PNG bytes are base64-encoded into the <img> tag itself, so the HTML file can be opened
        # or emailed on its own without the separate image file.
        encoded = base64.b64encode(chart_path.read_bytes()).decode("ascii")
        relative = f'src="{html.escape(chart_path.name)}"'
        body = body.replace(relative, f'src="data:image/png;base64,{encoded}"')
    return _HTML_PAGE.render(title=_ascii(title), body=body)


def write_research_brief(
    summary: StockSummary,
    sentiment: SentimentResult,
    signal: SignalRecommendation,
    chart_path: Path,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Write the Markdown and HTML briefs. Returns the two paths.

    A missing chart is recorded in the Markdown as no image; the HTML is still written.
    """
    directory = Path(output_dir) if output_dir is not None else config.OUTPUTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    chart = Path(chart_path)
    chart_name = chart.name if chart.is_file() else None

    # Build both views from the same text so the sections cannot drift apart.
    text = build_brief_markdown(summary, sentiment, signal, chart_name)
    title = f"{summary.company_name or summary.ticker} ({summary.ticker}) equity research brief"
    page = build_brief_html(text, title, chart if chart_name else None)

    markdown_path = directory / config.BRIEF_MARKDOWN_TEMPLATE.format(ticker=summary.ticker)
    html_path = directory / config.BRIEF_HTML_TEMPLATE.format(ticker=summary.ticker)
    markdown_path.write_text(text, encoding="utf-8", newline="\n")
    html_path.write_text(page, encoding="utf-8", newline="\n")
    return markdown_path, html_path
