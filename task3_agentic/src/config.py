"""Task 3 constants. Every window, limit, threshold and path the agent uses is named here."""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)', Date: 2026-10-09 (see CITATIONS.md Entry 14)

from pathlib import Path

# ---------------------------------------------------------------------------
# Research target
# ---------------------------------------------------------------------------
TICKER = "AAPL"
# The query from the specification. {ticker} is filled at run time.
RESEARCH_QUERY_TEMPLATE = (
    "Analyse the current financial health and market sentiment of {ticker}. "
    "Identify the top three risks to its share price over the next 90 days "
    "and suggest one data-driven hedge strategy."
)
# Tickers are 1-10 characters: letters, digits, and the . - ^ = used by Yahoo (BRK.B, ^GSPC, EURUSD=X).
TICKER_PATTERN = r"^[A-Z0-9.\-^=]{1,10}$"

# ---------------------------------------------------------------------------
# get_price_data
# ---------------------------------------------------------------------------
# The periods yfinance accepts for daily bars.
VALID_PERIODS = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
DEFAULT_PERIOD = "1y"
# SMA-200 needs 200 sessions of warm-up, so prices are always fetched over at least 3 years.
# The requested period only decides the window of the period return and the "period" stats.
INDICATOR_WARMUP_PERIOD = "3y"
# Periods longer than the warm-up are fetched as requested.
PERIODS_LONGER_THAN_WARMUP = ("5y", "10y", "max")
# How many of the latest daily bars are returned to the agent. The rest stays in the session store.
RECENT_BARS = 10

# Extra .info fields reported as the fundamentals block (all optional; Yahoo omits some per ticker).
FUNDAMENTAL_KEYS = {
    "marketCap": "market_cap",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "profitMargins": "profit_margin",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "returnOnEquity": "return_on_equity",
    "debtToEquity": "debt_to_equity_pct",  # Yahoo reports this in percent (78 = 0.78x)
    "currentRatio": "current_ratio",
    "freeCashflow": "free_cash_flow",
    "totalCash": "total_cash",
    "totalDebt": "total_debt",
    "beta": "beta",
}
# Ratios from Yahoo are fractions (0.24 = 24%); these are converted to percent for readability.
FUNDAMENTAL_FRACTION_FIELDS = ("profit_margin", "revenue_growth", "earnings_growth", "return_on_equity")

# ---------------------------------------------------------------------------
# get_news
# ---------------------------------------------------------------------------
DEFAULT_NEWS_COUNT = 10
MIN_NEWS_COUNT = 1
MAX_NEWS_COUNT = 25
# Google News RSS links are ~250-character redirects. Longer URLs are dropped to save context.
NEWS_URL_MAX_CHARS = 150
# Task 1 aims for at least this many headlines before it falls back to the snapshot.
NEWS_MINIMUM_FOR_FALLBACK = 10

# ---------------------------------------------------------------------------
# calculate_volatility
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25
DEFAULT_VOL_WINDOW = 20
MIN_VOL_WINDOW = 5
MAX_VOL_WINDOW = 252
# Comparison windows reported next to the requested one (about 1 month, 1 quarter, 1 year).
VOL_COMPARISON_WINDOWS = (20, 60, 252)
# Volatility is fetched over this period when the session has no prices yet.
VOL_FETCH_PERIOD = "2y"
# Current volatility is ranked against the rolling readings of the last year.
VOL_PERCENTILE_LOOKBACK = 252
# Percentile cut-offs for the regime label.
VOL_REGIME_LOW_PERCENTILE = 25.0
VOL_REGIME_HIGH_PERCENTILE = 75.0
VOL_REGIME_LOW = "low"
VOL_REGIME_NORMAL = "normal"
VOL_REGIME_ELEVATED = "elevated"
# The query asks about the next 90 days. Calendar days are converted to trading days for the move.
HEDGE_HORIZON_CALENDAR_DAYS = 90
# Drawdown is measured over the last year of sessions.
DRAWDOWN_LOOKBACK = 252

# ---------------------------------------------------------------------------
# llm_sentiment
# ---------------------------------------------------------------------------
# Each headline is one LLM call (Task 1 scorer). The cap keeps a run inside Groq's free per-minute budget.
MAX_SENTIMENT_HEADLINES = 10
SENTIMENT_TOP_N = 3
# A headline passed to llm_sentiment must match a title from get_news or web_search. A shortened
# title still matches if it is contained in (or contains) a known one and is at least this long.
HEADLINE_MATCH_MIN_CHARS = 20

# ---------------------------------------------------------------------------
# web_search (ddgs, the renamed duckduckgo-search package)
# ---------------------------------------------------------------------------
SEARCH_MAX_RESULTS = 6
SEARCH_REGION = "us-en"
SEARCH_SAFESEARCH = "moderate"
# "m" = past month. Analyst commentary older than that says little about the next 90 days.
SEARCH_TIMELIMIT = "m"
SEARCH_TIMEOUT_SECONDS = 10
SEARCH_SNIPPET_MAX_CHARS = 300
# A rate-limited backend is tried this many times in total before moving on.
SEARCH_ATTEMPTS_PER_BACKEND = 2
SEARCH_RETRY_WAIT_SECONDS = 2.0
# The simplified fallback query keeps only this many words.
SEARCH_SIMPLIFIED_MAX_WORDS = 6

# ---------------------------------------------------------------------------
# Tool envelope and trace
# ---------------------------------------------------------------------------
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"
# Upper bound on one tool result as the agent sees it. Keeps the context inside Groq's free tier.
TOOL_MESSAGE_MAX_CHARS = 2500
TRUNCATION_MARKER = "...[truncated]"
# The specification asks for the output truncated to 200 characters in agent_trace.jsonl.
TRACE_OUTPUT_MAX_CHARS = 200
DEFAULT_AGENT_NAME = "research_agent"

# ---------------------------------------------------------------------------
# Agent LLM
# ---------------------------------------------------------------------------
LLM_PROVIDER_GROQ = "groq"
LLM_PROVIDER_OPENROUTER = "openrouter"
GROQ_MODEL = "openai/gpt-oss-120b"
# Same family, smaller, with its own Groq daily token quota. Used when gpt-oss-120b hits its
# free-tier cap (200k tokens per rolling 24 h), before any OpenRouter model.
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
GROQ_MODELS = (GROQ_MODEL, GROQ_FALLBACK_MODEL)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Free OpenRouter models that list "tools" in supported_parameters (checked 2026-10-09).
# Tried in order only when Groq fails after its retries. Nemotron goes first: Gemma's free
# route was rate-limited upstream in every test on 2026-10-09.
OPENROUTER_FALLBACK_MODELS = ("nvidia/nemotron-3-super-120b-a12b:free", "google/gemma-4-31b-it:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage",
    "X-Title": "CDAZZDEV MLE Task 3",
}
ENV_GROQ_API_KEY = "GROQ_API_KEY"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
# Low temperature: the agent should plan consistently, not creatively.
AGENT_TEMPERATURE = 0.1
# gpt-oss spends part of the budget on hidden reasoning before the tool call or the answer.
# Groq counts max_tokens against the 8k tokens-per-minute budget, so the agent cap stays modest.
AGENT_MAX_TOKENS = 1536
REPORT_MAX_TOKENS = 4096
# Tool-choice turns need little deliberation; low effort roughly halves their hidden reasoning tokens.
# The report writer does the synthesis, so it keeps medium.
AGENT_REASONING_EFFORT = "low"
REPORT_REASONING_EFFORT = "medium"
LLM_TIMEOUT_SECONDS = 90
# Groq's free tier allows 8k tokens per minute, so per-minute 429s are expected; the SDK waits out
# retry-after (up to 60 s) on each retry. A daily-limit 429 is not retried for long: FallbackChain
# skips that model for the rest of the session after its first failure.
GROQ_MAX_RETRIES = 5
OPENROUTER_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Agent graph limits
# ---------------------------------------------------------------------------
# Tool rounds (agent turns that call tools) before the agent must write the report.
MAX_TOOL_ROUNDS = 8
# Extra tool rounds granted each time the report check sends the agent back.
REVISION_TOOL_ROUNDS = 3
# How many times the report check may send the agent back before the report is accepted with warnings.
MAX_REPORT_REVISIONS = 2
# Repair calls when the structured report fails Pydantic validation.
REPORT_REPAIR_ATTEMPTS = 1
# LangGraph super-step limit, a hard stop above the budgets above.
RECURSION_LIMIT = 60
# How many results per tool go into the report-writing digest (newest first).
DIGEST_RESULTS_PER_TOOL = 2
# Agent notes passed to the report writer are cut to this length.
AGENT_NOTES_MAX_CHARS = 2000

# ---------------------------------------------------------------------------
# 3B multi-agent pipeline
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3B plan', Date: 2026-10-09 (see CITATIONS.md Entry 16)
# ---------------------------------------------------------------------------
AGENT_A_NAME = "data_analyst"
AGENT_B_NAME = "research_writer"
ORCHESTRATOR_NAME = "orchestrator"
# Tool access per agent, as the specification assigns it. Enforced in three layers (see multi_agent.py).
AGENT_A_TOOLS = ("get_price_data", "calculate_volatility", "llm_sentiment")
AGENT_B_TOOLS = ("web_search", "get_news")
# Tool-round budgets: A's first pass, B's research pass, and A's answer to the clarification.
ANALYST_MAX_ROUNDS = 5
WRITER_MAX_ROUNDS = 5
ANALYST_CLARIFY_ROUNDS = 3
# B sends exactly one clarification request per run (the critique loop runs once, always).
CLARIFICATION_ROUNDS = 1
# Repair calls for B's request and B's final report when validation fails.
REQUEST_REPAIR_ATTEMPTS = 1
FINAL_REPORT_REPAIR_ATTEMPTS = 1
# Bounds on the interpretive fields of A's brief.
BRIEF_MIN_FINDINGS = 2
BRIEF_MAX_FINDINGS = 5
# A's clarification answer passed to B is cut to this length.
CLARIFICATION_ANSWER_MAX_CHARS = 600
# How close a number in B's report must be to A's returned value to count as incorporated.
# Sentiment scores live in [-1, 1]; percentages (volatility, returns) are compared in percentage points.
INCORPORATION_SCORE_TOLERANCE = 0.01
INCORPORATION_PCT_TOLERANCE = 0.5
INCORPORATION_RELATIVE_TOLERANCE = 0.01
# Inter-agent message log and the pipeline's report files.
HANDOFFS_FILE_TEMPLATE = "{ticker}_{date}_handoffs.json"
MULTI_AGENT_REPORT_SUFFIX = "_multi_agent"
MULTI_AGENT_STATUS_FAILED = "failed"

# ---------------------------------------------------------------------------
# 3C memory, cache and observability
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan', Date: 2026-10-09 (see CITATIONS.md Entry 18)
# ---------------------------------------------------------------------------
MODE_RESEARCH = "research"
MODE_FOLLOWUP = "followup"
# A follow-up may call at most this many tool rounds; the point is to answer from memory.
FOLLOWUP_TOOL_ROUNDS = 1
# Numbers in a follow-up answer count as cited when they match the remembered value this closely.
FOLLOWUP_MATCH_TOLERANCE = 0.01
# ...or within this fraction of the value, so a price written to one decimal place still counts.
FOLLOWUP_RELATIVE_TOLERANCE = 0.002
# The cache key's date is the market's calendar date, not UTC.
CACHE_TIMEZONE = "America/New_York"
CACHE_SCHEMA_VERSION = 1
CACHE_FILE_TEMPLATE = "{ticker}_{date}_{pipeline}.json"
PIPELINE_SINGLE_AGENT = "single_agent"
PIPELINE_MULTI_AGENT = "multi_agent"
PIPELINES = (PIPELINE_SINGLE_AGENT, PIPELINE_MULTI_AGENT)
# The fields every agent_trace.jsonl line must carry (spec: tool, inputs, output <= 200 chars, duration).
TRACE_REQUIRED_FIELDS = ("timestamp", "session_id", "agent", "tool", "args", "status", "output", "duration_ms")

# ---------------------------------------------------------------------------
# Report checks
# ---------------------------------------------------------------------------
REQUIRED_RISKS = 3
MIN_KEY_METRICS = 3
# The hedge's volatility numbers must match a calculate_volatility result within this many percentage points.
VOL_CHECK_TOLERANCE_PCT_POINTS = 0.5
REPORT_STATUS_VALIDATED = "validated"
REPORT_STATUS_UNVALIDATED = "accepted_with_warnings"
REPORT_STATUS_FAILED = "failed"
RISK_DISCLAIMER = (
    "This report was produced automatically by an LLM agent for a technical assessment. "
    "It is not investment advice or a recommendation to trade any security. "
    "Market data, headlines and search results can be delayed, incomplete or wrong, "
    "and volatility-based ranges are statistical estimates, not forecasts."
)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
# The DECIDE line shows at most this much of the model's note or reasoning.
DISPLAY_DECISION_MAX_CHARS = 320
DISPLAY_OBSERVATION_MAX_CHARS = 260
# 3B: a handoff payload is printed up to this length (the full payload is in the handoffs JSON file).
DISPLAY_HANDOFF_MAX_CHARS = 900

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TASK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parent
LOGS_DIR = TASK_DIR / "logs"
OUTPUTS_DIR = TASK_DIR / "outputs"
CACHE_DIR = TASK_DIR / "cache"
TRACE_FILE = LOGS_DIR / "agent_trace.jsonl"
# 3C: cache and memory events (cache_hit, cache_miss, followup_answered, ...), kept out of the tool trace.
EVENTS_FILE = LOGS_DIR / "session_events.jsonl"
REPORT_FILE_TEMPLATE = "{ticker}_{date}_research_report{suffix}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
