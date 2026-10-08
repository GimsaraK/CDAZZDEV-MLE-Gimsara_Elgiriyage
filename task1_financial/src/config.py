"""Central configuration for the Task 1 pipeline.

Every tunable value (windows, thresholds, retry policy, paths) lives here so the
rest of the code contains no magic numbers.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)

from pathlib import Path

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
TICKER = "AAPL"

# Relative period string (never hardcoded dates). 3 years gives a full 2 years
# of valid 200-day SMA values after the warm-up window.
HISTORY_PERIOD = "3y"
HISTORY_INTERVAL = "1d"
MIN_HISTORY_YEARS = 2

DAYS_PER_YEAR = 365.25
TRADING_DAYS_PER_YEAR = 252
WEEKS_IN_52W_WINDOW = 52

# Raw prices match quoted market prices (current price, 52-week range).
# Adjusted prices remove split/dividend jumps (indicators, YTD return).
PRICE_COL_RAW = "Close"
PRICE_COL_ADJ = "Adj Close"
HIGH_COL = "High"
LOW_COL = "Low"
VOLUME_COL = "Volume"
REQUIRED_OHLCV_COLUMNS = ("Open", HIGH_COL, LOW_COL, PRICE_COL_RAW, PRICE_COL_ADJ, VOLUME_COL)

# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_WINDOW = 20
BB_NUM_STD = 2
BB_STD_DDOF = 0  # population std, as defined by John Bollinger

# RSI value returned when there is no price movement at all (no gains, no losses).
RSI_NEUTRAL_VALUE = 50.0
RSI_MAX_VALUE = 100.0

# ---------------------------------------------------------------------------
# Momentum signal
# ---------------------------------------------------------------------------
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_MIDLINE = 50

BB_PCT_B_LOWER = 0.0
BB_PCT_B_MID = 0.5
BB_PCT_B_UPPER = 1.0

# Sessions used to judge whether the MACD histogram is rising or falling.
MACD_HIST_LOOKBACK = 5
# A golden/death cross within this many sessions is reported as a recent event.
CROSSOVER_LOOKBACK_DAYS = 20

# Score is the mean of component votes in [-1, 1]. Labels are symmetric around 0:
# |score| >= STRONG -> Strong Bullish/Bearish, |score| >= MILD -> Bullish/Bearish, else Neutral.
MOMENTUM_STRONG_THRESHOLD = 0.6
MOMENTUM_MILD_THRESHOLD = 0.2
MOMENTUM_LABEL_STRONG_BULLISH = "Strong Bullish"
MOMENTUM_LABEL_BULLISH = "Bullish"
MOMENTUM_LABEL_NEUTRAL = "Neutral"
MOMENTUM_LABEL_BEARISH = "Bearish"
MOMENTUM_LABEL_STRONG_BEARISH = "Strong Bearish"
MOMENTUM_LABEL_UNAVAILABLE = "Unavailable"

# ---------------------------------------------------------------------------
# Summary output formatting
# ---------------------------------------------------------------------------
PRICE_DECIMALS = 2
RATIO_DECIMALS = 4
INDICATOR_DECIMALS = 4

# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
NEWS_TARGET_COUNT = 15
NEWS_MIN_COUNT = 10

GOOGLE_NEWS_RSS_URL_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={ceid}"
GOOGLE_NEWS_QUERY_TEMPLATE = '"{ticker}" OR "{company}" stock when:{recency}'
# Google News search operator restricting results to recent articles.
GOOGLE_NEWS_RECENCY_WINDOW = "7d"
GOOGLE_NEWS_LOCALE = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
GOOGLE_NEWS_TITLE_SEPARATOR = " - "

# Search results include low-signal items that are not news: quote/options listing pages
# and boilerplate institutional-holdings filing stories. Titles matching any of these
# (case-insensitive) regular expressions are dropped.
NEWS_EXCLUDE_TITLE_PATTERNS = (
    r"\boptions? chain\b",
    r"stock price, news, quote",
    r"quotes? ?& ?history",
    r"\b[A-Z]{1,6}\d{6}[CP]\d{8}\b",  # OCC option contract symbol, e.g. AAPL261007C00315000
    r"\bshares (?:purchased|sold|acquired|bought)\b.*\bby\b",  # "... Shares Sold by XYZ LLC"
    r"\b(?:trims|boosts|raises|lowers|cuts|increases|decreases|lifts|grows|reduces|sells|buys|acquires|takes)\b"
    r".{0,40}\b(?:holdings|position|stake)\s+in\b",  # "XYZ LLC Trims Stock Holdings in ..."
)

# At most this many headlines per publisher, so no single outlet dominates the sample.
NEWS_MAX_PER_PUBLISHER = 3

HTTP_TIMEOUT_SECONDS = 15
HTTP_USER_AGENT = "Mozilla/5.0 (compatible; CDAZZDEV-MLE-Assessment/1.0)"

# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_MIN_SECONDS = 1
RETRY_BACKOFF_MAX_SECONDS = 10

# Allowed relative gap between our computed 52-week range and Yahoo's value
# before a warning is logged (intraday vs daily-bar differences are expected).
WEEK52_MISMATCH_TOLERANCE_PCT = 2.0

# Info fields kept from yfinance's (large) .info payload.
INFO_KEYS = (
    "longName",
    "shortName",
    "currency",
    "trailingPE",
    "forwardPE",
    "trailingEps",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "regularMarketPrice",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TASK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TASK_DIR / "data"
OUTPUTS_DIR = TASK_DIR / "outputs"

SNAPSHOT_OHLCV_TEMPLATE = "{ticker}_ohlcv.csv"
SNAPSHOT_OHLCV_META_TEMPLATE = "{ticker}_ohlcv_meta.json"
SNAPSHOT_INFO_TEMPLATE = "{ticker}_info.json"
SNAPSHOT_NEWS_TEMPLATE = "{ticker}_news.json"

SUMMARY_OUTPUT_TEMPLATE = "{ticker}_summary.json"
CHART_OUTPUT_TEMPLATE = "{ticker}_technical_chart.png"
# Sessions shown in the technical chart and used for the indicator cross-check.
CHART_LOOKBACK_SESSIONS = TRADING_DAYS_PER_YEAR * MIN_HISTORY_YEARS
CHART_DPI = 120

# ---------------------------------------------------------------------------
# LLM sentiment and signal (Task 1B)
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 1B plan', Date: 2026-10-08 (see CITATIONS.md Entry 6)
# ---------------------------------------------------------------------------
# Groq is tried first. OpenRouter is used only when Groq is missing or its response fails validation.
LLM_PROVIDER_GROQ = "groq"
LLM_PROVIDER_OPENROUTER = "openrouter"
LLM_PROVIDER_ORDER = (LLM_PROVIDER_GROQ, LLM_PROVIDER_OPENROUTER)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Groq retired llama-3.3-70b-versatile for free and developer accounts on 2026-08-16.
# Their documented replacement is openai/gpt-oss-120b (https://console.groq.com/docs/deprecations).
# OpenRouter has no free Llama 3.3 70B route, so the fallback is a free instruction model
# from the OpenRouter catalog (checked 2026-10-08).
LLM_MODEL_GROQ = "openai/gpt-oss-120b"
LLM_MODEL_OPENROUTER = "google/gemma-4-31b-it:free"

LLM_TEMPERATURE = 0.1
# gpt-oss spends some of this budget on a hidden reasoning trace before the JSON,
# so the cap is larger than the few sentences we actually ask for.
LLM_MAX_TOKENS_SENTIMENT = 2048
LLM_MAX_TOKENS_SIGNAL = 4096
LLM_TIMEOUT_SECONDS = 60

# Ask every provider for a JSON object. Pydantic still validates the text afterwards.
JSON_RESPONSE_FORMAT = {"type": "json_object"}

# One repair call per provider after a validation failure, then the next provider.
LLM_REPAIR_ATTEMPTS = 1

# Buy/Hold/Sell justification length required by the specification.
JUSTIFICATION_MIN_SENTENCES = 3
JUSTIFICATION_MAX_SENTENCES = 5

SENTIMENT_POSITIVE = "positive"
SENTIMENT_NEGATIVE = "negative"
SENTIMENT_NEUTRAL = "neutral"
# Maps a label to a vote so the overall score is a confidence-weighted mean in [-1, +1].
SENTIMENT_VOTE = {
    SENTIMENT_POSITIVE: 1,
    SENTIMENT_NEGATIVE: -1,
    SENTIMENT_NEUTRAL: 0,
}

# Same cut-offs as the momentum score (MOMENTUM_STRONG_THRESHOLD / MOMENTUM_MILD_THRESHOLD).
SENTIMENT_LABEL_STRONG_POSITIVE = "Strong Positive"
SENTIMENT_LABEL_POSITIVE = "Positive"
SENTIMENT_LABEL_NEUTRAL = "Neutral"
SENTIMENT_LABEL_NEGATIVE = "Negative"
SENTIMENT_LABEL_STRONG_NEGATIVE = "Strong Negative"
SENTIMENT_LABEL_UNAVAILABLE = "Unavailable"

SIGNAL_BUY = "Buy"
SIGNAL_HOLD = "Hold"
SIGNAL_SELL = "Sell"
# Used only when every provider fails. Recorded with fallback=True so it is not read as a real thesis.
SIGNAL_FALLBACK = SIGNAL_HOLD

ENV_GROQ_API_KEY = "GROQ_API_KEY"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"

# OpenRouter asks callers to identify themselves. These are not secrets.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage",
    "X-Title": "CDAZZDEV MLE Task 1",
}

# ---------------------------------------------------------------------------
# Equity research brief (Task 1 bonus)
# ---------------------------------------------------------------------------
# How many headlines the brief shows. Ranked by abs(vote) * confidence.
BRIEF_TOP_HEADLINES = 3
BRIEF_MARKDOWN_TEMPLATE = "{ticker}_research_brief.md"
BRIEF_HTML_TEMPLATE = "{ticker}_research_brief.html"
# Fixed text. The renderer always includes it; the model cannot omit or rewrite it.
RISK_DISCLAIMER = (
    "This brief is an automated first-pass summary produced for a technical assessment. "
    "It is not investment advice, a solicitation, or a recommendation to buy, sell, or hold any security. "
    "Prices, indicators, and headlines can be wrong, delayed, or incomplete. "
    "Past price action does not predict future returns. "
    "Do your own research before making any financial decision."
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
