# Task 1 - Financial AI: LLM-Powered Equity Research Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage/blob/main/task1_financial/task1_equity_research.ipynb)

An automated equity research assistant. It ingests real market data and news, computes technical indicators from first principles, uses an LLM to reason about sentiment and signals, and produces a structured first-pass research brief.

| Part | Status |
|---|---|
| 1A - Financial data pipeline | Done |
| 1B - LLM sentiment and signal reasoning | Next |
| Bonus - Report rendering | Planned |

## Contents

| Path | Description |
|---|---|
| `task1_equity_research.ipynb` | Main notebook, executed with outputs visible |
| `src/config.py` | Every tunable constant (windows, thresholds, retry policy, paths) - no magic numbers elsewhere |
| `src/data.py` | yfinance OHLCV + company info, validation and cleaning, snapshot fallback |
| `src/indicators.py` | SMA, EMA, Wilder RMA, RSI, MACD, Bollinger Bands from first principles |
| `src/news.py` | yfinance news + Google News RSS, curation (filtering, de-duplication, per-publisher cap), snapshot fallback |
| `src/summary.py` | Momentum composite signal and the validated `StockSummary` dictionary |
| `src/pipeline.py` | End-to-end orchestration; never raises, returns `ok` / `degraded` / `failed` |
| `src/retry.py`, `src/errors.py`, `src/logging_utils.py` | Retry policy (tenacity), handled exception types, logging setup |
| `tests/` | 92 offline pytest tests (no network) with recorded fixtures |
| `data/` | Committed snapshot (OHLCV CSV, company info, news) used if live sources fail |
| `outputs/` | Technical chart and the full summary JSON |
| `prompts/` | LLM prompt templates (Part 1B) |

## Part 1A - Financial Data Pipeline

### Design decisions

| Area | Decision | Why |
|---|---|---|
| Ticker | `AAPL` (config constant) | Liquid, consistently positive P/E, abundant news |
| History | `period="3y"` (relative, never hardcoded dates) | Meets the 2-year minimum and leaves 2 full years of valid SMA-200 values |
| Prices | Raw `Close` / `High` / `Low` for current price and 52-week range; `Adj Close` for indicators and YTD | Quoted values match the market; indicators are not distorted by splits or dividends |
| RSI | Wilder smoothing, seeded with the simple average of the first 14 changes | Wilder's original definition |
| EMA | Explicit recursion, alpha = 2/(n+1), seeded with the first price | Transparent first-principles implementation, identical to `ewm(adjust=False)` |
| Bollinger | Population standard deviation (`ddof=0`) | Bollinger's own definition |
| 52-week range | Calendar window (last 52 weeks by date), cross-checked against Yahoo's values | Not shifted by market holidays |
| YTD | Adjusted close vs the last close of the previous calendar year | Standard market convention |
| P/E | `trailingPE`, else price / trailing EPS, else `None` with a stated reason; `forwardPE` reported separately | P/E is meaningless for non-positive EPS |
| Momentum | 7 indicator rules vote +1 / -1 / 0; score = mean of the available votes in [-1, 1], mapped to 5 symmetric labels | Transparent and explainable, with a per-component breakdown that Part 1B reuses |
| News | yfinance news first, Google News RSS (last 7 days) as top-up; low-signal items filtered, de-duplicated, newest first, max 3 per publisher; target 15, minimum 10 | yfinance news currently returns no items, so a keyless second source is essential |
| Robustness | tenacity retries (exponential backoff, full jitter), committed snapshot fallback, Pydantic validation, stage-isolated pipeline | No unhandled exceptions; every degradation is logged and surfaced |

### Verification

- **Unit tests:** `tests/test_indicators.py` checks every indicator against hand-computed values and the [StockCharts RSI worked example](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi), plus edge cases (flat, monotonic, NaN input). The other test modules cover data cleaning, both yfinance news schemas, RSS parsing, curation, snapshot fallback, the P/E chain, the YTD and 52-week logic, the momentum thresholds, and a check that `src/` contains no hardcoded date strings.
- **Independent cross-check:** the notebook compares our indicators with the open-source [`ta`](https://github.com/bukosabino/ta) library (validation only, never used in the pipeline). Over the last two years, every column matches exactly, except RSI, which agrees to within 1.4e-7.

### Results (AAPL, run of 2026-10-07)

| Field | Value |
|---|---|
| Current price | 336.67 USD |
| 52-week high / low | 345.34 / 243.42 (matches Yahoo's values) |
| P/E (trailing) | 38.61 (forward 35.13) |
| YTD return | +24.18% |
| Momentum signal | Bullish (score 0.43: trend components bullish, MACD below its signal and fading) |
| Headlines | 15 from Google News RSS (35 listing / filing items filtered, per-publisher cap applied) |

The full summary dictionary is in [`outputs/AAPL_summary.json`](outputs/AAPL_summary.json) and the chart in [`outputs/AAPL_technical_chart.png`](outputs/AAPL_technical_chart.png).

![AAPL technical chart](outputs/AAPL_technical_chart.png)

## How to Run

**Colab:** open the notebook with the badge above and run all cells. The first cell clones this repository and installs `requirements.txt`. Part 1A needs no API keys.

**Locally:**

```bash
pip install -r task1_financial/requirements.txt
python -m pytest task1_financial/tests -q          # 92 offline tests
jupyter notebook task1_financial/task1_equity_research.ipynb
```

Programmatic use (also the entry point for Part 1B and Task 3):

```python
from task1_financial.src.pipeline import run_pipeline

result = run_pipeline("AAPL")
result.status          # "ok" | "degraded" | "failed"
result.summary_dict    # clean summary dictionary
result.prices          # OHLCV + indicator columns
result.news.headlines  # list of headline strings
```

Part 1B will need `GROQ_API_KEY` or `OPENROUTER_API_KEY` (Colab Secrets or `.env`).

## Notes and Limitations

- yfinance's news endpoint currently returns no items for AAPL, so headlines come from the Google News RSS search. Its parser still supports both yfinance news schemas in case the endpoint recovers.
- The snapshot in `data/` reflects the last successful run. When used, the pipeline reports `status="degraded"` and logs the snapshot timestamp.
- The 52-week range uses daily bars. Yahoo's figure can differ slightly on days with intraday extremes; differences above 2% are logged.

---

_Not investment advice. Produced for an engineering assessment only._
