# Task 3 - Agentic Workflows: Multi-Agent Financial Research System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage/blob/main/task3_agentic/task3_multi_agent.ipynb)

A tool-using research agent extended into a two-agent pipeline (Data Analyst and Research Writer), with structured handoffs, a critique loop, short-term and persistent memory, and full tool-call observability.

**Research query:**
> Analyse the current financial health and market sentiment of [TICKER]. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy.

## Contents

| Path | Description |
|---|---|
| `task3_multi_agent.ipynb` | Main notebook, executed with the full agent trace visible |
| `src/` | Tools, agents, handoff schemas, memory, and tracing |
| `cache/` | Persistent research briefs keyed by ticker and date (JSON) |
| `logs/agent_trace.jsonl` | Every tool call with inputs, output (truncated to 200 chars), and duration |
| `outputs/` | Final research reports |
| `requirements.txt` | Python dependencies |

## Configuration

| Setting | Value |
|---|---|
| Agent framework | _TBD (LangChain / LangGraph / CrewAI)_ |
| LLM provider / model | _TBD_ |
| Ticker | _TBD_ |

## Tools

| Tool | Description |
|---|---|
| `get_price_data(ticker, period)` | yfinance OHLCV with computed indicators |
| `get_news(ticker, n)` | Recent headlines as a structured list |
| `calculate_volatility(ticker, window)` | Annualised historical volatility |
| `llm_sentiment(headlines)` | LLM-based structured sentiment score |
| `web_search(query)` | Analyst commentary via duckduckgo-search |

## Architecture

### 3A - Single research agent

_TBD_

### 3B - Multi-agent pipeline

| Agent | Role | Tool access | Output |
|---|---|---|---|
| Agent A - Data Analyst | Quantitative analysis | `get_price_data`, `calculate_volatility`, `llm_sentiment` | Structured JSON data brief |
| Agent B - Research Writer | Qualitative synthesis | `web_search`, `get_news` | Final research report |

Handoff schema, critique loop design, and the architecture diagram: _TBD_

### 3C - Memory and observability

- **Short-term memory:** _TBD_
- **Persistent cache:** `cache/<TICKER>_<YYYY-MM-DD>.json`
- **Trace log:** `logs/agent_trace.jsonl`, one JSON object per tool call

## Bonus - Observability Platform

_TBD: LangSmith run screenshot or Streamlit trace dashboard._

## How to Run

1. Open the notebook with the Colab badge above.
2. Add `GROQ_API_KEY` or `OPENROUTER_API_KEY` (and optionally `LANGCHAIN_API_KEY` for LangSmith) in the Colab Secrets panel.
3. Run all cells.

---

_Not investment advice. Produced for an engineering assessment only._
