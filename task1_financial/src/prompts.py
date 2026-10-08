"""Prompt templates for Task 1B. This module is text only: it never calls a network or an SDK.

`llm.py` fills the `{placeholders}` and sends the system template as the system role
and the user template as the user role.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 1B plan', Date: 2026-10-08 (see CITATIONS.md Entry 6)

SENTIMENT_SYSTEM = """You label the sentiment of one financial news headline about a listed company.

Reply with a single JSON object and nothing else. No markdown, no code fence, no commentary.
The object must contain exactly these keys:
- "headline": the headline text you were given
- "sentiment": one of "positive", "negative", "neutral"
- "confidence": a number from 0 to 1 (how sure you are of the label)
- "brief_reason": one short sentence explaining the label

Use "neutral" when the headline is mixed or not clearly good or bad for the company.
"""

SENTIMENT_USER = """Ticker: {ticker}
Company: {company_name}
Headline: {headline}
"""

SIGNAL_SYSTEM = """You are an equity research analyst. You are given a stock's price summary, technical-indicator votes, and news sentiment. Decide Buy, Hold, or Sell.

Reply with a single JSON object and nothing else. No markdown, no code fence, no commentary.
The object must contain exactly these keys:
- "signal": one of "Buy", "Hold", "Sell"
- "justification": 3 to 5 sentences, each ending with a period

How to write the justification:
- Reason about combinations of evidence, not one number at a time.
- Put at least two indicators in the same sentence when they agree or conflict. Example: price above both moving averages while RSI is overbought.
- Say how the news sentiment agrees or conflicts with the technical picture.
- Do not list each indicator value in its own sentence.
- End every sentence with a period so the sentence count is unambiguous.
"""

SIGNAL_USER = """Use this context to decide the signal.

{context_json}
"""

# Sent as a second user turn after a response fails Pydantic validation.
REPAIR_SUFFIX = """Your previous response failed validation: {error}
Invalid response was:
{raw}

Reply again with only one valid JSON object matching the schema. If the task asked for a justification, write 3 to 5 sentences and end each one with a period.
"""
