# Reflection

Word count: 588 / 600

## Architectural Decisions

**Shared principle: never trust a single source or a single model call.** Every external dependency has a fallback, and every LLM output is validated with Pydantic before use.

**Task 1.** Indicators are computed from first principles (Wilder RSI) and cross-checked against the `ta` library. Market data falls back to a committed snapshot and news falls back from yfinance to Google News RSS, so a network error never stops the pipeline. The LLM stage calls Groq `gpt-oss-120b`. An invalid response gets one repair call with the validation error, then goes to OpenRouter. Headline scores are combined as a confidence-weighted mean.

**Task 2.** I chose a closed-book compliance assistant on a fictional policy manual, so correctness is checkable: the right policy id and an action the clause allows. The 200 examples come from an 8-topic x 5-situation taxonomy, deduplicated, with every topic in every split. The teacher (`gpt-oss-120b`) differs from the student (Qwen2.5-1.5B). QLoRA uses NF4, rank 16 on all seven projection modules, and three epochs, and the adapter is merged onto a float16 base. Evaluation combines ROUGE-L, BERTScore, an independent LLM judge and a manual review. The bonus RAG fallback re-queries only when an answer's perplexity passes a threshold calibrated on validation, retrieving manual clauses from ChromaDB by dense (MiniLM) search.

**Task 3.** I used LangGraph because the control flow (tool loop, report check, replan) should be explicit and testable. Tools return a typed envelope with a hint instead of raising, so the agent can recover. A deterministic check rejects reports whose evidence cites failed tools or whose hedge numbers do not match `calculate_volatility`. In 3B, tool restriction is enforced in code at three layers, handoffs are Pydantic models, and the one critique round carries B's headlines to A. Memory uses a LangGraph checkpointer; the cache is keyed by ticker, market date and pipeline.

## What I Would Improve With More Time

- **Task 1:** backtest the momentum rule and the Buy/Hold/Sell signal against forward returns before trusting either.
- **Task 2:** the 45% hallucination rate is the main gap: confused look-alike clauses, leaked actions and invented details. I would add contrastive pairs of similar clauses, verify teacher labels by hand, build a larger human-checked test set, and fine-tune with retrieved excerpts plus distractor policies in the prompt (RAFT-style) so the model learns to use context.
- **Task 3:** replace the fixed handoff order with a supervisor that decides when to hand off, allow more than one critique round, and move memory to a persistent checkpointer.

## Limitations Encountered

- **The RAG fallback did not help.** Perplexity flagged hallucinations well (6 of the 7 flagged answers), but the model, trained only on scenario-only prompts, answered `NONE` when given policy text, so exact ids fell from 80% to 65%. It is reported as a measured negative result.
- **Free-tier quotas shaped the design.** Groq's daily token cap and OpenRouter's 50 free requests per day ended several full runs midway. Hence the model fallback chain, low reasoning effort for agent turns, and a graceful `failed` status instead of exceptions. Gemma was rate limited on every judge call, so Nemotron graded all Task 2 answers.
- **Data sources are unofficial.** yfinance returned no news, so headlines come from Google News RSS, and the final Task 1 price is intraday.
- **Small evaluation sets.** Twenty test rows give wide uncertainty on every Task 2 metric, and the dataset reflects one teacher's style.
- **LLM non-determinism.** Tool order, sentiment scores and report wording change between runs. Each notebook shows one recorded run; the checks validate structure and numbers, not judgement.
