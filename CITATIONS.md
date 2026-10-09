# Citations

This file documents every instance of AI assistance and every piece of adapted open-source code in this repository. Inline citations in the code use the same formats.

## Citation Formats

AI-generated or AI-assisted code:

```
# AI-ASSISTED: <Tool> (<model>), Prompt: '<prompt summary>'
```

Adapted open-source code:

```
# SOURCE: Adapted from <repository URL>, file: <file name>, Lines <start>-<end>
```

---

## 1. AI-Assisted Work - Prompt Log

Every prompt given to an AI assistant during this assessment, in chronological order. Prompts are quoted verbatim. Code produced from a prompt also carries an inline `# AI-ASSISTED:` comment referencing the same prompt.

### Entry 1 - Assessment analysis and requirements checklist

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Analyze the assessment specification in @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly and understand the requirements. I've attached it to this prompt. This pdf contains some instructions to follow such as Submission Requirements, Citations, Tasks to complete, marking rubric, etc. I want you to analyze this pdf thoroughly and make a checklist (md file) so that I can keep track of all the requirements without missing on any thing. Call AskUserQuestion if needed to clarify any issues.'
```

- Scope: General
- Output: a private requirements-tracking checklist. It is kept local-only and not committed, because it reproduces the confidential assessment brief.



### Entry 2 - Project structure scaffold

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Now with your understanding on the assessment requirements I need you to create the Project Structure with the relevant files needed to be pushed to GitHub-just the general project structure such as README.md, gitignore, per task sub folder, etc Call AskUserQuestion if needed'
```

- Scope: General
- Files: `README.md`, `.gitignore`, `.env.example`, `CITATIONS.md` (template), `REFLECTION.md` (template), `task1_financial/README.md`, `task2_genai/README.md`, `task3_agentic/README.md`, per-task `requirements.txt`, the notebook skeletons (section headings only), and empty `src/__init__.py` / `.gitkeep` placeholders.



### Entry 3 - Task 1A implementation planning

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Lets start implementing each of the tasks now - part by part. First read and understand Task 1 thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 1A. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.'
```

- Scope: Task 1
- Output: implementation plan for Task 1A, with 20 design decisions confirmed by the user (kept as a local-only planning document). No repository code changed.



### Entry 4 - Task 1A implementation

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md'
```

- Scope: Task 1
- Note: during implementation the user also chose (via a multiple-choice question) to filter institutional-holdings filing headlines and cap headlines at 3 per publisher.
- Files: `task1_financial/src/` (`config.py`, `logging_utils.py`, `errors.py`, `retry.py`, `indicators.py`, `data.py`, `news.py`, `summary.py`, `pipeline.py`), `task1_financial/tests/` (`conftest.py`, `test_indicators.py`, `test_data.py`, `test_news.py`, `test_summary.py`, `fixtures/`), `task1_financial/task1_equity_research.ipynb` (Part 1A cells), `task1_financial/requirements.txt`, `task1_financial/README.md`, `task1_financial/__init__.py`. Generated artefacts: `task1_financial/data/` (snapshot), `task1_financial/outputs/` (chart, summary JSON).
- Each module and notebook cell carries an inline `# AI-ASSISTED:` comment referencing this entry.



### Entry 5 - Task 1B implementation planning

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 1B. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.'
```

- Scope: Task 1
- Output: implementation plan for Task 1B, with the design decisions confirmed by the user (kept as a local-only planning document). No repository code changed.



### Entry 6 - Task 1B implementation

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Task 1B Implementation Plan - LLM Sentiment and Signal Reasoning (40 marks) Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself. To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.'
```

- Scope: Task 1
- Files: `task1_financial/src/prompts.py`, `task1_financial/src/llm.py`, `task1_financial/src/config.py`, `task1_financial/src/errors.py`, `task1_financial/src/pipeline.py`, `task1_financial/tests/test_llm.py`, `task1_financial/requirements.txt`, `task1_financial/task1_equity_research.ipynb` (Part 1B cells), `task1_financial/README.md`.
- Note: Groq retired `llama-3.3-70b-versatile` for free accounts, so the primary model is `openai/gpt-oss-120b`. OpenRouter has no free Llama 3.3 70B route, so the fallback model is `google/gemma-4-31b-it:free`. The local notebook run scored 15 headlines and returned Hold.



### Entry 7 - Task 2A implementation planning

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 2A. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.'
```

- Scope: Task 2
- Output: implementation plan for Task 2A (compliance-policy dataset). No repository code changed by that planning step.



### Entry 8 - Task 2A implementation

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Task 2A: Compliance Policy Dataset Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself. To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.'
```

- Scope: Task 2
- Files: `task2_genai/src/`, `task2_genai/tests/test_dataset.py`, `task2_genai/data/policy_manual.json`, `task2_genai/prompts/`, `task2_genai/data/train.jsonl`, `task2_genai/data/val.jsonl`, `task2_genai/data/test.jsonl`, `task2_genai/task2_finetuning.ipynb` (Part 2A cells and appendix A), `task2_genai/README.md`, `task2_genai/requirements.txt`.
- Note: Teacher `openai/gpt-oss-120b` on Groq. Student chat template `Qwen/Qwen2.5-1.5B-Instruct`. 200 accepted examples, split 160/20/20. OpenRouter is only the validation fallback.



### Entry 9 - Task 2B implementation planning

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 2B. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.'
```

- Scope: Task 2
- Output: implementation plan for Task 2B (QLoRA on a Colab T4). No repository code changed by that planning step.



### Entry 10 - Task 2B implementation

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Task 2B: QLoRA Fine-Tuning on Colab Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself. To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.'
```

- Scope: Task 2
- Files: `task2_genai/src/train_config.py`, `task2_genai/src/qlora.py`, `task2_genai/tests/test_train_config.py`, `task2_genai/task2_finetuning.ipynb` (sections 5-9), `task2_genai/README.md`.
- Note: Training is not run here. The notebook skips the GPU cells until it is executed on a Colab T4. Loss numbers and the Hub link stay open until that save.



### Entry 11 - Task 2C implementation planning

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 2C. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.'
```

- Scope: Task 2
- Output: implementation plan for Task 2C (evaluation and baseline comparison), with the design decisions confirmed by the user: BERTScore plus an LLM judge, Gemma 4 31B on OpenRouter as the judge with a gpt-oss-120b fallback, a four-criterion 0-2 rubric, canonical-JSON ROUGE-L, all 20 test answers reviewed by the user, and labels grounded in the policy manual. No repository code changed by that planning step.



### Entry 12 - Task 2C implementation

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2C evaluation plan' (the plan approved in Entry 11)
```

- Scope: Task 2
- Files: `task2_genai/src/eval_config.py`, `task2_genai/src/eval_metrics.py`, `task2_genai/src/judge.py`, `task2_genai/src/inference.py`, `task2_genai/src/prompts.py` (judge prompt), `task2_genai/src/client.py` (judge temperature and token budget, `build_judge_clients`), `task2_genai/prompts/judge_system.txt`, `task2_genai/tests/test_eval.py`, `task2_genai/task2_finetuning.ipynb` (sections 10-13, and the setup cell now puts the cloned repo on `sys.path` on Colab), `task2_genai/README.md`.
- Note: The backup judge was changed after planning from `openai/gpt-oss-120b` (the teacher) to `nvidia/nemotron-3-super-120b-a12b:free`, so no judge grades answers against references it wrote. Groq's free catalogue had only gpt-oss and Qwen models, so both judges run on OpenRouter.



### Entry 13 - Task 3A implementation planning

```
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 3A. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.'
```

- Scope: Task 3
- Output: implementation plan for Task 3A (tool-using research agent). No repository code changed by that planning step.



### Entry 14 - Task 3A implementation

```
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3A plan (the plan approved in Entry 13)'
```

- Scope: Task 3
- Files: `task3_agentic/__init__.py`, `task3_agentic/src/` (`config.py`, `schemas.py`, `session.py`, `tracing.py`, `logging_utils.py`, `volatility.py`, `tools.py`, `llm.py`, `prompts.py`, `report.py`, `agent.py`, `display.py`), `task3_agentic/tests/` (`conftest.py`, `test_volatility.py`, `test_tools.py`, `test_tracing.py`, `test_report.py`, `test_agent.py`), `task3_agentic/task3_multi_agent.ipynb` (sections 0-5), `task3_agentic/requirements.txt`, `task3_agentic/README.md`, root `README.md` (Task 3 status). Generated artefacts: `task3_agentic/logs/agent_trace.jsonl`, `task3_agentic/outputs/` (research reports).
- Note: `web_search` uses `ddgs`, the renamed `duckduckgo-search` package. The OpenRouter fallbacks (`google/gemma-4-31b-it:free`, then `nvidia/nemotron-3-super-120b-a12b:free`) were picked from the models list filtered on `:free` and `tools` support. Two checks were added after the first live runs: an explicit no-figures-from-memory rule for the report writer, and a rule that every tool that only failed must appear in `data_gaps`.



### Entry 15 - Task 3B implementation planning

```
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 3B. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.'
```

- Scope: Task 3
- Output: implementation plan for Task 3B (two-agent pipeline). No repository code changed by that planning step.



### Entry 16 - Task 3B implementation

```
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3B plan'
```

- Scope: Task 3
- Files: new `task3_agentic/src/react.py`, `task3_agentic/src/handoff.py`, `task3_agentic/src/multi_agent.py`, `task3_agentic/tests/test_handoff.py`, `task3_agentic/tests/test_multi_agent.py`. Edited: `task3_agentic/src/agent.py` (loop moved to `react.py`), `tools.py` (per-agent toolkits, headline guard), `tracing.py` (agent attribution, `ToolAccessError` guard), `session.py` (per-agent records, `known_headlines`), `config.py`, `prompts.py`, `schemas.py` (`clarification_used`, `agent_tools`), `report.py` (writer digest, provenance lines, call-time output folder), `display.py` (`MultiAgentPrinter`, handoff and schema tables), the tests that score headlines (they now fetch news first), `task3_agentic/task3_multi_agent.ipynb` (sections 6-9; headings 8 and 9 swapped), `task3_agentic/README.md`, root `README.md`.



### Entry 17 - Task 3C implementation planning

```
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 3C. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.'
```

- Scope: Task 3
- Output: implementation plan for Task 3C (memory and observability). No repository code changed by that planning step.



### Entry 18 - Task 3C implementation

```
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 3C plan'
```

- Scope: Task 3
- Files: new `task3_agentic/src/memory.py`, `task3_agentic/src/cache.py`, `task3_agentic/src/dashboard_data.py`, `task3_agentic/dashboard/app.py`, `task3_agentic/tests/test_memory.py`, `task3_agentic/tests/test_cache.py`, `task3_agentic/tests/test_observability.py`. Edited: `task3_agentic/src/agent.py` (checkpointer thread, follow-up mode), `tracing.py` (`EventLogger`, `audit_trace`, `trace_line_count`), `config.py`, `prompts.py` (`FOLLOWUP`, `FOLLOWUP_ANSWER_NOW`), `display.py` (follow-up and cache tables), `task3_agentic/task3_multi_agent.ipynb` (events-log cell after section 1, cache wrapper in section 8, sections 10-13), `task3_agentic/requirements.txt` (`streamlit`), `task3_agentic/README.md`, root `README.md`.
- Note: The first tests found a gap in the follow-up budget: an agent that used its one tool round had no turn left to answer. The follow-up now gets its tool round plus one answer turn. A tool request in that last turn is dropped, not left unanswered in the thread.
- Note: Verified offline only: 21 new tests, every 3B and 3C notebook cell dry-run with scripted models (paths redirected to the scratchpad), and the dashboard smoke-tested headless with `streamlit.testing` against the real trace. Live notebook output for sections 6-13, the committed cache sample and the dashboard screenshot wait on fresh Groq quota.



### Entry 19 - Task 2 bonus (RAG fallback layer) implementation planning

```
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 2 Bonus RAG Fallback Layer. Make sure all requirements mentioned in the specification for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.', Date: 2026-10-09
```

- Scope: Task 2
- Output: implementation plan for the Task 2 bonus RAG fallback layer, with the design decisions confirmed by the user:
  - the trigger is answer perplexity, with the threshold calibrated on the validation split (Youden's J);
  - the ChromaDB store holds the 16 policy-manual clauses, uses Chroma's default MiniLM embeddings, and is rebuilt in memory;
  - retrieval takes the top 3 policies, and their excerpts go in the user turn;
  - the RAG answer is kept whenever the fallback fires;
  - evaluation compares fine-tuned only, gated RAG and always-RAG on the 20 test rows, with deterministic metrics only;
  - the notebook shows one before-and-after example, chosen by a stated rule;
  - the section runs standalone on a Colab T4.
  No repository code changed by that planning step.



### Entry 20 - Task 2 bonus (RAG fallback layer) implementation

```
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 2 RAG fallback plan', Date: 2026-10-09
```

- Scope: Task 2
- Files: new `task2_genai/src/rag.py`, `task2_genai/tests/test_rag.py`. Edited: `task2_genai/src/inference.py` (`generate_scored`, `perplexity_from_logprobs`), `task2_genai/src/eval_config.py` (RAG constants and paths), `task2_genai/src/prompts.py` (`RAG_USER_TEMPLATE`, `RAG_PROMPT_VARIANTS`), `task2_genai/src/eval_metrics.py` (`keys` argument on `score_predictions`), `task2_genai/task2_finetuning.ipynb` (Bonus part, section 14), `task2_genai/requirements.txt` (`chromadb`), `task2_genai/README.md`, root `README.md`.
- Note: The first Colab run used one re-query template ending in "answer ["NONE"] if none applies", and the model answered NONE on all 20 test rows; its numbers are kept in `task2_genai/outputs/rag_attempt1_summary.json`. With the user's approval, the re-query prompt is now chosen on the validation rows from four layouts (`select_variant`), before the test rows are run.

---



## 2. Adapted Open-Source Code


| Source (URL)                                                                                                                                                                                                                                                                         | File                                        | Lines                             | Used in                                                                                           | Notes                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi) | RSI worked-example spreadsheet (cs-rsi.xls) | Closing prices and RSI(14) column | `task1_financial/tests/test_indicators.py` (`STOCKCHARTS_CLOSES`, `STOCKCHARTS_RSI_FIRST_VALUES`) | Published reference data used as a test oracle; inline `# SOURCE:` comment in the test file |


---



## 3. Teacher-Model Data Generation (Task 2)


| Item               | Value                                                                                                                                                                                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Teacher model      | `openai/gpt-oss-120b`                                                                                                                                                                                                                                                          |
| Provider           | Groq (OpenRouter `google/gemma-4-31b-it:free` only if a Groq response fails validation)                                                                                                                                                                                        |
| Full system prompt | `[task2_genai/prompts/teacher_system.txt](task2_genai/prompts/teacher_system.txt)` and appendix A of `[task2_genai/task2_finetuning.ipynb](task2_genai/task2_finetuning.ipynb)`                                                                                                |
| Task 2C judge      | `google/gemma-4-31b-it:free` on OpenRouter, with `nvidia/nemotron-3-super-120b-a12b:free` (OpenRouter) only for rows Gemma could not grade. Neither is the teacher or the student. Full prompt: `[task2_genai/prompts/judge_system.txt](task2_genai/prompts/judge_system.txt)` |


---



## 4. Other References

Documentation, papers, or tutorials consulted closely:

- J. Welles Wilder Jr., *New Concepts in Technical Trading Systems* (1978): RSI definition and Wilder smoothing (`task1_financial/src/indicators.py`).
- G. Appel, *Technical Analysis: Power Tools for Active Investors* (2005): MACD definition.
- J. Bollinger, *Bollinger on Bollinger Bands* (2001): Bollinger Bands with population standard deviation.
- StockCharts ChartSchool articles on RSI, MACD and Bollinger Bands: formula cross-reference.
- `ta` library, [https://github.com/bukosabino/ta](https://github.com/bukosabino/ta): used **only** in the Task 1 notebook's validation cell, to cross-check our indicator implementations. It is not used by the pipeline.
- yfinance documentation, [https://ranaroussi.github.io/yfinance/](https://ranaroussi.github.io/yfinance/): `Ticker.history`, `Ticker.info`, `Ticker.news`.
- tenacity documentation, [https://tenacity.readthedocs.io/](https://tenacity.readthedocs.io/): retry policy (`Retrying`, `wait_random_exponential`).
- Google News RSS search endpoint (`news.google.com/rss/search`): free, keyless headline source.
- Groq OpenAI-compatible chat API, [https://console.groq.com/docs/openai](https://console.groq.com/docs/openai), and deprecations, [https://console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations): primary LLM endpoint for Task 1B (`openai/gpt-oss-120b`, the replacement after `llama-3.3-70b-versatile` was retired).
- OpenRouter models list, [https://openrouter.ai/api/v1/models](https://openrouter.ai/api/v1/models): used to pick a free fallback model (`google/gemma-4-31b-it:free`) after confirming Llama 3.3 70B is paid there.
- C.-Y. Lin, *ROUGE: A Package for Automatic Evaluation of Summaries* (2004), and the `rouge-score` package, [https://github.com/google-research/google-research/tree/master/rouge](https://github.com/google-research/google-research/tree/master/rouge): ROUGE-L F-measure in Task 2C.
- T. Zhang et al., *BERTScore: Evaluating Text Generation with BERT* (ICLR 2020), and the `bert-score` package, [https://github.com/Tiiiger/bert_score](https://github.com/Tiiiger/bert_score): BERTScore F1 in Task 2C.
- LangGraph documentation, [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/): `StateGraph`, `add_messages`, conditional edges, `ToolNode` and `stream_mode="values"` in Task 3 (`task3_agentic/src/agent.py`).
- LangChain documentation, [https://python.langchain.com/](https://python.langchain.com/): `StructuredTool` with `response_format="content_and_artifact"`, `bind_tools`, `with_structured_output` (`method="json_schema"` on Groq), and the `Runnable` interface used by `FallbackChain` (`task3_agentic/src/tools.py`, `llm.py`).
- S. Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR 2023): the reason-act-observe loop the Task 3 agent follows.
- `ddgs` package (renamed from `duckduckgo-search`), [https://github.com/deedy5/ddgs](https://github.com/deedy5/ddgs): `DDGS.text` / `DDGS.news` arguments and the `RatelimitException` / `TimeoutException` / `DDGSException` types, read from the installed source (`task3_agentic/src/tools.py`).
- Close-to-close historical volatility (sample standard deviation of daily log returns, annualised with sqrt(252)), as in J. C. Hull, *Options, Futures, and Other Derivatives*, chapter on estimating volatility from historical data (`task3_agentic/src/volatility.py`).

