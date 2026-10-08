# Citations

This file documents every instance of AI assistance and every piece of adapted open-source code in this repository. Inline citations in the code use the same formats.

## Citation Formats

AI-generated or AI-assisted code:

```
# AI-ASSISTED: <Tool> (<model>), Prompt: '<prompt summary>', Date: YYYY-MM-DD
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
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Analyze the assessment specification in @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly and understand the requirements. I've attached it to this prompt. This pdf contains some instructions to follow such as Submission Requirements, Citations, Tasks to complete, marking rubric, etc. I want you to analyze this pdf thoroughly and make a checklist (md file) so that I can keep track of all the requirements without missing on any thing. Call AskUserQuestion if needed to clarify any issues.', Date: 2026-10-07
```
- Scope: General
- Output: a private requirements-tracking checklist. It is kept local-only and not committed, because it reproduces the confidential assessment brief.

### Entry 2 - Project structure scaffold

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Now with your understanding on the assessment requirements I need you to create the Project Structure with the relevant files needed to be pushed to GitHub-just the general project structure such as README.md, gitignore, per task sub folder, etc Call AskUserQuestion if needed', Date: 2026-10-07
```
- Scope: General
- Files: `README.md`, `.gitignore`, `.env.example`, `CITATIONS.md` (template), `REFLECTION.md` (template), `task1_financial/README.md`, `task2_genai/README.md`, `task3_agentic/README.md`, per-task `requirements.txt`, the notebook skeletons (section headings only), and empty `src/__init__.py` / `.gitkeep` placeholders.

### Entry 3 - Task 1A implementation planning

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Now analyze the specification again @CDAZZDEV_Senior_MLE_Assessment_2026.pdf thoroughly. Lets start implementing each of the tasks now - part by part. First read and understand Task 1 thoroughly. Then call EnterPlanMode to create a comprehensive plan for Task 1A. Make sure all requirements in the @SUBMISSION_CHECKLIST.md for this task is covered in the plan including the marking rubric for this task. Call AskUserQuestion for all architectural judgment and engineering decisions.', Date: 2026-10-07
```
- Scope: Task 1
- Output: implementation plan for Task 1A, with 20 design decisions confirmed by the user (kept as a local-only planning document). No repository code changed.

### Entry 4 - Task 1A implementation

```
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08
```
- Scope: Task 1
- Note: during implementation the user also chose (via a multiple-choice question) to filter institutional-holdings filing headlines and cap headlines at 3 per publisher.
- Files: `task1_financial/src/` (`config.py`, `logging_utils.py`, `errors.py`, `retry.py`, `indicators.py`, `data.py`, `news.py`, `summary.py`, `pipeline.py`), `task1_financial/tests/` (`conftest.py`, `test_indicators.py`, `test_data.py`, `test_news.py`, `test_summary.py`, `fixtures/`), `task1_financial/task1_equity_research.ipynb` (Part 1A cells), `task1_financial/requirements.txt`, `task1_financial/README.md`, `task1_financial/__init__.py`. Generated artefacts: `task1_financial/data/` (snapshot), `task1_financial/outputs/` (chart, summary JSON).
- Each module and notebook cell carries an inline `# AI-ASSISTED:` comment referencing this entry.

---

## 2. Adapted Open-Source Code

| Source (URL) | File | Lines | Used in | Notes |
|---|---|---|---|---|
| https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi | RSI worked-example spreadsheet (cs-rsi.xls) | Closing prices and RSI(14) column | `task1_financial/tests/test_indicators.py` (`STOCKCHARTS_CLOSES`, `STOCKCHARTS_RSI_FIRST_VALUES`) | Published reference data used as a test oracle; inline `# SOURCE:` comment in the test file |

---

## 3. Teacher-Model Data Generation (Task 2)

| Item | Value |
|---|---|
| Teacher model | _TBD_ |
| Provider | _TBD_ |
| Full system prompt | See [`task2_genai/prompts/`](task2_genai/prompts/) and the appendix cell of [`task2_genai/task2_finetuning.ipynb`](task2_genai/task2_finetuning.ipynb) |

---

## 4. Other References

Documentation, papers, or tutorials consulted closely:

- J. Welles Wilder Jr., *New Concepts in Technical Trading Systems* (1978): RSI definition and Wilder smoothing (`task1_financial/src/indicators.py`).
- G. Appel, *Technical Analysis: Power Tools for Active Investors* (2005): MACD definition.
- J. Bollinger, *Bollinger on Bollinger Bands* (2001): Bollinger Bands with population standard deviation.
- StockCharts ChartSchool articles on RSI, MACD and Bollinger Bands: formula cross-reference.
- `ta` library, https://github.com/bukosabino/ta: used **only** in the Task 1 notebook's validation cell, to cross-check our indicator implementations. It is not used by the pipeline.
- yfinance documentation, https://ranaroussi.github.io/yfinance/: `Ticker.history`, `Ticker.info`, `Ticker.news`.
- tenacity documentation, https://tenacity.readthedocs.io/: retry policy (`Retrying`, `wait_random_exponential`).
- Google News RSS search endpoint (`news.google.com/rss/search`): free, keyless headline source.
