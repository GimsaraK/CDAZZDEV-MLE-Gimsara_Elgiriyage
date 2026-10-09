# CDAZZDEV Senior Machine Learning Engineer - Technical Assessment

Submission by **Gimsara Elgiriyage** for the Senior Machine Learning Engineer technical assessment at Ceylon Dazzling Dev Holding (Pvt.) Ltd.

The assessment has three independent tasks covering Financial AI, Generative AI, and Agentic Workflows. Each task lives in its own sub-folder with a dedicated Colab notebook and README.

## Tasks

| Task | Domain | Folder | Notebook | Status |
|---|---|---|---|---|
| Task 1 | Financial AI - LLM-powered equity research assistant | [`task1_financial/`](task1_financial/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage/blob/main/task1_financial/task1_equity_research.ipynb) | 1A, 1B, and the research brief done locally |
| Task 2 | Generative AI - domain-specific QLoRA fine-tuning pipeline | [`task2_genai/`](task2_genai/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage/blob/main/task2_genai/task2_finetuning.ipynb) | 2A, 2B and 2C done: Colab T4 run saved with outputs, model on the Hub, evaluation, manual review and analysis written |
| Task 3 | Agentic Workflows - multi-agent financial research system | [`task3_agentic/`](task3_agentic/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage/blob/main/task3_agentic/task3_multi_agent.ipynb) | 3A done locally (fault-demo re-run pending); 3B and 3C (memory, cache, trace, Streamlit dashboard) implemented and tested offline; full live notebook run pending |

## Submission Links

| Deliverable | Link |
|---|---|
| GitHub repository | https://github.com/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage |
| Task 2 fine-tuned model (Hugging Face Hub) | [huggingface.co/GimsaraK/northwind-compliance-qwen2.5-1.5b](https://huggingface.co/GimsaraK/northwind-compliance-qwen2.5-1.5b) |
| Video walkthrough (optional) | _TBD_ |

## Repository Structure

```
CDAZZDEV-MLE-Gimsara_Elgiriyage/
|-- README.md                  # This file
|-- CITATIONS.md               # AI tool usage and adapted open-source code
|-- REFLECTION.md              # Architectural decisions, improvements, limitations (max 600 words)
|-- .env.example               # Template for required environment variables (no real keys)
|-- .gitignore
|
|-- task1_financial/           # Task 1 - Financial AI
|   |-- README.md
|   |-- requirements.txt
|   |-- task1_equity_research.ipynb
|   |-- prompts/               # LLM prompt templates (kept separate from business logic)
|   |-- src/                   # Data pipeline, indicators, LLM reasoning modules
|   `-- outputs/               # Rendered research brief, charts
|
|-- task2_genai/               # Task 2 - Generative AI
|   |-- README.md
|   |-- requirements.txt
|   |-- task2_finetuning.ipynb
|   |-- prompts/               # Teacher-model system prompt, evaluation / judge prompts
|   |-- data/                  # Generated dataset and train / val / test JSONL splits
|   |-- src/                   # Data generation, training, evaluation helpers
|   `-- outputs/               # Loss curves, evaluation results, diversity plots
|
`-- task3_agentic/             # Task 3 - Agentic Workflows
    |-- README.md
    |-- requirements.txt
    |-- task3_multi_agent.ipynb
    |-- src/                   # Tools, agents, memory, tracing
    |-- tests/                 # Offline tests (scripted models, synthetic data)
    |-- cache/                 # Persistent research briefs keyed by ticker and date
    |-- logs/                  # agent_trace.jsonl (every tool call, inputs, output, duration)
    `-- outputs/               # Final research reports
```

## Getting Started

### Option A - Google Colab (recommended)

1. Click the **Open In Colab** badge for the task you want to run.
2. Open the **Secrets** panel (key icon in the left sidebar) and add the keys listed in [`.env.example`](.env.example) that the task needs.
3. For Task 2, select a GPU runtime (`Runtime > Change runtime type > T4 GPU`).
4. Run all cells from top to bottom.

### Option B - Local

```bash
git clone https://github.com/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage.git
cd CDAZZDEV-MLE-Gimsara_Elgiriyage

python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

pip install -r task1_financial/requirements.txt   # or task2_genai / task3_agentic
cp .env.example .env                              # then fill in your own keys
jupyter notebook
```

Task 2 needs a CUDA GPU for QLoRA training, so Colab is the practical option there.

## Secrets and Credentials

No API keys or tokens are stored in this repository. Every notebook reads credentials at runtime from Colab Secrets or from a local `.env` file, and `.env` is excluded by `.gitignore`.

## Free-Tier Tooling

Everything runs on free tiers: Google Colab, Groq / OpenRouter free models, Hugging Face Hub, yfinance, LangChain / LangGraph / CrewAI, Weights & Biases (personal), and DuckDuckGo search through `ddgs` (the renamed `duckduckgo-search` package).

## Citations and Reflection

- [`CITATIONS.md`](CITATIONS.md) lists every instance of AI assistance and adapted open-source code.
- [`REFLECTION.md`](REFLECTION.md) covers architectural decisions, future improvements, and limitations across all tasks.

## Disclaimer

The financial analyses in this repository are produced for an engineering assessment only. They are not investment advice.
