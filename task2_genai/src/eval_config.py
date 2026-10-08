"""Task 2C evaluation settings: decoding, metrics, judge, and manual-review labels.

Every value has a one-line reason. This module does not import torch, so the
offline tests and the local review cells can use it without a GPU stack.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2C evaluation plan' (see CITATIONS.md Entry 12)

from pathlib import Path

from . import config

# Per-row files. The review and analysis cells read these instead of kernel state,
# so they can run locally after the Colab run.
EVAL_PREDICTIONS_PATH = config.OUTPUTS_DIR / "eval_predictions.jsonl"
EVAL_METRICS_PATH = config.OUTPUTS_DIR / "eval_metrics.json"
JUDGE_SCORES_PATH = config.OUTPUTS_DIR / "judge_scores.jsonl"
MANUAL_REVIEW_PATH = config.OUTPUTS_DIR / "manual_review.json"
EVAL_CHART_PATH = config.OUTPUTS_DIR / "eval_comparison.png"
JUDGE_PROMPT_PATH = config.PROMPTS_DIR / "judge_system.txt"

# Section 8 saves the merged model here on Colab. Used only if the Hub load fails.
MERGED_LOCAL_DIR = Path("/content/northwind-merged")

# Model keys used in every per-row file.
BASE_KEY = "base"
FINETUNED_KEY = "finetuned"
MODEL_KEYS = (BASE_KEY, FINETUNED_KEY)
MODEL_LABELS = {BASE_KEY: "Base + system prompt", FINETUNED_KEY: "Fine-tuned"}

# The longest gold answer is about 108 tokens. 256 leaves room without letting a
# rambling base model run for minutes.
EVAL_MAX_NEW_TOKENS = 256
# Greedy decoding, so both models are scored on one deterministic answer each.
EVAL_DO_SAMPLE = False
# Qwen's generation_config sets a repetition penalty. 1.0 turns it off, so decoding is plain
# greedy for both models and a JSON answer is not pushed away from repeating its own keys.
EVAL_REPETITION_PENALTY = 1.0

# rouge_score stems words so "approve" and "approved" match.
ROUGE_USE_STEMMER = True
# BERTScore's default English model. Named here so the run does not depend on a library default.
BERTSCORE_MODEL = "roberta-large"
BERTSCORE_BATCH_SIZE = 8

# Judges: neither is the teacher (gpt-oss-120b, which wrote the references) or the student (Qwen).
# Gemma is listed as the 2A backup teacher but wrote none of the dataset (fallback_calls is 0).
# Nemotron grades only rows Gemma could not, and each verdict records its grader. Groq's free
# catalogue only offers gpt-oss and Qwen models, so both judges run on OpenRouter.
JUDGE_MODEL_PRIMARY = "google/gemma-4-31b-it:free"
JUDGE_MODEL_BACKUP = "nvidia/nemotron-3-super-120b-a12b:free"
# Temperature 0 so a re-run grades the same answer the same way.
JUDGE_TEMPERATURE = 0.0
# Nemotron is a reasoning model and spends part of the budget before the JSON.
JUDGE_MAX_TOKENS = 4000
# OpenRouter's free tier allows about 20 requests a minute, and both judges share it.
# 3.5 s keeps 40 calls under that. Saved verdicts are reused, so a daily cap only delays rows.
JUDGE_MIN_INTERVAL_SECONDS = 3.5
JUDGE_SCORE_MAX = 2
JUDGE_CRITERIA = ("policy_correct", "action_faithful", "rationale_grounded", "no_invention")
JUDGE_TOTAL_MAX = JUDGE_SCORE_MAX * len(JUDGE_CRITERIA)

# Manual review. The spec asks for at least 10. All 20 test rows are reviewed.
LABEL_CORRECT = "correct"
LABEL_PARTIAL = "partially_correct"
LABEL_HALLUCINATED = "hallucinated"
LABELS = (LABEL_CORRECT, LABEL_PARTIAL, LABEL_HALLUCINATED)
MIN_REVIEWED = 10

# Judged against the policy manual, not only the gold label. A gold label can be wrong.
LABEL_RUBRIC = (
    ("correct", "The policy_ids are the right ones for the scenario under the manual, and the required "
     "action is one the clause allows. Wording may differ from the gold answer."),
    ("partially_correct", "A real, relevant policy is named, but the action is incomplete or slightly off, "
     "or one of two required ids is missing. Nothing is invented."),
    ("hallucinated", "An id that is not in the manual, an invented rule, limit, or deadline, a wrong policy "
     "stated as applying, NONE when a policy does apply, a policy cited for a case the manual does not "
     "cover, or no usable JSON answer."),
)

# Scenario text is cut to this many characters in the review table so it fits on screen.
REVIEW_SCENARIO_CHARS = 160
