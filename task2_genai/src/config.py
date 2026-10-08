"""Constants for the Task 2A compliance dataset.

The teacher writes the rows. The student id is recorded here only so the saved
JSONL uses that model's chat template. Training is Task 2B and is not run here.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parent
DATA_DIR = TASK_DIR / "data"
PROMPTS_DIR = TASK_DIR / "prompts"
OUTPUTS_DIR = TASK_DIR / "outputs"
POLICY_MANUAL_PATH = DATA_DIR / "policy_manual.json"
ACCEPTED_PATH = DATA_DIR / "accepted.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
TEACHER_PROMPT_PATH = PROMPTS_DIR / "teacher_system.txt"
TEACHER_USER_PATH = PROMPTS_DIR / "teacher_user.txt"
STUDENT_PROMPT_PATH = PROMPTS_DIR / "student_system.txt"
DIVERSITY_REPORT_PATH = OUTPUTS_DIR / "diversity_report.json"
GENERATION_REPORT_PATH = OUTPUTS_DIR / "generation_report.json"

# Teacher and student must differ. Groq retired Llama 3.3 70B for free accounts
# on 2026-08-16; gpt-oss-120b is the documented replacement and already used in Task 1.
TEACHER_MODEL_GROQ = "openai/gpt-oss-120b"
TEACHER_MODEL_OPENROUTER = "google/gemma-4-31b-it:free"
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

LLM_PROVIDER_GROQ = "groq"
LLM_PROVIDER_OPENROUTER = "openrouter"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ENV_GROQ_API_KEY = "GROQ_API_KEY"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/GimsaraK/CDAZZDEV-MLE-Gimsara_Elgiriyage",
    "X-Title": "CDAZZDEV MLE Task 2",
}

# 0.7, not 0, so the five scenarios in one cell are not paraphrases of each other.
TEACHER_TEMPERATURE = 0.7
# gpt-oss spends part of this budget on a hidden reasoning trace before the JSON.
TEACHER_MAX_TOKENS = 8000
TEACHER_TIMEOUT_SECONDS = 120
TEACHER_REPAIR_ATTEMPTS = 1
EXAMPLES_PER_CALL = 5
# Fresh calls per cell after the first, if dedup or schema drops leave the cell short.
MAX_CALLS_PER_CELL = 4

RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_MIN_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 20.0

# Accepted rows, then the 80/10/10 split. 8 topics x 5 situations x 5 rows = 200.
TARGET_ACCEPTED = 200
TARGET_PER_CELL = 5
TRAIN_SIZE = 160
VAL_SIZE = 20
TEST_SIZE = 20
SPLIT_SEED = 42

# Scenario length gate. The user message asks each situation for a band inside this range.
MIN_SCENARIO_WORDS = 40
MAX_SCENARIO_WORDS = 180

# Character 5-gram Jaccard. At or above this, the new scenario is a near-duplicate.
NEAR_DUP_NGRAM = 5
NEAR_DUP_JACCARD = 0.80

NONE_POLICY_ID = "NONE"
SITUATIONS = ("clear_breach", "borderline", "compliant", "not_covered", "two_policy")

# Asked of the teacher so the length histogram is not a single spike.
LENGTH_HINTS = {
    "clear_breach": "40 to 70 words",
    "borderline": "70 to 110 words",
    "compliant": "50 to 90 words",
    "not_covered": "60 to 100 words",
    "two_policy": "120 to 180 words",
}

# Small English list for the keyword chart. Not a general NLP stopword list.
STOPWORDS = frozenset(
    """
    a an the and or but if then of to for from in on at by with without into over
    is are was were be been being it this that these those he she they them his her
    their its as not no nor so than too very can could should would will just about
    after before during while who whom which what when where how has have had do does
    did
    """.split()
)
