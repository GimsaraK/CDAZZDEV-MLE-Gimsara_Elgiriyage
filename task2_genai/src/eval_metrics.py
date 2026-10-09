"""Task 2C scoring: answer parsing, ROUGE-L, BERTScore, domain checks, and the manual review.

Not named evaluate.py, which would shadow the Hugging Face `evaluate` package.
rouge_score, bert_score, and matplotlib are imported inside functions, so the
local review cells and the offline tests need none of them.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2C evaluation plan' (see CITATIONS.md Entry 12)

import json
from collections import Counter
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from . import eval_config as ec
from .generate import _read_jsonl, _strip_fence, _write_jsonl

ANSWER_FIELDS = ("policy_ids", "rationale", "required_action")


# ---------------------------------------------------------------- test rows and predictions


def load_test_rows() -> List[dict]:
    """The held-out split. Only Task 2C reads this file. The trainer never does."""
    return _read_jsonl(config.TEST_PATH)


def gold_answer(row: dict) -> str:
    """The assistant turn stored in the test row. It is already canonical JSON."""
    return row["messages"][-1]["content"]


def build_prediction_records(test_rows: List[dict], outputs: Dict[str, List[str]], ft_source: Optional[str]) -> List[dict]:
    """One record per test row with the gold answer and each model's raw output."""
    records = []
    for index, row in enumerate(test_rows):
        record = {
            "id": index,
            "topic": row["topic"],
            "situation": row["situation"],
            "scenario": row["messages"][1]["content"],
            "gold": gold_answer(row),
            "ft_source": ft_source,
        }
        for key in ec.MODEL_KEYS:
            if key in outputs:
                record[key] = outputs[key][index]
        records.append(record)
    return records


def write_predictions(records: List[dict], path=None) -> None:
    _write_jsonl(path or ec.EVAL_PREDICTIONS_PATH, records)


def read_predictions(path=None) -> List[dict]:
    return _read_jsonl(path or ec.EVAL_PREDICTIONS_PATH)


# ---------------------------------------------------------------- answer parsing


def parse_answer(text: str) -> Tuple[Optional[dict], bool]:
    """(answer dict or None, strict_ok).

    strict_ok means the whole output is one JSON object, which is what the system
    prompt asks for. If that fails, a fenced or chatty answer is still read from
    its first {...} block so its fields can be scored.
    """
    raw = (text or "").strip()
    # Pass 1 (strict): the whole reply must be exactly one JSON object.
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload, True
    except json.JSONDecodeError:
        pass
    # Pass 2 (lenient): drop a ``` fence, then take the text from the first "{" to the last "}".
    # This rescues answers like 'Here is the answer: {...}' so their fields can still be scored,
    # while strict_ok=False records that the format instruction was not followed.
    candidate = _strip_fence(raw)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None, False
    try:
        payload = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return None, False
    return (payload, False) if isinstance(payload, dict) else (None, False)


def answer_ids(answer: Optional[dict]) -> List[str]:
    """policy_ids as a list of strings. A bare string id counts as a one-item list."""
    if not answer:
        return []
    ids = answer.get("policy_ids")
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, list):
        return []
    return [str(item).strip() for item in ids if str(item).strip()]


def answer_field(answer: Optional[dict], field: str) -> str:
    if not answer:
        return ""
    value = answer.get(field)
    return value.strip() if isinstance(value, str) else ""


def canonical_answer(text: str) -> str:
    """Sorted-key compact JSON when the output parses, the stripped text otherwise.

    The gold answers use the same form (format.assistant_json), so key order and
    whitespace do not change ROUGE-L or BERTScore.
    """
    answer, _ = parse_answer(text)
    if answer is None:
        return (text or "").strip()
    payload = {
        "policy_ids": answer_ids(answer),
        "rationale": answer_field(answer, "rationale"),
        "required_action": answer_field(answer, "required_action"),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


# ---------------------------------------------------------------- per-row checks


def domain_metrics(text: str, gold_text: str, manual_ids: set, situation: str) -> Dict:
    """Deterministic task checks. These do not depend on any model or library."""
    answer, strict_ok = parse_answer(text)
    gold, _ = parse_answer(gold_text)
    pred_ids = answer_ids(answer)
    gold_ids = answer_ids(gold)
    allowed = set(manual_ids) | {config.NONE_POLICY_ID}
    result = {
        "json_valid": strict_ok,
        "parsed": answer is not None,
        # Same ids in any order, and no duplicates (the length check catches ["P1", "P1"] vs ["P1"]).
        "policy_exact": answer is not None and set(pred_ids) == set(gold_ids) and len(pred_ids) == len(gold_ids),
        "unknown_id": any(policy_id not in allowed for policy_id in pred_ids),
        "none_correct": None,
    }
    # Only the not-covered rows test whether the model admits no rule applies.
    if situation == "not_covered":
        result["none_correct"] = pred_ids == [config.NONE_POLICY_ID]
    return result


def rouge_l(candidates: Sequence[str], references: Sequence[str]) -> List[float]:
    """ROUGE-L F-measure per pair."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=ec.ROUGE_USE_STEMMER)
    # ROUGE-L is based on the longest common subsequence of words. Note the argument order:
    # rouge_score expects score(reference, candidate).
    return [scorer.score(ref, cand)["rougeL"].fmeasure for cand, ref in zip(candidates, references)]


def bertscore_f1(candidates: Sequence[str], references: Sequence[str]) -> List[float]:
    """BERTScore F1 per pair. An empty answer is scored as a placeholder word, not skipped."""
    from bert_score import score

    cands = [cand if cand.strip() else "(empty)" for cand in candidates]
    _, _, f1 = score(
        cands,
        list(references),
        model_type=ec.BERTSCORE_MODEL,
        batch_size=ec.BERTSCORE_BATCH_SIZE,
        verbose=False,
    )
    return [float(value) for value in f1.tolist()]


def score_predictions(records: List[dict], manual_ids: set, rouge=rouge_l, keys: Sequence[str] = ec.MODEL_KEYS) -> Dict[str, List[dict]]:
    """Per-row ROUGE-L (whole answer and two fields) and domain checks for each model.

    `keys` names the answer fields to score: the two 2C models by default; the RAG bonus passes its three
    pipelines, so both sections are scored by exactly the same code.
    """
    scores = {}
    for key in keys:
        rows = [record for record in records if key in record]
        if not rows:
            continue
        outputs = [record[key] for record in rows]
        golds = [record["gold"] for record in rows]
        parsed = [parse_answer(text)[0] for text in outputs]
        gold_parsed = [parse_answer(text)[0] for text in golds]
        # Three ROUGE-L views per row: the whole answer (in canonical JSON, so key order and spacing
        # cannot change the score), and the two free-text fields compared on their own.
        whole = rouge([canonical_answer(text) for text in outputs], golds)
        action = rouge(
            [answer_field(answer, "required_action") for answer in parsed],
            [answer_field(answer, "required_action") for answer in gold_parsed],
        )
        rationale = rouge(
            [answer_field(answer, "rationale") for answer in parsed],
            [answer_field(answer, "rationale") for answer in gold_parsed],
        )
        per_row = []
        for index, record in enumerate(rows):
            row = {
                "id": record["id"],
                "rouge_l": whole[index],
                "rouge_l_action": action[index],
                "rouge_l_rationale": rationale[index],
            }
            row.update(domain_metrics(record[key], record["gold"], manual_ids, record["situation"]))
            per_row.append(row)
        scores[key] = per_row
    return scores


def add_bertscore(scores: Dict[str, List[dict]], records: List[dict], scorer=bertscore_f1) -> None:
    """Attach BERTScore F1 to each per-row dict, using the same canonical text as ROUGE-L."""
    by_id = {record["id"]: record for record in records}
    for key, rows in scores.items():
        cands = [canonical_answer(by_id[row["id"]][key]) for row in rows]
        refs = [by_id[row["id"]]["gold"] for row in rows]
        for row, value in zip(rows, scorer(cands, refs)):
            row["bertscore_f1"] = value


# ---------------------------------------------------------------- summary table


def _pct(values: List[bool]) -> Optional[float]:
    return 100.0 * sum(1 for value in values if value) / len(values) if values else None


def summarize(scores: Dict[str, List[dict]], judge_rows: Optional[List[dict]] = None) -> Dict[str, Dict]:
    """Model -> metric -> value. Missing metrics are None, never zero."""
    summary = {}
    # Group the judge results by model ("base" / "finetuned") so each model's means use only its own rows.
    judged_by_model = {}
    for row in judge_rows or []:
        judged_by_model.setdefault(row["model"], []).append(row)
    for key, rows in scores.items():
        entry = {
            "n": len(rows),
            "rouge_l": mean(row["rouge_l"] for row in rows),
            "rouge_l_action": mean(row["rouge_l_action"] for row in rows),
            "rouge_l_rationale": mean(row["rouge_l_rationale"] for row in rows),
            "bertscore_f1": None,
            "json_valid_pct": _pct([row["json_valid"] for row in rows]),
            "policy_exact_pct": _pct([row["policy_exact"] for row in rows]),
            "unknown_id_pct": _pct([row["unknown_id"] for row in rows]),
            # none_correct is None on rows that are not "not_covered", so those rows are left out of this %.
            "none_correct_pct": _pct([row["none_correct"] for row in rows if row["none_correct"] is not None]),
            "none_rows": sum(1 for row in rows if row["none_correct"] is not None),
        }
        if all("bertscore_f1" in row for row in rows):
            entry["bertscore_f1"] = mean(row["bertscore_f1"] for row in rows)
        judged = [row for row in judged_by_model.get(key, []) if row.get("status") == "judged"]
        entry["judged"] = len(judged)
        entry["judge_total"] = mean(row["total"] for row in judged) if judged else None
        for criterion in ec.JUDGE_CRITERIA:
            entry["judge_" + criterion] = mean(row["scores"][criterion]["score"] for row in judged) if judged else None
        entry["judge_graders"] = dict(Counter(row["grader"] for row in judged))
        summary[key] = entry
    return summary


# (summary key, display name, format). The table prints these rows in this order.
TABLE_ROWS = (
    ("rouge_l", "ROUGE-L F1, whole answer (canonical JSON)", "{:.3f}"),
    ("rouge_l_action", "ROUGE-L F1, required_action field", "{:.3f}"),
    ("rouge_l_rationale", "ROUGE-L F1, rationale field", "{:.3f}"),
    ("bertscore_f1", "BERTScore F1 (" + ec.BERTSCORE_MODEL + ")", "{:.3f}"),
    ("judge_total", "LLM judge total, mean of %d" % ec.JUDGE_TOTAL_MAX, "{:.2f}"),
    ("judge_policy_correct", "Judge: policy_correct, mean of 2", "{:.2f}"),
    ("judge_action_faithful", "Judge: action_faithful, mean of 2", "{:.2f}"),
    ("judge_rationale_grounded", "Judge: rationale_grounded, mean of 2", "{:.2f}"),
    ("judge_no_invention", "Judge: no_invention, mean of 2", "{:.2f}"),
    ("judged", "Judge: rows graded", "{:d}"),
    ("json_valid_pct", "Valid JSON, strict (%)", "{:.0f}"),
    ("policy_exact_pct", "policy_ids exact match (%)", "{:.0f}"),
    ("unknown_id_pct", "Answers with an id not in the manual (%)", "{:.0f}"),
    ("none_correct_pct", "NONE on not-covered rows (%)", "{:.0f}"),
)


def comparison_rows(summary: Dict[str, Dict], keys: Sequence[str] = None) -> List[Tuple[str, str, str]]:
    """(metric, base, fine-tuned) strings for the notebook table. Absent values print as n/a."""
    wanted = keys or [key for key, _, _ in TABLE_ROWS]
    rows = []
    for key, name, fmt in TABLE_ROWS:
        if key not in wanted:
            continue
        cells = []
        for model in ec.MODEL_KEYS:
            value = summary.get(model, {}).get(key)
            cells.append("n/a" if value is None else fmt.format(value))
        if key == "judged":
            cells = [cell if cell == "n/a" else "%s of %d" % (cell, summary[model]["n"]) for cell, model in zip(cells, ec.MODEL_KEYS)]
        if key == "none_correct_pct":
            name = "%s, n=%d" % (name, summary.get(ec.FINETUNED_KEY, summary.get(ec.BASE_KEY, {})).get("none_rows", 0))
        rows.append((name, cells[0], cells[1]))
    return rows


def write_metrics(summary: Dict[str, Dict], path=None) -> None:
    target = path or ec.EVAL_METRICS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


# (summary key, axis label, divisor) for the chart. Every bar is on a 0-1 scale.
CHART_METRICS = (
    ("rouge_l", "ROUGE-L", 1.0),
    ("bertscore_f1", "BERTScore F1", 1.0),
    ("judge_total", "Judge total / %d" % ec.JUDGE_TOTAL_MAX, float(ec.JUDGE_TOTAL_MAX)),
    ("policy_exact_pct", "Exact policy ids", 100.0),
    ("json_valid_pct", "Valid JSON", 100.0),
)


def save_comparison_chart(summary: Dict[str, Dict], path=None) -> None:
    """Grouped bars, base vs fine-tuned, all metrics scaled to 0-1. Missing metrics are left out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Chart only metrics that both models have, so no bar is missing its partner.
    metrics = [item for item in CHART_METRICS if all(summary.get(m, {}).get(item[0]) is not None for m in ec.MODEL_KEYS)]
    if not metrics:
        return
    # Two categorical slots that pass the colour-blind separation check. Text stays in ink colours.
    colors = {ec.BASE_KEY: "#2a78d6", ec.FINETUNED_KEY: "#eb6834"}
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    width = 0.28
    gap = 0.02
    fig, axis = plt.subplots(figsize=(7.5, 3.8), facecolor=surface)
    axis.set_facecolor(surface)
    positions = range(len(metrics))
    # Grouped bars: base is shifted left of each tick and fine-tuned right of it. Every value is divided
    # by its divisor so percentages and the judge total share the same 0-1 axis.
    for offset, model in ((-(width + gap) / 2, ec.BASE_KEY), ((width + gap) / 2, ec.FINETUNED_KEY)):
        values = [summary[model][key] / divisor for key, _, divisor in metrics]
        bars = axis.bar([p + offset for p in positions], values, width, label=ec.MODEL_LABELS[model], color=colors[model], zorder=2)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, "%.2f" % value, ha="center", va="bottom", fontsize=8, color=ink)
    axis.set_xticks(list(positions))
    axis.set_xticklabels([label for _, label, _ in metrics], fontsize=9, color=ink)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score (0-1)", color=muted)
    axis.tick_params(axis="y", colors=muted)
    axis.yaxis.grid(True, color=grid, linewidth=1, zorder=0)
    axis.set_title("Held-out test set, %d rows" % summary[ec.FINETUNED_KEY]["n"], color=ink, loc="left", pad=26)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(grid)
    # Legend sits in its own row under the title, so it never covers a bar label.
    axis.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=ink, fontsize=9)
    fig.tight_layout()
    target = path or ec.EVAL_CHART_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=120, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------- manual review


def suggest_label(metrics: Dict, judge_row: Optional[dict], pred_ids: List[str], gold_ids: List[str]) -> Tuple[str, str]:
    """A starting label for the human reviewer, with the reason. The reviewer has the final say."""
    # Rules run from most to least severe; the first one that fires decides the label.
    # 1-3: no answer, an invented id, or citing a rule where none applies -> hallucinated.
    # 4: wrong ids -> hallucinated if it added a wrong policy, partial if it only missed one.
    # 5: right ids -> the judge's scores decide between correct and partial.
    if not metrics["parsed"]:
        return ec.LABEL_HALLUCINATED, "no usable JSON answer"
    if metrics["unknown_id"]:
        return ec.LABEL_HALLUCINATED, "names an id that is not in the manual"
    if metrics["none_correct"] is False:
        return ec.LABEL_HALLUCINATED, "cites a policy for a case the manual does not cover"
    if not metrics["policy_exact"]:
        extra = [policy_id for policy_id in pred_ids if policy_id not in gold_ids]
        if pred_ids == [config.NONE_POLICY_ID]:
            return ec.LABEL_HALLUCINATED, "says no policy applies, but %s does" % ", ".join(gold_ids)
        if extra:
            return ec.LABEL_HALLUCINATED, "applies %s, expected %s" % (", ".join(extra), ", ".join(gold_ids))
        return ec.LABEL_PARTIAL, "misses %s" % ", ".join(policy_id for policy_id in gold_ids if policy_id not in pred_ids)
    if judge_row is None or judge_row.get("status") != "judged":
        return ec.LABEL_CORRECT, "ids match; action not judged, check it against the manual"
    scores = judge_row["scores"]
    if scores["no_invention"]["score"] == 0:
        return ec.LABEL_HALLUCINATED, "judge: " + scores["no_invention"]["reason"]
    if scores["action_faithful"]["score"] == ec.JUDGE_SCORE_MAX and scores["no_invention"]["score"] == ec.JUDGE_SCORE_MAX:
        return ec.LABEL_CORRECT, "ids match and the judge found the action faithful"
    return ec.LABEL_PARTIAL, "judge: " + scores["action_faithful"]["reason"]


def build_review_rows(records: List[dict], scores: Dict[str, List[dict]], judge_rows: Optional[List[dict]]) -> List[dict]:
    """Every fine-tuned answer with a suggested label. `label` stays empty for the reviewer."""
    metrics_by_id = {row["id"]: row for row in scores[ec.FINETUNED_KEY]}
    judge_by_id = {row["id"]: row for row in judge_rows or [] if row["model"] == ec.FINETUNED_KEY}
    rows = []
    for record in records:
        answer, _ = parse_answer(record[ec.FINETUNED_KEY])
        gold, _ = parse_answer(record["gold"])
        label, reason = suggest_label(
            metrics_by_id[record["id"]], judge_by_id.get(record["id"]), answer_ids(answer), answer_ids(gold)
        )
        rows.append(
            {
                "id": record["id"],
                "topic": record["topic"],
                "situation": record["situation"],
                "scenario": record["scenario"],
                "expected": record["gold"],
                "output": record[ec.FINETUNED_KEY],
                "suggested_label": label,
                "suggested_reason": reason,
                "label": None,
                "note": "",
            }
        )
    return rows


def read_review(path=None) -> Optional[dict]:
    target = path or ec.MANUAL_REVIEW_PATH
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def write_review(rows: List[dict], path=None) -> bool:
    """Write the review file. Returns False, and writes nothing, if a human label already exists."""
    target = path or ec.MANUAL_REVIEW_PATH
    existing = read_review(target)
    # Guard against a notebook re-run wiping the human's labels: once any label is filled in, never overwrite.
    if existing and any(row.get("label") for row in existing.get("rows", [])):
        return False
    payload = {
        "instructions": "Set label to one of %s for every row, and add a short note. "
        "Judge against the policy manual, not only the expected answer." % ", ".join(ec.LABELS),
        "rubric": {name: text for name, text in ec.LABEL_RUBRIC},
        "rows": rows,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8", newline="\n")
    return True


def hallucination_rate(review_rows: List[dict]) -> Dict:
    """Rate over human-labelled rows only. Pending until at least MIN_REVIEWED rows carry a label."""
    labelled = [row for row in review_rows if row.get("label")]
    bad = [row["id"] for row in labelled if row["label"] not in ec.LABELS]
    if bad:
        raise ValueError("rows %s have a label outside %s" % (bad, ec.LABELS))
    counts = {label: sum(1 for row in labelled if row["label"] == label) for label in ec.LABELS}
    result = {"labelled": len(labelled), "total": len(review_rows), "counts": counts, "rate_pct": None}
    if len(labelled) < ec.MIN_REVIEWED:
        result["status"] = "pending"
        return result
    result["status"] = "ok"
    result["rate_pct"] = 100.0 * counts[ec.LABEL_HALLUCINATED] / len(labelled)
    return result
