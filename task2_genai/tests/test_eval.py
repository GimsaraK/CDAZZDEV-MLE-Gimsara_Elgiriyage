"""Offline checks for the Task 2C evaluation helpers. No GPU, no network, no model download."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2C evaluation plan' (see CITATIONS.md Entry 12)

import json

import pytest

from task2_genai.src import config
from task2_genai.src import eval_config as ec
from task2_genai.src.eval_metrics import (
    build_review_rows,
    canonical_answer,
    comparison_rows,
    domain_metrics,
    hallucination_rate,
    load_test_rows,
    parse_answer,
    read_review,
    score_predictions,
    suggest_label,
    summarize,
    write_review,
)
from task2_genai.src.judge import JudgeVerdict, judge_all, judge_one
from task2_genai.src.manual import load_manual
from task2_genai.src.train_config import training_files

MANUAL_IDS = {policy["id"] for policy in load_manual()["policies"]}
GOLD = json.dumps(
    {"policy_ids": ["NW-GIFT"], "rationale": "Gifts over 50 USD are declined.", "required_action": "Decline the gift."},
    separators=(",", ":"),
    sort_keys=True,
)


def _verdict(score=2):
    return {name: {"score": score, "reason": "ok"} for name in ec.JUDGE_CRITERIA}


def _overlap(cand, ref):
    """Stand-in for ROUGE-L so the tests do not need rouge_score."""
    return 1.0 if cand == ref else 0.0


def _fake_rouge(cands, refs):
    return [_overlap(c, r) for c, r in zip(cands, refs)]


class FakeJudge:
    def __init__(self, provider, replies):
        self.provider = provider
        self.model = provider + "-model"
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _record(row_id=0, situation="clear_breach", base=GOLD, finetuned=GOLD):
    return {
        "id": row_id,
        "topic": "gifts",
        "situation": situation,
        "scenario": "A buyer is offered a watch by a supplier.",
        "gold": GOLD,
        "base": base,
        "finetuned": finetuned,
        "ft_source": "test",
    }


def test_canonical_form_ignores_key_order_and_whitespace():
    reordered = '{ "required_action": "Decline the gift.", "policy_ids": ["NW-GIFT"],\n "rationale": "Gifts over 50 USD are declined." }'
    assert canonical_answer(reordered) == GOLD
    assert canonical_answer("not json at all ") == "not json at all"


def test_fenced_answer_parses_leniently_but_not_strictly():
    answer, strict = parse_answer("```json\n" + GOLD + "\n```")
    assert answer is not None and strict is False
    answer, strict = parse_answer(GOLD)
    assert answer is not None and strict is True
    assert parse_answer("Sure! Here is my answer.") == (None, False)


def test_domain_metrics_unknown_id_none_and_two_policy_sets():
    unknown = json.dumps({"policy_ids": ["NW-MADEUP"], "rationale": "x", "required_action": "y"})
    assert domain_metrics(unknown, GOLD, MANUAL_IDS, "clear_breach")["unknown_id"] is True
    none_gold = json.dumps({"policy_ids": ["NONE"], "rationale": "x", "required_action": "y"})
    cited = json.dumps({"policy_ids": ["NW-GIFT"], "rationale": "x", "required_action": "y"})
    assert domain_metrics(cited, none_gold, MANUAL_IDS, "not_covered")["none_correct"] is False
    assert domain_metrics(none_gold, none_gold, MANUAL_IDS, "not_covered")["none_correct"] is True
    two_gold = json.dumps({"policy_ids": ["NW-GIFT", "NW-EXPENSE"], "rationale": "x", "required_action": "y"})
    swapped = json.dumps({"policy_ids": ["NW-EXPENSE", "NW-GIFT"], "rationale": "x", "required_action": "y"})
    assert domain_metrics(swapped, two_gold, MANUAL_IDS, "two_policy")["policy_exact"] is True
    assert domain_metrics(cited, two_gold, MANUAL_IDS, "two_policy")["policy_exact"] is False


def test_suggest_label_branches():
    base = {"parsed": True, "unknown_id": False, "none_correct": None, "policy_exact": True}
    judged = {"status": "judged", "scores": _verdict(2)}
    assert suggest_label(dict(base, parsed=False), None, [], ["NW-GIFT"])[0] == ec.LABEL_HALLUCINATED
    assert suggest_label(dict(base, unknown_id=True), None, ["X"], ["NW-GIFT"])[0] == ec.LABEL_HALLUCINATED
    assert suggest_label(dict(base, none_correct=False), None, ["NW-GIFT"], ["NONE"])[0] == ec.LABEL_HALLUCINATED
    wrong = dict(base, policy_exact=False)
    assert suggest_label(wrong, None, ["NONE"], ["NW-GIFT"])[0] == ec.LABEL_HALLUCINATED
    assert suggest_label(wrong, None, ["NW-LEAVE"], ["NW-GIFT"])[0] == ec.LABEL_HALLUCINATED
    assert suggest_label(wrong, None, ["NW-GIFT"], ["NW-GIFT", "NW-EXPENSE"])[0] == ec.LABEL_PARTIAL
    assert suggest_label(base, judged, ["NW-GIFT"], ["NW-GIFT"])[0] == ec.LABEL_CORRECT
    weak = {"status": "judged", "scores": dict(_verdict(2), action_faithful={"score": 1, "reason": "incomplete"})}
    assert suggest_label(base, weak, ["NW-GIFT"], ["NW-GIFT"])[0] == ec.LABEL_PARTIAL
    invented = {"status": "judged", "scores": dict(_verdict(2), no_invention={"score": 0, "reason": "made up a limit"})}
    assert suggest_label(base, invented, ["NW-GIFT"], ["NW-GIFT"]) == (ec.LABEL_HALLUCINATED, "judge: made up a limit")


def test_verdict_rejects_out_of_range_score_and_missing_criterion():
    assert JudgeVerdict.model_validate(_verdict(2)).total == ec.JUDGE_TOTAL_MAX
    with pytest.raises(Exception):
        JudgeVerdict.model_validate(_verdict(3))
    partial = _verdict(1)
    partial.pop("no_invention")
    with pytest.raises(Exception):
        JudgeVerdict.model_validate(partial)


def test_judge_falls_back_and_records_the_grader():
    good = json.dumps(_verdict(1))
    gemma = FakeJudge("openrouter", [RuntimeError("openrouter rate limited (HTTP 429)")])
    backup = FakeJudge("backup", [good])
    outcome = judge_one([gemma, backup], "system", "user")
    assert outcome["status"] == "judged"
    assert outcome["grader"] == "backup:backup-model"
    assert outcome["total"] == 4


def test_judge_repairs_once_then_unjudged_when_all_fail():
    gemma = FakeJudge("openrouter", ["not json", json.dumps(_verdict(2))])
    assert judge_one([gemma], "s", "u")["grader"] == "openrouter:openrouter-model"
    assert gemma.calls == 2
    broken = FakeJudge("openrouter", ["{}", "{}"])
    down = FakeJudge("backup", [RuntimeError("outage")])
    outcome = judge_one([broken, down], "s", "u")
    assert outcome["status"] == "unjudged"
    assert "openrouter" in outcome["error"] and "backup" in outcome["error"]


def test_judge_all_reuses_cached_rows(tmp_path):
    path = tmp_path / "judge.jsonl"
    records = [_record(0), _record(1)]
    first = FakeJudge("openrouter", [json.dumps(_verdict(2))] * 4)
    rows = judge_all(records, [first], path=path, sleep=lambda _: None)
    assert [row["status"] for row in rows] == ["judged"] * 4
    second = FakeJudge("openrouter", [])
    again = judge_all(records, [second], path=path, sleep=lambda _: None)
    assert second.calls == 0
    assert again == rows


def test_summary_and_table_report_missing_metrics_as_na():
    records = [_record(0), _record(1, finetuned="no json here")]
    scores = score_predictions(records, MANUAL_IDS, rouge=_fake_rouge)
    summary = summarize(scores)
    assert summary["base"]["policy_exact_pct"] == 100.0
    assert summary["finetuned"]["json_valid_pct"] == 50.0
    assert summary["finetuned"]["bertscore_f1"] is None
    table = dict((name, (base, ft)) for name, base, ft in comparison_rows(summary))
    assert table["BERTScore F1 (roberta-large)"] == ("n/a", "n/a")


def test_hallucination_rate_pending_then_computed():
    rows = [{"id": i, "label": None} for i in range(20)]
    assert hallucination_rate(rows)["status"] == "pending"
    for i, row in enumerate(rows):
        row["label"] = ec.LABEL_HALLUCINATED if i < 3 else ec.LABEL_CORRECT
    result = hallucination_rate(rows)
    assert result["status"] == "ok" and result["rate_pct"] == 15.0
    rows[0]["label"] = "wrong"
    with pytest.raises(ValueError):
        hallucination_rate(rows)


def test_review_file_never_overwrites_human_labels(tmp_path):
    path = tmp_path / "review.json"
    records = [_record(0)]
    scores = score_predictions(records, MANUAL_IDS, rouge=_fake_rouge)
    rows = build_review_rows(records, scores, None)
    assert rows[0]["label"] is None and rows[0]["suggested_label"] == ec.LABEL_CORRECT
    assert write_review(rows, path) is True
    saved = read_review(path)
    saved["rows"][0]["label"] = ec.LABEL_PARTIAL
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert write_review(rows, path) is False
    assert read_review(path)["rows"][0]["label"] == ec.LABEL_PARTIAL


def test_eval_reads_test_split_and_trainer_does_not():
    rows = load_test_rows()
    assert len(rows) == config.TEST_SIZE
    assert config.TEST_PATH.name not in training_files()


def test_rouge_l_is_one_on_identical_text():
    pytest.importorskip("rouge_score")
    from task2_genai.src.eval_metrics import rouge_l

    assert rouge_l([GOLD], [GOLD]) == [pytest.approx(1.0)]
