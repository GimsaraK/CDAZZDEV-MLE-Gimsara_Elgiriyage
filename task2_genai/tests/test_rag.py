"""Offline checks for the Task 2 RAG fallback layer. No GPU, no network, no model download."""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 2 RAG fallback plan', Date: 2026-10-09 (see CITATIONS.md Entry 20)

import hashlib
import json
import math

import numpy as np
import pytest

pytest.importorskip("chromadb")
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings  # noqa: E402

from task2_genai.src import eval_config as ec  # noqa: E402
from task2_genai.src.eval_metrics import load_test_rows  # noqa: E402
from task2_genai.src.inference import ScoredAnswer, perplexity_from_logprobs  # noqa: E402
from task2_genai.src.manual import load_manual  # noqa: E402
from task2_genai.src.rag import (  # noqa: E402
    augmented_messages,
    build_records,
    build_store,
    calibrate_threshold,
    choose_example,
    detector_report,
    manual_documents,
    pipeline_table,
    policy_context,
    retrieval_recall,
    retrieve_policies,
    score_pipelines,
    select_variant,
)

MANUAL = load_manual()
MANUAL_IDS = {policy["id"] for policy in MANUAL["policies"]}
DIM = 128


class WordHashEmbedding(EmbeddingFunction):
    """Bag of words hashed into a fixed vector: texts sharing words are close. Deterministic across runs."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = np.zeros(DIM, dtype=np.float32)
            for word in text.lower().replace(".", " ").replace(",", " ").split():
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return vectors

    @staticmethod
    def name() -> str:
        return "word_hash_test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "WordHashEmbedding":
        return WordHashEmbedding()


@pytest.fixture(scope="module")
def store():
    return build_store(MANUAL, embedding_function=WordHashEmbedding(), name="rag_test_manual")


def answer(ids, action="Do the thing.", rationale="Because the rule says so."):
    return json.dumps({"policy_ids": ids, "required_action": action, "rationale": rationale})


# ---------------------------------------------------------------- store and retrieval


def test_manual_documents_are_one_rule_and_one_action_per_policy():
    ids, texts, metadatas = manual_documents(MANUAL)
    assert len(ids) == len(set(ids)) == 2 * len(MANUAL["policies"])
    for policy in MANUAL["policies"]:
        parts = {m["part"] for m in metadatas if m["policy_id"] == policy["id"]}
        assert parts == {ec.RAG_PART_RULE, ec.RAG_PART_ACTION}
    # Every chunk names its policy, so retrieved text never loses its source.
    assert all(text.startswith(meta["policy_id"]) for text, meta in zip(texts, metadatas))


def test_store_holds_every_chunk_and_rebuilds_without_duplicates(store):
    assert store.count() == 16
    # Building twice under one name replaces the first collection instead of adding a second copy.
    build_store(MANUAL, embedding_function=WordHashEmbedding(), name="rag_test_rebuild")
    again = build_store(MANUAL, embedding_function=WordHashEmbedding(), name="rag_test_rebuild")
    assert again.count() == 16


def test_retrieval_returns_distinct_policies_best_first(store):
    gift = next(p for p in MANUAL["policies"] if p["id"] == "NW-GIFT")
    ranked = retrieve_policies(store, gift["rule"], k=3)
    assert ranked[0] == "NW-GIFT"
    assert len(ranked) == len(set(ranked)) == 3
    assert retrieve_policies(store, gift["rule"], k=1) == ["NW-GIFT"]


def test_context_and_requery_prompt_keep_the_scenario_and_system_turn():
    row = load_test_rows()[0]
    context = policy_context(["NW-GIFT", "NW-LEAVE"], MANUAL)
    assert context.index("[NW-GIFT]") < context.index("[NW-LEAVE]")
    assert "Required action:" in context
    messages = augmented_messages(row, context)
    assert messages[0] == row["messages"][0]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].endswith(row["messages"][1]["content"])
    assert context in messages[1]["content"]


def test_every_prompt_variant_carries_the_context_and_the_scenario():
    row = load_test_rows()[0]
    context = policy_context(["NW-GIFT"], MANUAL)
    for variant in ec.RAG_VARIANT_ORDER:
        messages = augmented_messages(row, context, variant)
        joined = messages[0]["content"] + messages[1]["content"]
        assert context in joined and row["messages"][1]["content"] in joined
        assert messages[0]["content"].startswith(row["messages"][0]["content"])
    # The system variant leaves the user turn exactly as in training.
    system_variant = augmented_messages(row, context, "system_excerpts")
    assert system_variant[1] == row["messages"][1]
    assert row["messages"][0]["content"] != system_variant[0]["content"]
    # Building a prompt never changes the dataset row itself.
    assert context not in row["messages"][0]["content"]


def test_variant_selection_prefers_exact_ids_and_reports_none_collapse():
    rows = load_test_rows()[:4]
    golds = [row["messages"][-1]["content"] for row in rows]
    none = answer(["NONE"])
    answers = {
        "user_excerpts_none_hint": [none] * 4,
        "user_excerpts_first": golds[:3] + [none],
        "user_scenario_first": golds,
        "system_excerpts": golds,
    }
    best, table = select_variant(rows, answers, MANUAL_IDS)
    by_name = {entry["variant"]: entry for entry in table}
    assert by_name["user_excerpts_none_hint"]["none_answers"] == 4
    # user_scenario_first and system_excerpts tie on every score: the earlier-listed variant wins.
    assert best == "user_scenario_first"


# ---------------------------------------------------------------- confidence


def test_perplexity_is_exp_of_mean_negative_log_prob():
    assert perplexity_from_logprobs([math.log(0.5), math.log(0.5)]) == pytest.approx(2.0)
    assert perplexity_from_logprobs([0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_perplexity_stops_at_the_first_stop_token_and_skips_bad_values():
    logprobs = [math.log(0.5), math.log(0.5), math.log(0.01), math.log(0.01)]
    # Token 99 is the stop token at position 1; positions 2 and 3 are padding and must not count.
    assert perplexity_from_logprobs(logprobs, [7, 99, 0, 0], {99}) == pytest.approx(2.0)
    assert perplexity_from_logprobs([math.log(0.5), float("-inf")]) == pytest.approx(2.0)
    assert perplexity_from_logprobs([]) is None


def test_youden_picks_the_cut_that_separates_wrong_from_right():
    threshold, method, table = calibrate_threshold([1.1, 1.2, 1.3, 1.8, 2.0], [False, False, False, True, True])
    assert method == ec.RAG_METHOD_YOUDEN
    assert threshold == 1.8
    assert max(entry["youden_j"] for entry in table) == pytest.approx(1.0)


def test_youden_ties_go_to_the_higher_threshold():
    # right 1.0 and 1.4, wrong 1.2 and 1.6. At 1.2: TPR 1, FPR 0.5, J 0.5. At 1.6: TPR 0.5, FPR 0, J 0.5.
    # Equal J, so the higher threshold wins: it re-queries fewer rows.
    threshold, method, table = calibrate_threshold([1.0, 1.2, 1.4, 1.6], [False, True, False, True])
    assert method == ec.RAG_METHOD_YOUDEN
    assert {entry["threshold"]: round(entry["youden_j"], 6) for entry in table}[1.2] == 0.5
    assert threshold == 1.6


def test_answers_without_a_perplexity_are_left_out_of_calibration_and_always_fall_back():
    threshold, _, _ = calibrate_threshold([1.0, None, 1.2, 1.4, 1.6], [False, True, True, False, True])
    assert threshold == 1.6
    rows = load_test_rows()[:1]
    records = build_records(rows, [ScoredAnswer("", None, 0)], [ScoredAnswer("{}", 1.0, 2)], [["NW-GIFT"]], 5.0)
    assert records[0]["triggered"]


def test_too_few_wrong_answers_falls_back_to_a_percentile():
    values = [1.0, 1.1, 1.2, 1.3, 1.4]
    threshold, method, table = calibrate_threshold(values, [False, False, False, False, True])
    assert method == ec.RAG_METHOD_PERCENTILE and table == []
    assert threshold == pytest.approx(float(np.percentile(values, ec.RAG_FALLBACK_PERCENTILE)))


# ---------------------------------------------------------------- pipelines and reports


@pytest.fixture(scope="module")
def pipeline():
    rows = load_test_rows()[:4]
    gold_ids = [json.loads(row["messages"][-1]["content"])["policy_ids"] for row in rows]
    # Rows 0 and 1: confident and right. Rows 2 and 3: unsure and wrong first, right after RAG.
    first = [
        ScoredAnswer(rows[0]["messages"][-1]["content"], 1.05, 30),
        ScoredAnswer(rows[1]["messages"][-1]["content"], 1.10, 30),
        ScoredAnswer(answer(["NW-UNKNOWN"]), 1.60, 30),
        ScoredAnswer(answer(["NW-ACCEPT"] if gold_ids[3] != ["NW-ACCEPT"] else ["NW-GIFT"]), 1.70, 30),
    ]
    rag = [ScoredAnswer(row["messages"][-1]["content"], 1.02, 30) for row in rows]
    retrieved = [ids + ["NW-LEAVE"] for ids in gold_ids]
    records = build_records(rows, first, rag, retrieved, threshold=1.5)
    scores, summary = score_pipelines(records, MANUAL_IDS)
    return rows, records, scores, summary


def test_gated_uses_the_rag_answer_only_where_the_fallback_fired(pipeline):
    _, records, _, _ = pipeline
    assert [record["triggered"] for record in records] == [False, False, True, True]
    for record in records:
        expected = record[ec.RAG_ALWAYS] if record["triggered"] else record[ec.RAG_NONE]
        assert record[ec.RAG_GATED] == expected


def test_pipeline_summary_shows_the_fix(pipeline):
    _, _, _, summary = pipeline
    assert summary[ec.RAG_NONE]["policy_exact_pct"] == pytest.approx(50.0)
    assert summary[ec.RAG_GATED]["policy_exact_pct"] == pytest.approx(100.0)
    assert summary[ec.RAG_NONE]["unknown_id_pct"] == pytest.approx(25.0)
    table = pipeline_table(summary, ["policy_exact_pct", "rouge_l"])
    assert [len(row) for row in table] == [4, 4]


def test_detector_report_counts_flagged_wrong_answers(pipeline):
    _, records, _, _ = pipeline
    review = [{"id": 2, "label": ec.LABEL_HALLUCINATED}, {"id": 0, "label": ec.LABEL_HALLUCINATED}]
    report = detector_report(records, review, MANUAL_IDS)
    assert report["flagged"] == 2
    assert report["wrong_policy_ids"] == {"bad": 2, "flagged_bad": 2, "recall_pct": 100.0, "precision_pct": 100.0}
    assert report["hallucinated"]["bad"] == 2 and report["hallucinated"]["flagged_bad"] == 1


def test_retrieval_recall_counts_rows_whose_gold_ids_were_retrieved(pipeline):
    _, records, _, _ = pipeline
    result = retrieval_recall(records)
    assert result["hits"] == result["rows"]


def test_example_rule_prefers_a_fixed_hallucination(pipeline):
    _, records, scores, _ = pipeline
    review = [{"id": 3, "label": ec.LABEL_HALLUCINATED}]
    example, reason = choose_example(records, scores, review)
    assert example["id"] == 3 and reason.startswith("rule 1")
    example, reason = choose_example(records, scores, [])
    assert example["triggered"] and reason.startswith("rule 2")
