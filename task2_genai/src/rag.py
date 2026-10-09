"""Task 2 bonus: a retrieval-augmented fallback for low-confidence answers.

The fine-tuned model answers closed-book first. When that answer's perplexity is at or above a threshold
calibrated on the validation rows, the policy clauses most similar to the scenario are retrieved from a
ChromaDB store built from the policy manual, and the model is asked again with those clauses in the user
turn. Everything except the two model calls is here and runs on CPU, so it is tested offline.

Pipeline map (each stage is marked with a "STAGE n" banner below):

  INDEXING (once per run, before any question)
    STAGE 1  Chunking ............ manual_documents()   policy manual -> 16 chunks (rule + required action per policy)
    STAGE 2  Embedding ........... inside build_store() Chroma turns each chunk into a 384-dim vector (all-MiniLM-L6-v2)
    STAGE 3  Storing / indexing .. build_store()        vectors + text + metadata go into an HNSW index (cosine distance)

  ANSWERING (per question)
    STAGE 4  Confidence gate ..... inference.generate_scored() + calibrate_threshold() + build_records()
                                   closed-book answer -> perplexity -> re-query only if perplexity >= threshold
    STAGE 5  Retrieval ........... retrieve_policies()  the scenario is embedded the same way and the nearest
                                   chunks are found; they are reduced to the top 3 distinct policies
    STAGE 6  Augmentation ........ policy_context() + augmented_messages()  retrieved text is put into the prompt
    STAGE 7  Generation (re-query) inference.generate_scored() on the augmented prompt (called from the notebook)

  EVALUATION
    STAGE 8  select_variant() (prompt layout chosen on validation), score_pipelines(), detector_report(),
             retrieval_recall(), choose_example(), and the files and chart.

Retrieval method: DENSE retrieval (semantic vector search), not BM25. BM25 is a sparse, lexical method that
scores documents by shared words (term frequency x inverse document frequency). Here both the chunks and
the question are turned into sentence-embedding vectors by the same model (all-MiniLM-L6-v2, a
sentence-transformers bi-encoder), and relevance is the cosine similarity between them, so a scenario about
"a supplier dinner during a tender" can match the gift clause without sharing its exact words. Chroma finds the
nearest vectors with an HNSW (Hierarchical Navigable Small World) approximate-nearest-neighbour index. There
is no keyword (BM25) component, no hybrid scoring, and no re-ranker; the only post-processing is collapsing
chunks to distinct policies (STAGE 5).
"""
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 2 RAG fallback plan', Date: 2026-10-09 (see CITATIONS.md Entry 20)

import json
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from . import eval_config as ec
from .eval_metrics import (
    TABLE_ROWS,
    answer_ids,
    canonical_answer,
    domain_metrics,
    gold_answer,
    parse_answer,
    rouge_l,
    score_predictions,
    summarize,
)
from .manual import load_manual, policy_by_id
from .prompts import RAG_PROMPT_VARIANTS

# ================================================================ INDEXING: build the vector store
# ---------------------------------------------------------------- STAGE 1 - CHUNKING
# The "documents" are the policy manual the training data was generated from (data/policy_manual.json).
# Chunking strategy: structural, not fixed-size. Each policy has two natural parts, its rule and its required
# action, so each part becomes one chunk (8 policies x 2 = 16 chunks). The chunks are short (one to three
# sentences), so no chunk needs splitting and no overlap window is needed.


def manual_documents(manual: Optional[Dict] = None) -> Tuple[List[str], List[str], List[dict]]:
    """(ids, texts, metadatas): two documents per policy, its rule and its required action.

    Splitting them lets a scenario match on either the situation (rule) or the outcome (action). Each text
    starts with the policy id and title, so a retrieved chunk never loses the policy it belongs to.
    """
    manual = manual if manual is not None else load_manual()
    # Three parallel lists, the shape Chroma's collection.add() expects:
    #   ids       - a unique key per chunk, e.g. "NW-GIFT:rule"
    #   texts     - the chunk text that gets embedded (and returned on retrieval)
    #   metadatas - fields stored next to the vector, used to map a retrieved chunk back to its policy
    ids, texts, metadatas = [], [], []
    for policy in manual["policies"]:
        # Prefixing every chunk with the policy id and title gives the embedding the policy's name as
        # context, and keeps a chunk readable on its own when it is retrieved.
        head = f"{policy['id']} {policy['title']}."
        for part, label, body in (
            (ec.RAG_PART_RULE, "Rule", policy["rule"]),
            (ec.RAG_PART_ACTION, "Required action", policy["required_action"]),
        ):
            ids.append(f"{policy['id']}:{part}")
            texts.append(f"{head} {label}: {body}")
            metadatas.append({"policy_id": policy["id"], "topic": policy["topic"], "part": part})
    return ids, texts, metadatas


# ---------------------------------------------------------------- STAGE 2 - EMBEDDING and STAGE 3 - STORING
# Both happen in build_store(). There is no separate embed() call in this file: the collection is created with
# an embedding function, and Chroma calls it on the chunk texts inside collection.add() (STAGE 2), then writes
# each vector with its text and metadata into the collection's HNSW index (STAGE 3). Queries are embedded by
# the same function inside collection.query() (STAGE 5), so chunks and questions live in the same vector space.


def build_store(manual: Optional[Dict] = None, embedding_function=None, name: str = ec.RAG_COLLECTION):
    """An in-memory ChromaDB collection of the manual's clauses.

    embedding_function=None uses Chroma's default (all-MiniLM-L6-v2, ONNX on CPU); the tests pass a small
    deterministic one so they need no download. In-memory clients share collections inside one process, so
    an existing collection with this name is dropped first: the store is always rebuilt from the manual.
    """
    import chromadb

    # STAGE 3 (storage setup): an in-memory ChromaDB client. Nothing is written to disk; the store is rebuilt
    # from the manual on every run (16 chunks take well under a second).
    client = chromadb.EphemeralClient()
    if name in [collection.name for collection in client.list_collections()]:
        client.delete_collection(name)
    # The index type is HNSW (approximate nearest-neighbour graph) and its distance is cosine:
    # cosine distance = 1 - cosine similarity, so 0 means same direction (same meaning) and 2 means opposite.
    kwargs = {"configuration": {"hnsw": {"space": ec.RAG_DISTANCE}}}
    # STAGE 2 (embedding model): with no embedding_function, Chroma uses its default, all-MiniLM-L6-v2
    # (a sentence-transformers model run through ONNX on CPU, 384-dimensional output, ~80 MB download the
    # first time). The tests pass a tiny deterministic one instead.
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    collection = client.create_collection(name, **kwargs)
    # STAGE 1 output -> STAGE 2 + 3: add() embeds every chunk text and stores vector + text + metadata.
    ids, texts, metadatas = manual_documents(manual)
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return collection


# ================================================================ ANSWERING: retrieve, augment, re-query
# ---------------------------------------------------------------- STAGE 5 - RETRIEVAL (dense vector search)
# (STAGE 4, the confidence gate that decides whether retrieval happens at all, is further down: see
# calibrate_threshold() and build_records(); the perplexity itself comes from inference.generate_scored().)


def retrieve_policies(collection, scenario: str, k: int = ec.RAG_TOP_K_POLICIES) -> List[str]:
    """The k most relevant policy ids for a scenario, best first.

    Every chunk is ranked (the store is small), then distinct policy ids are kept in rank order, so a policy
    whose rule and action both match is counted once and k always means k different policies.
    """
    # query() embeds the scenario with the same model as the chunks (query embedding), then returns the chunks
    # ordered by cosine distance, nearest first. This is the dense-retrieval step: no keyword matching.
    # n_results = all 16 chunks, because the ranking is then reduced to policies below.
    result = collection.query(query_texts=[scenario], n_results=collection.count(), include=["metadatas"])
    # Post-processing (chunk -> document): the two chunks of one policy can both rank high, so walk the ranked
    # chunks and keep each policy id the first time it appears, until k distinct policies are found.
    ranked: List[str] = []
    for metadata in result["metadatas"][0]:
        policy_id = metadata["policy_id"]
        if policy_id not in ranked:
            ranked.append(policy_id)
        if len(ranked) == k:
            break
    return ranked


# ---------------------------------------------------------------- STAGE 6 - AUGMENTATION (prompt construction)
# The retrieved policy ids are turned back into text and placed in the re-query prompt. Note the model gets
# the FULL policy (rule and action) for each retrieved id, not only the chunk that matched, so it never sees
# half a policy.


def policy_context(policy_ids: Sequence[str], manual: Optional[Dict] = None) -> str:
    """The full text of the retrieved policies, in retrieval order, for the re-query prompt."""
    # The text comes from the manual itself, not from the stored chunks, so the excerpts are exact policy wording.
    policies = policy_by_id(manual if manual is not None else load_manual())
    blocks = []
    for policy_id in policy_ids:
        policy = policies[policy_id]
        blocks.append(
            f"[{policy_id}] {policy['title']}\nRule: {policy['rule']}\nRequired action: {policy['required_action']}"
        )
    return "\n\n".join(blocks)


def scenario_of(row: dict) -> str:
    """The user turn of a dataset row: the scenario, exactly as the model saw it in 2C."""
    return row["messages"][1]["content"]


def first_pass_messages(row: dict) -> List[dict]:
    """System + user turns only, the same prompt as section 10 (the gold assistant turn is left out)."""
    return row["messages"][:2]


def augmented_messages(row: dict, context: str, variant: str = ec.RAG_FIRST_VARIANT) -> List[dict]:
    """The re-query prompt for one prompt variant (see prompts.RAG_PROMPT_VARIANTS).

    "user" variants keep the trained system turn and put the excerpts and the scenario in the user turn.
    The "system" variant appends the excerpts to the system turn and keeps the user turn exactly as in
    training (the plain scenario).
    """
    position, template = RAG_PROMPT_VARIANTS[variant]
    # Copies (dict(...)), so building a prompt never changes the dataset row itself.
    system, user = dict(row["messages"][0]), dict(row["messages"][1])
    if position == "system":
        system["content"] = system["content"] + template.format(context=context)
    else:
        user["content"] = template.format(context=context, scenario=scenario_of(row))
    return [system, user]
    # STAGE 7 - GENERATION (the re-query) is not in this file: the notebook passes these messages to
    # inference.generate_scored(), the same greedy decoding as the first answer.


# ---------------------------------------------------------------- STAGE 4 - CONFIDENCE GATE (when to retrieve)
# The first answer's perplexity comes from inference.generate_scored(). calibrate_threshold() sets the cut-off
# on the validation rows, and build_records() applies it: retrieval + re-query only when perplexity >= threshold.


def is_wrong(answer_text: str, row: dict, manual_ids: set) -> bool:
    """A wrong answer for calibration: its policy ids do not exactly match the gold ids."""
    return not domain_metrics(answer_text, gold_answer(row), manual_ids, row["situation"])["policy_exact"]


def calibrate_threshold(perplexities: Sequence[float], wrong: Sequence[bool]) -> Tuple[float, str, List[dict]]:
    """(threshold, method, table). The fallback fires when an answer's perplexity is >= threshold.

    Youden's J: for each observed perplexity t, TPR = share of wrong answers with perplexity >= t (caught)
    and FPR = share of right answers with perplexity >= t (re-queried for nothing); pick the t with the
    largest TPR - FPR. Ties go to the higher t, which re-queries fewer rows for the same benefit. With too
    few wrong or right answers J is not meaningful, so a fixed percentile of the perplexities is used.
    """
    # An answer with no scoreable tokens has no perplexity; it says nothing about where the cut should be.
    pairs = [(float(value), flag) for value, flag in zip(perplexities, wrong) if value is not None]
    values = [value for value, _ in pairs]
    wrong = [flag for _, flag in pairs]
    n_wrong = sum(1 for flag in wrong if flag)
    n_right = len(values) - n_wrong
    if n_wrong < ec.RAG_MIN_CLASS or n_right < ec.RAG_MIN_CLASS:
        threshold = float(np.percentile(values, ec.RAG_FALLBACK_PERCENTILE))
        return threshold, ec.RAG_METHOD_PERCENTILE, []
    table = []
    for candidate in sorted(set(values)):
        caught = sum(1 for value, flag in zip(values, wrong) if flag and value >= candidate)
        false_alarm = sum(1 for value, flag in zip(values, wrong) if not flag and value >= candidate)
        tpr, fpr = caught / n_wrong, false_alarm / n_right
        table.append({"threshold": candidate, "tpr": tpr, "fpr": fpr, "youden_j": tpr - fpr})
    # max over (J, threshold): the highest J, and among equal J values the highest threshold.
    best = max(table, key=lambda entry: (round(entry["youden_j"], 12), entry["threshold"]))
    return best["threshold"], ec.RAG_METHOD_YOUDEN, table


# ================================================================ EVALUATION
# ---------------------------------------------------------------- STAGE 8a - choosing the prompt layout (validation)


def select_variant(rows: List[dict], answers: Dict[str, Sequence[str]], manual_ids: set) -> Tuple[str, List[dict]]:
    """(best variant, table): the re-query prompt chosen on the validation rows, with RAG on every row.

    Ranked by exact policy ids (the task's main correctness check), then whole-answer ROUGE-L, then the
    listed order (simpler first). "NONE answers" shows a collapse like the first run's at a glance.
    """
    golds = [gold_answer(row) for row in rows]
    table = []
    for order, variant in enumerate(ec.RAG_VARIANT_ORDER):
        if variant not in answers:
            continue
        texts = list(answers[variant])
        checks = [domain_metrics(text, gold, manual_ids, row["situation"]) for text, gold, row in zip(texts, golds, rows)]
        rouge = rouge_l([canonical_answer(text) for text in texts], golds)
        table.append({
            "variant": variant,
            "policy_exact_pct": 100.0 * sum(c["policy_exact"] for c in checks) / len(rows),
            "rouge_l": mean(rouge),
            "none_answers": sum(1 for text in texts if answer_ids(parse_answer(text)[0]) == [config.NONE_POLICY_ID]),
            "order": order,
        })
    best = max(table, key=lambda entry: (round(entry["policy_exact_pct"], 9), round(entry["rouge_l"], 9), -entry["order"]))
    return best["variant"], table


# ---------------------------------------------------------------- STAGE 8b - the three pipelines (test rows)


def build_records(
    rows: List[dict],
    first: Sequence,
    rag: Sequence,
    retrieved: Sequence[Sequence[str]],
    threshold: float,
) -> List[dict]:
    """One record per test row: the first answer, the RAG answer, and what each pipeline returns.

    The RAG answer is generated for every row once. The gated pipeline uses it only where the first answer's
    perplexity is >= threshold (the fallback fired); the always-RAG pipeline uses it everywhere.
    """
    records = []
    for index, row in enumerate(rows):
        first_answer, rag_answer = first[index], rag[index]
        # STAGE 4 applied: this line is the confidence gate. At or above the threshold the RAG answer is used.
        # No perplexity means an empty answer: confidence is unknown, so the fallback fires.
        triggered = first_answer.perplexity is None or first_answer.perplexity >= threshold
        records.append(
            {
                "id": index,
                "topic": row["topic"],
                "situation": row["situation"],
                "scenario": scenario_of(row),
                "gold": gold_answer(row),
                "first_perplexity": first_answer.perplexity,
                "rag_perplexity": rag_answer.perplexity,
                "triggered": triggered,
                "retrieved_ids": list(retrieved[index]),
                ec.RAG_NONE: first_answer.text,
                ec.RAG_GATED: rag_answer.text if triggered else first_answer.text,
                ec.RAG_ALWAYS: rag_answer.text,
            }
        )
    return records


def score_pipelines(records: List[dict], manual_ids: set) -> Tuple[Dict[str, List[dict]], Dict[str, Dict]]:
    """Per-row scores and the summary for each pipeline, by the same code that scored 2C."""
    scores = score_predictions(records, manual_ids, keys=ec.RAG_PIPELINES)
    return scores, summarize(scores)


def pipeline_table(summary: Dict[str, Dict], keys: Sequence[str]) -> List[Tuple[str, ...]]:
    """(metric, value per pipeline) rows, with the same metric names and formats as the 2C tables."""
    rows = []
    for key, name, fmt in TABLE_ROWS:
        if key not in keys:
            continue
        cells = []
        for pipeline in ec.RAG_PIPELINES:
            value = summary.get(pipeline, {}).get(key)
            cells.append("n/a" if value is None else fmt.format(value))
        rows.append((name, *cells))
    return rows


def retrieval_recall(records: List[dict]) -> Dict:
    """Share of rows whose gold policy ids were all retrieved. NONE rows have nothing to retrieve."""
    hits, total = 0, 0
    for record in records:
        gold_ids = [pid for pid in answer_ids(parse_answer(record["gold"])[0]) if pid != config.NONE_POLICY_ID]
        if not gold_ids:
            continue
        total += 1
        hits += all(pid in record["retrieved_ids"] for pid in gold_ids)
    return {"rows": total, "hits": hits, "recall_pct": 100.0 * hits / total if total else None}


def detector_report(records: List[dict], review_rows: Optional[List[dict]], manual_ids: set) -> Dict:
    """How well 'perplexity >= threshold' flagged the answers that needed help.

    Two references: the manual-review label 'hallucinated' (section 12) and wrong policy ids. For each,
    recall = flagged bad / all bad, and precision = flagged bad / all flagged.
    """
    labels = {row["id"]: row.get("label") for row in (review_rows or [])}
    flagged = [record for record in records if record["triggered"]]

    def stats(is_bad) -> Dict:
        bad = [record for record in records if is_bad(record)]
        flagged_bad = [record for record in flagged if is_bad(record)]
        return {
            "bad": len(bad),
            "flagged_bad": len(flagged_bad),
            "recall_pct": 100.0 * len(flagged_bad) / len(bad) if bad else None,
            "precision_pct": 100.0 * len(flagged_bad) / len(flagged) if flagged else None,
        }

    report = {"flagged": len(flagged), "rows": len(records)}
    report["wrong_policy_ids"] = stats(
        lambda record: not domain_metrics(record[ec.RAG_NONE], record["gold"], manual_ids, record["situation"])["policy_exact"]
    )
    if labels:
        report["hallucinated"] = stats(lambda record: labels.get(record["id"]) == ec.LABEL_HALLUCINATED)
    return report


def choose_example(records: List[dict], scores: Dict[str, List[dict]], review_rows: Optional[List[dict]]) -> Tuple[Optional[dict], str]:
    """The before-and-after example, by a fixed rule (so it is not picked by hand).

    Rule 1: the first triggered row (lowest id) whose first answer the manual review labelled hallucinated
    and whose RAG answer has the exact gold policy ids. Rule 2, if none: the triggered row with the largest
    whole-answer ROUGE-L gain. None if the fallback never fired.
    """
    labels = {row["id"]: row.get("label") for row in (review_rows or [])}
    exact = {row["id"]: row["policy_exact"] for row in scores[ec.RAG_ALWAYS]}
    triggered = [record for record in records if record["triggered"]]
    for record in triggered:
        if labels.get(record["id"]) == ec.LABEL_HALLUCINATED and exact[record["id"]]:
            return record, "rule 1: first triggered row labelled hallucinated whose RAG answer has the right policy ids"
    if not triggered:
        return None, "the fallback did not fire on any test row"
    before = {row["id"]: row["rouge_l"] for row in scores[ec.RAG_NONE]}
    after = {row["id"]: row["rouge_l"] for row in scores[ec.RAG_ALWAYS]}
    best = max(triggered, key=lambda record: after[record["id"]] - before[record["id"]])
    return best, "rule 2: no triggered hallucinated row was fixed, so the triggered row with the largest ROUGE-L gain"


def per_row_table(records: List[dict], scores: Dict[str, List[dict]], review_rows: Optional[List[dict]]) -> List[dict]:
    """One display row per test row: confidence, the decision, and the policy ids before and after."""
    labels = {row["id"]: row.get("label") for row in (review_rows or [])}
    exact_before = {row["id"]: row["policy_exact"] for row in scores[ec.RAG_NONE]}
    exact_after = {row["id"]: row["policy_exact"] for row in scores[ec.RAG_ALWAYS]}
    table = []
    for record in records:
        table.append(
            {
                "id": record["id"],
                "situation": record["situation"],
                "review label (first answer)": labels.get(record["id"]) or "-",
                "perplexity": round(record["first_perplexity"], 3) if record["first_perplexity"] is not None else None,
                "fallback": "RAG" if record["triggered"] else "-",
                "retrieved": ", ".join(record["retrieved_ids"]),
                "gold ids": ", ".join(answer_ids(parse_answer(record["gold"])[0])),
                "first ids": ", ".join(answer_ids(parse_answer(record[ec.RAG_NONE])[0])) or "(unparsed)",
                "RAG ids": ", ".join(answer_ids(parse_answer(record[ec.RAG_ALWAYS])[0])) or "(unparsed)",
                "ids right before / after": f"{'yes' if exact_before[record['id']] else 'no'} / {'yes' if exact_after[record['id']] else 'no'}",
            }
        )
    return table


# ---------------------------------------------------------------- STAGE 8c - files and chart


def write_results(records: List[dict], path=None) -> None:
    path = path or ec.RAG_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(summary: Dict, path=None) -> None:
    path = path or ec.RAG_SUMMARY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def read_results(path=None) -> Optional[List[dict]]:
    """The saved per-row records, or None before the GPU run has written them."""
    path = path or ec.RAG_RESULTS_PATH
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_summary(path=None) -> Optional[Dict]:
    path = path or ec.RAG_SUMMARY_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_perplexity_chart(records: List[dict], review_rows: Optional[List[dict]], threshold: float, path=None) -> None:
    """Perplexity of each test row's first answer, coloured by its manual-review label, with the threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or ec.RAG_CHART_PATH
    labels = {row["id"]: row.get("label") for row in (review_rows or [])}
    colours = {ec.LABEL_CORRECT: "#2e7d32", ec.LABEL_PARTIAL: "#f9a825", ec.LABEL_HALLUCINATED: "#c62828"}
    figure, axis = plt.subplots(figsize=(10.5, 3.8))
    for label, colour in colours.items():
        points = [(r["id"], r["first_perplexity"]) for r in records if labels.get(r["id"]) == label and r["first_perplexity"] is not None]
        if points:
            axis.scatter([p[0] for p in points], [p[1] for p in points], color=colour, label=label, s=45, zorder=3)
    unlabelled = [(r["id"], r["first_perplexity"]) for r in records if r["id"] not in labels and r["first_perplexity"] is not None]
    if unlabelled:
        axis.scatter([p[0] for p in unlabelled], [p[1] for p in unlabelled], color="#607d8b", label="not reviewed", s=45, zorder=3)
    axis.axhline(threshold, color="#1565c0", linestyle="--", linewidth=1.2, label=f"threshold {threshold:.3f}")
    axis.set_xlabel("Test row id")
    axis.set_ylabel("Answer perplexity")
    axis.set_title("Fine-tuned answer confidence on the test rows (at or above the line: RAG fallback)")
    axis.set_xticks([r["id"] for r in records])
    axis.grid(alpha=0.3)
    # Outside the plot area, so the legend never hides a dot.
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    kept = [value for value in values if value is not None]
    return mean(kept) if kept else None
