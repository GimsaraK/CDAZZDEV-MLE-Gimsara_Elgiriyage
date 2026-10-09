"""Prompt-length, topic, and keyword summaries for the diversity rubric.

Plots are written next to a JSON report. The notebook displays the same figures.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from . import config
from .textutil import tokens, word_count


def _user_text(row: dict) -> str:
    """The scenario is the user turn. Length and keywords are computed on that text only."""
    for message in row["messages"]:
        if message["role"] == "user":
            return message["content"]
    return ""


def length_stats(lengths: Sequence[int]) -> Dict[str, float]:
    """Min, median, p90, max, and mean. Median and p90 use a sorted copy."""
    ordered = sorted(lengths)
    if not ordered:
        return {"min": 0, "median": 0, "p90": 0, "max": 0, "mean": 0, "n": 0}

    def _at(fraction: float) -> float:
        # Nearest-rank on the sorted list. Fine for a few hundred rows.
        # fraction * (n - 1) is the position of that percentile; min/max clamp it inside the list.
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return float(ordered[index])

    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": _at(0.5),
        "p90": _at(0.9),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def keyword_counts(texts: Sequence[str], top_n: int = 20) -> List[Dict[str, int]]:
    """Top unigrams after the small stopword list, as {token, count} rows."""
    counts: Counter = Counter()
    for text in texts:
        for token in tokens(text):
            # Skip filler words and 1-2 letter tokens so the chart shows topic words, not "the" or "is".
            if token in config.STOPWORDS or len(token) < 3:
                continue
            counts[token] += 1
    return [{"token": token, "count": count} for token, count in counts.most_common(top_n)]


def diversity_report(rows: Sequence[dict]) -> Dict:
    """Numbers behind the charts: length stats, topic counts, situation counts, keywords."""
    texts = [_user_text(row) for row in rows]
    lengths = [word_count(text) for text in texts]
    topics = Counter(row["topic"] for row in rows)
    situations = Counter(row["situation"] for row in rows)
    return {
        "length_words": length_stats(lengths),
        "topic_counts": dict(sorted(topics.items())),
        "situation_counts": dict(sorted(situations.items())),
        "top_keywords": keyword_counts(texts),
        "lengths": lengths,
    }


def write_diversity_plots(report: Dict, output_dir: Path) -> Dict[str, str]:
    """Save the histogram and the two bar charts. Returns the file names."""
    import matplotlib

    # "Agg" renders to files without needing a display, so this works on servers and in tests.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Length histogram. Bins of 10 words keep a 40-180 range readable.
    fig, axis = plt.subplots(figsize=(7, 3.5))
    # Bin edges 0, 10, 20, ... up past the longest scenario; "+ [10]" keeps max() valid if the list is empty.
    axis.hist(report["lengths"], bins=range(0, max(report["lengths"] + [10]) + 10, 10), color="#2c5f8a")
    axis.set_title("Scenario length (words)")
    axis.set_xlabel("Words in the user turn")
    axis.set_ylabel("Examples")
    fig.tight_layout()
    length_path = output_dir / "prompt_length_hist.png"
    fig.savefig(length_path, dpi=120)
    plt.close(fig)
    paths["length_hist"] = length_path.name

    def _bar(counts: Dict[str, int], title: str, filename: str, xlabel: str) -> None:
        fig, axis = plt.subplots(figsize=(7, 3.5))
        labels = list(counts.keys())
        axis.bar(labels, [counts[label] for label in labels], color="#2c5f8a")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Examples")
        axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths[filename] = path.name

    _bar(report["topic_counts"], "Examples per policy topic", "topic_frequency.png", "Topic")
    _bar(report["situation_counts"], "Examples per situation type", "situation_frequency.png", "Situation")
    return paths


def write_diversity_outputs(rows: Sequence[dict], output_dir: Path = None) -> Dict:
    """Write the JSON report and the plots. The JSON omits the raw length list."""
    output_dir = output_dir or config.OUTPUTS_DIR
    report = diversity_report(rows)
    write_diversity_plots(report, output_dir)
    public = {key: value for key, value in report.items() if key != "lengths"}
    config.DIVERSITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.DIVERSITY_REPORT_PATH.write_text(json.dumps(public, indent=2), encoding="utf-8", newline="\n")
    return report
