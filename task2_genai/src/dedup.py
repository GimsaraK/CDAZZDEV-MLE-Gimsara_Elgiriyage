"""Exact and near-duplicate checks on the employee scenario.

Exact match is a hash of the normalized text. Near match is character 5-gram
Jaccard. A score at or above the threshold is dropped so the dataset is not
minor rewrites of one story.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

from typing import List, Optional, Set

from . import config
from .textutil import scenario_key


def char_ngrams(text: str, n: int = config.NEAR_DUP_NGRAM) -> Set[str]:
    """Overlapping character n-grams after whitespace is removed and case is folded."""
    # Removing all whitespace makes the comparison about the characters, not the spacing.
    collapsed = "".join((text or "").lower().split())
    # A text shorter than n has no full n-gram: use the whole text as its only "gram".
    if len(collapsed) < n:
        return {collapsed} if collapsed else set()
    # Slide a window of n characters one step at a time: "abcdef" with n=5 -> {"abcde", "bcdef"}.
    # A set (not a list) ignores repeats, which is what Jaccard similarity expects.
    return {collapsed[i : i + n] for i in range(len(collapsed) - n + 1)}


def jaccard(left: Set[str], right: Set[str]) -> float:
    """Intersection over union. Two empty sets count as a match."""
    if not left and not right:
        return 1.0
    # Shared n-grams divided by all distinct n-grams: 1.0 = same text, 0.0 = nothing in common.
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


class DuplicateIndex:
    """Accepted scenarios only. A candidate is compared before it is added."""

    def __init__(self, threshold: float = config.NEAR_DUP_JACCARD, n: int = config.NEAR_DUP_NGRAM):
        self.threshold = threshold
        self.n = n
        self.exact = set()
        self.grams: List[Set[str]] = []
        self.exact_dropped = 0
        self.near_dropped = 0

    def add_existing(self, scenario: str) -> None:
        """Index a row that was already accepted (used when a run resumes)."""
        self.exact.add(scenario_key(scenario))
        self.grams.append(char_ngrams(scenario, self.n))

    def reject_reason(self, scenario: str) -> Optional[str]:
        """'exact', 'near', or None. Does not record the drop; see consider()."""
        # Cheap check first: an identical (normalised) scenario is caught by its hash in O(1).
        if scenario_key(scenario) in self.exact:
            return "exact"
        # Then compare against every accepted scenario. This is O(n) per candidate, which is fine
        # for a few hundred rows; a much larger dataset would need MinHash/LSH instead.
        fresh = char_ngrams(scenario, self.n)
        for previous in self.grams:
            if jaccard(fresh, previous) >= self.threshold:
                return "near"
        return None

    def consider(self, scenario: str) -> Optional[str]:
        """Return a drop reason, or None and add the scenario to the index."""
        reason = self.reject_reason(scenario)
        if reason == "exact":
            self.exact_dropped += 1
            return reason
        if reason == "near":
            self.near_dropped += 1
            return reason
        self.add_existing(scenario)
        return None
