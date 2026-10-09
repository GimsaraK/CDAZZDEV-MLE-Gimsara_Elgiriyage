"""80/10/10 split that still puts every topic in train, validation, and test.

With 8 topics and sizes 160/20/20, a plain random cut can leave a topic out of
the small validation set. Each topic therefore gives one row to each split
first. The remaining rows fill the quotas.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2A dataset plan', Date: 2026-10-08 (see CITATIONS.md Entry 8)

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def split_rows(
    rows: Sequence[dict],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Return train, val, test. Raises if a topic has fewer than 3 rows or the sizes do not add up."""
    if train_size + val_size + test_size != len(rows):
        raise ValueError(
            f"split sizes {train_size}+{val_size}+{test_size} do not equal {len(rows)} rows"
        )
    # Bucket rows by topic so each topic can be guaranteed a place in every split.
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["topic"]].append(row)
    # A private Random(seed) makes the split reproducible without touching the global random state.
    rng = random.Random(seed)
    for topic in grouped:
        rng.shuffle(grouped[topic])

    # One row per split per topic, before any extra rows are assigned.
    train: List[dict] = []
    val: List[dict] = []
    test: List[dict] = []
    pool: List[dict] = []
    for topic in sorted(grouped):
        items = grouped[topic]
        if len(items) < 3:
            raise ValueError(f"topic {topic} has {len(items)} rows; need at least 3 so it appears in every split")
        val.append(items[0])
        test.append(items[1])
        train.append(items[2])
        pool.extend(items[3:])
    rng.shuffle(pool)

    # Fill validation, then test, then train. Topic coverage is already guaranteed.
    need_val = val_size - len(val)
    need_test = test_size - len(test)
    if need_val < 0 or need_test < 0:
        raise ValueError("more topics than validation or test slots")
    # The shuffled pool is cut into three consecutive slices: [val | test | everything else -> train].
    val.extend(pool[:need_val])
    test.extend(pool[need_val : need_val + need_test])
    train.extend(pool[need_val + need_test :])
    if len(train) != train_size or len(val) != val_size or len(test) != test_size:
        raise ValueError(
            f"split ended at {len(train)}/{len(val)}/{len(test)}, expected {train_size}/{val_size}/{test_size}"
        )
    return train, val, test
