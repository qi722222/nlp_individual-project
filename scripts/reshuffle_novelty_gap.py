"""Sanity-check the warm/cold novelty gap across multiple random splits.

For each seed, re-split dataset.json into 90/10, compute the fraction of val
answers whose words all appear in train answers ('warm' by word-level defn),
and show how many val answers are fully novel. If the warm/cold ratio is
stable across seeds, our current split is representative. If seed 42 looks
unusually skewed, that's evidence the split is unlucky.

This does NOT touch data/train.json or data/val.json — pure analysis.
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
raw = json.loads((REPO_ROOT / "data" / "dataset.json").read_text())

# Same dedup as prepare_data.py
seen = set()
data = []
for row in raw:
    q = row["question"]
    if q in seen:
        continue
    seen.add(q)
    data.append(row)

print(f"{len(data)} unique rows after dedup\n")

header = f"{'seed':>6} {'exact%':>8} {'all_w%':>8} {'some_w%':>9} {'none%':>8}"
print(header)
print("-" * len(header))

for seed in [42, 1, 7, 13, 99, 2026, 31415]:
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    n_train = int(len(data) * 0.9)
    train = [data[i] for i in idx[:n_train]]
    val = [data[i] for i in idx[n_train:]]

    train_ans = set(r["correct_answer"].lower().strip() for r in train)
    train_words = set()
    for r in train:
        for w in r["correct_answer"].lower().strip().split():
            train_words.add(w)

    counts = {"exact": 0, "all_words": 0, "some_words": 0, "none": 0}
    for r in val:
        gold = r["correct_answer"].lower().strip()
        if gold in train_ans:
            counts["exact"] += 1
            continue
        words = gold.split()
        if all(w in train_words for w in words):
            counts["all_words"] += 1
        elif any(w in train_words for w in words):
            counts["some_words"] += 1
        else:
            counts["none"] += 1

    n = len(val)
    print(
        f"{seed:>6} "
        f"{100*counts['exact']/n:>7.1f}% "
        f"{100*counts['all_words']/n:>7.1f}% "
        f"{100*counts['some_words']/n:>8.1f}% "
        f"{100*counts['none']/n:>7.1f}%"
    )

print()
print("Interpretation:")
print("  If seed 42 looks similar to the others -> split is representative.")
print("  If seed 42 has notably lower exact% or higher none% -> unlucky split.")
