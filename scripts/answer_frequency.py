"""Check answer repetition in the dataset.

If a small number of unique answers cover most rows, then val->train 'exact'
overlap is structural (pigeonhole), not a property of the split. In that case
the 'warm' accuracy is mostly answer-frequency memorization, not QA skill.
"""
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
raw = json.loads((REPO_ROOT / "data" / "dataset.json").read_text())

# dedup same as prepare_data.py
seen = set()
data = []
for row in raw:
    q = row["question"]
    if q in seen:
        continue
    seen.add(q)
    data.append(row)

answers = [r["correct_answer"].lower().strip() for r in data]
counter = Counter(answers)

n_rows = len(data)
n_unique = len(counter)

print(f"Total rows (deduped by question): {n_rows}")
print(f"Unique answer strings:            {n_unique}")
print(f"Average times each answer used:   {n_rows / n_unique:.2f}")
print()

# How many rows do the top-K answers account for?
print(f"{'top-K':>8} {'rows covered':>14} {'% of total':>12}")
print("-" * 38)
sorted_counts = counter.most_common()
cum = 0
for k in [10, 50, 100, 200, 500, 1000, n_unique]:
    cum = sum(c for _, c in sorted_counts[:k])
    print(f"{k:>8} {cum:>14} {100*cum/n_rows:>11.1f}%")

print()
print("Most common answers (top 15):")
for ans, c in sorted_counts[:15]:
    print(f"  {c:>4}x  {ans!r}")

print()
# Answers that appear only once — these are the REAL cold-start candidates
singletons = [a for a, c in counter.items() if c == 1]
print(f"Singleton answers (appear exactly once): {len(singletons)} "
      f"({100*len(singletons)/n_unique:.1f}% of unique, "
      f"{100*len(singletons)/n_rows:.1f}% of rows)")
print()
print("Implication:")
print("  If ~50% of rows are covered by a few hundred answers, then val->train")
print("  exact overlap is guaranteed by the dataset structure, not the split.")
print("  A 90/10 split will ALWAYS give ~50% exact overlap on this dataset.")
