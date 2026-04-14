"""Split val accuracy into 'answer seen in train' vs 'cold-start' subsets.

Reads output/val_generations.json (written by evaluate.py) and data/train.json.
Classifies each val row by whether its gold answer appears in train, then
computes strict accuracy (gold substring in pred) per subset.
"""
import json
from pathlib import Path

YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parents[1]

train = json.loads((REPO_ROOT / "data" / "train.json").read_text())
gens = json.loads((REPO_ROOT / "output" / "val_generations.json").read_text())

# Three different "seen" definitions — from strictest to loosest
train_answer_strings = set(r["correct_answer"].lower().strip() for r in train)
train_answer_words = set()
for r in train:
    for w in r["correct_answer"].lower().strip().split():
        train_answer_words.add(w)


def classify(gold: str):
    """Return one of: 'exact', 'all_words', 'some_words', 'none'."""
    gold = gold.lower().strip()
    if gold in train_answer_strings:
        return "exact"
    words = gold.split()
    if all(w in train_answer_words for w in words):
        return "all_words"
    if any(w in train_answer_words for w in words):
        return "some_words"
    return "none"


buckets = {"exact": [], "all_words": [], "some_words": [], "none": []}
for g in gens:
    gold = g["gold"]
    pred = g["pred"].lower().strip()
    correct = gold.lower().strip() in pred
    bucket = classify(gold)
    buckets[bucket].append((gold, pred, correct))


def stats(name, rows, color):
    n = len(rows)
    c = sum(1 for _, _, ok in rows if ok)
    acc = 100 * c / n if n else 0.0
    print(f"{color}{name:<40} n={n:>4}  correct={c:>4}  acc={acc:>6.2f}%{RESET}")
    return acc, c, n


print(f"{BOLD}{CYAN}Val accuracy by answer novelty (stricter -> looser){RESET}")
print(f"{CYAN}{'-' * 72}{RESET}")
stats("1. exact: whole answer string seen", buckets["exact"], GREEN)
stats("2. all-words seen (order may differ)", buckets["all_words"], CYAN)
stats("3. some-words seen (partial overlap)", buckets["some_words"], YELLOW)
stats("4. none: no word of gold in any train ans", buckets["none"], RED)
print()
total = sum(buckets.values(), [])
stats("overall", total, BOLD)

# Also compute two reductions:
#   warm1 = exact only          vs cold1 = everything else
#   warm2 = exact + all_words   vs cold2 = some_words + none
print()
print(f"{BOLD}Collapsed views:{RESET}")
warm1 = buckets["exact"]
cold1 = buckets["all_words"] + buckets["some_words"] + buckets["none"]
stats("  warm1 = exact match", warm1, GREEN)
stats("  cold1 = anything else", cold1, YELLOW)
if warm1 and cold1:
    a1 = 100 * sum(1 for _, _, ok in warm1 if ok) / len(warm1)
    a2 = 100 * sum(1 for _, _, ok in cold1 if ok) / len(cold1)
    print(f"  gap: {a1 - a2:+.1f}%")

print()
warm2 = buckets["exact"] + buckets["all_words"]
cold2 = buckets["some_words"] + buckets["none"]
stats("  warm2 = all words seen (any order)", warm2, GREEN)
stats("  cold2 = some/no words seen", cold2, YELLOW)
if warm2 and cold2:
    a1 = 100 * sum(1 for _, _, ok in warm2 if ok) / len(warm2)
    a2 = 100 * sum(1 for _, _, ok in cold2 if ok) / len(cold2)
    print(f"  gap: {a1 - a2:+.1f}%")

# Show a few examples from each bucket for eyeballing
print()
print(f"{BOLD}Sample misses by bucket:{RESET}")
for bname, color in [("exact", GREEN), ("all_words", CYAN), ("some_words", YELLOW), ("none", RED)]:
    misses = [r for r in buckets[bname] if not r[2]]
    if not misses:
        continue
    print(f"\n{color}[{bname}] {len(misses)} misses, sample:{RESET}")
    for gold, pred, _ in misses[:5]:
        print(f"  gold={gold!r:40s}  pred={pred!r}")
