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

train_answers = set(r["correct_answer"].lower().strip() for r in train)

seen, unseen = [], []
for g in gens:
    gold = g["gold"].lower().strip()
    pred = g["pred"].lower().strip()
    correct = gold in pred
    (seen if gold in train_answers else unseen).append((gold, pred, correct))

def stats(name, rows, color):
    n = len(rows)
    c = sum(1 for _, _, ok in rows if ok)
    acc = 100 * c / n if n else 0.0
    print(f"{color}{name:<30} n={n:>4}  correct={c:>4}  acc={acc:>6.2f}%{RESET}")
    return acc, c, n

print(f"{BOLD}{CYAN}Accuracy split by answer novelty{RESET}")
print(f"{CYAN}{'-' * 60}{RESET}")
seen_acc, seen_c, seen_n = stats("seen-in-train (warm)", seen, GREEN)
unseen_acc, unseen_c, unseen_n = stats("not-in-train (cold)", unseen, YELLOW)
total_acc, total_c, total_n = stats("overall", seen + unseen, BOLD)

print()
if seen_n and unseen_n:
    gap = seen_acc - unseen_acc
    gap_color = RED if gap > 20 else YELLOW if gap > 10 else GREEN
    print(f"{BOLD}Warm vs cold gap:{RESET} {gap_color}{gap:+.1f}%{RESET}")
    if gap > 20:
        print(f"{RED}  Large gap -> cold-start samples are the ceiling. Train augmentation would help more than hyperparam tuning.{RESET}")
    elif gap > 10:
        print(f"{YELLOW}  Moderate gap -> hyperparam tuning still has some headroom.{RESET}")
    else:
        print(f"{GREEN}  Small gap -> both subsets fail similarly; issue is not answer novelty.{RESET}")

# Show a few cold-start misses so we can eyeball if Llama-2 "should" know them
print()
print(f"{BOLD}Cold-start misses (first 10):{RESET}")
for gold, pred, ok in [r for r in unseen if not r[2]][:10]:
    print(f"  gold={gold!r:40s}  pred={pred!r}")
