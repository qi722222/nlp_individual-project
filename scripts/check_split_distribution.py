"""Compare answer-length distribution between train and val splits.
If val has significantly more multi-word answers than train, our split may
be unlucky and val is structurally harder.
"""
import json
from collections import Counter
from pathlib import Path

# ANSI colors (readable on both light and dark terminals)
YELLOW = "\033[33m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parents[1]

train = json.loads((REPO_ROOT / "data" / "train.json").read_text())
val = json.loads((REPO_ROOT / "data" / "val.json").read_text())

header = f"{'split':<6} {'n':>5} {'mean_words':>12} {'1-word':>10} {'2-word':>10} {'3+word':>10} {'multi %':>10}"
print(f"{BOLD}{CYAN}{header}{RESET}")
print(f"{CYAN}{'-' * 70}{RESET}")

rows = []
for name, d in [("train", train), ("val", val)]:
    lens = [len(r["correct_answer"].split()) for r in d]
    c = Counter(lens)
    mean = sum(lens) / len(lens)
    one = c.get(1, 0)
    two = c.get(2, 0)
    three_plus = sum(v for k, v in c.items() if k >= 3)
    multi = two + three_plus
    multi_pct = 100 * multi / len(d)
    rows.append((name, len(d), mean, one, two, three_plus, multi_pct))
    color = YELLOW if name == "train" else GREEN
    print(
        f"{color}{name:<6} {len(d):>5} {mean:>12.2f} {one:>10} {two:>10} {three_plus:>10} {multi_pct:>9.1f}%{RESET}"
    )

# Highlight the gap
train_multi = rows[0][6]
val_multi = rows[1][6]
gap = val_multi - train_multi
gap_color = RED if abs(gap) > 3 else GREEN
print()
print(f"{BOLD}Multi-word % gap (val - train): {gap_color}{gap:+.1f}%{RESET}")
if gap > 3:
    print(f"{RED}  ⚠ val has notably more multi-word answers -> structurally harder split{RESET}")
elif gap < -3:
    print(f"{YELLOW}  val has fewer multi-word answers -> should be easier, not the split issue{RESET}")
else:
    print(f"{GREEN}  distributions match -> split is fair{RESET}")

# Also: how many unique answers in val also appear in train?
train_ans = set(r["correct_answer"].lower().strip() for r in train)
val_overlap = sum(1 for r in val if r["correct_answer"].lower().strip() in train_ans)
print()
print(
    f"{BOLD}Val answers also seen in train:{RESET} {CYAN}{val_overlap}/{len(val)} "
    f"({100*val_overlap/len(val):.1f}%){RESET}"
)
print(
    f"{BOLD}Val answers NOT in train (cold-start):{RESET} {YELLOW}{len(val) - val_overlap} "
    f"({100*(len(val)-val_overlap)/len(val):.1f}%){RESET}"
)
