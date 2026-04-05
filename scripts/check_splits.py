"""Sanity-check train/val splits: count + overlap + sample peek."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

train = json.loads((REPO_ROOT / "data" / "train.json").read_text())
val = json.loads((REPO_ROOT / "data" / "val.json").read_text())

train_qs = {x["question"] for x in train}
val_qs = {x["question"] for x in val}

print(f"train size: {len(train)}")
print(f"val size:   {len(val)}")
print(f"unique questions in train: {len(train_qs)}")
print(f"unique questions in val:   {len(val_qs)}")
print(f"overlap (question text):   {len(train_qs & val_qs)}")

print("\nfirst 3 train samples:")
for x in train[:3]:
    print(f"  Q: {x['question']}")
    print(f"  A: {x['correct_answer']}")

print("\nfirst 3 val samples:")
for x in val[:3]:
    print(f"  Q: {x['question']}")
    print(f"  A: {x['correct_answer']}")
