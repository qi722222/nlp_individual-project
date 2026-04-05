"""Split dataset.json into train.json and val.json with a fixed seed."""
import json
import random
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "lora_config.yaml").read_text())

    data_path = REPO_ROOT / cfg["data_path"].lstrip("./")
    train_path = REPO_ROOT / cfg["train_path"].lstrip("./")
    val_path = REPO_ROOT / cfg["val_path"].lstrip("./")
    train_ratio = cfg["train_ratio"]
    seed = cfg["seed"]

    raw = json.loads(data_path.read_text())
    print(f"loaded {len(raw)} samples from {data_path}")

    # Deduplicate by question text, keeping first occurrence.
    # The original dataset has 17 duplicated question texts (34 rows):
    #   - 11 are exact duplicates (same question + same answer)
    #   - 6 have the same question but slightly different answer strings
    #     (e.g. 'cecum' vs 'the cecum', 'atoms' vs 'atom', 'haploid' vs 'haploid life cycle')
    # We keep only the first occurrence to guarantee train/val share no question text.
    seen = set()
    data = []
    dropped = 0
    for row in raw:
        q = row["question"]
        if q in seen:
            dropped += 1
            continue
        seen.add(q)
        data.append(row)
    print(f"deduplicated: dropped {dropped} rows, kept {len(data)} unique questions")

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n_train = int(len(data) * train_ratio)
    train = [data[i] for i in indices[:n_train]]
    val = [data[i] for i in indices[n_train:]]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps(train, ensure_ascii=False, indent=2))
    val_path.write_text(json.dumps(val, ensure_ascii=False, indent=2))

    print(f"train: {len(train)} -> {train_path}")
    print(f"val:   {len(val)} -> {val_path}")


if __name__ == "__main__":
    main()
