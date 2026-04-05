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

    data = json.loads(data_path.read_text())
    print(f"loaded {len(data)} samples from {data_path}")

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
