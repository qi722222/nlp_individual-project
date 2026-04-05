"""Plot train/val loss curves from trainer_log.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "lora_config.yaml").read_text())
    output_dir = REPO_ROOT / cfg["output_dir"].lstrip("./")
    log_path = output_dir / "trainer_log.json"
    log = json.loads(log_path.read_text())

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    for entry in log:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="train loss", alpha=0.8)
    plt.plot(eval_steps, eval_losses, label="val loss", marker="o")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("LoRA fine-tuning loss curves (Llama-2-7b)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "loss_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved -> {out_path}")


if __name__ == "__main__":
    main()
