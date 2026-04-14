"""Small hyperparameter grid search.

For each config combination in CONFIGS:
  1. Overwrite configs/lora_config.yaml with that combo's values
  2. Run train.py (writes adapter to output/adapter)
  3. Run the instructor-style test on val set
  4. Record accuracy
  5. Copy adapter to output/sweep/<combo_name>/
  6. After all runs, print a sorted table + save results.csv

Run from repo root:  python scripts/sweep.py [--small]
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "configs" / "lora_config.yaml"
ADAPTER_PATH = REPO_ROOT / "output" / "adapter"
SWEEP_DIR = REPO_ROOT / "output" / "sweep"
RESULTS_CSV = REPO_ROOT / "output" / "sweep_results.csv"
VAL_PATH = REPO_ROOT / "data" / "val.json"


# Full 18-combo grid
FULL_GRID = [
    {"lora_r": r, "lora_alpha": r * 2, "lora_dropout": dp, "learning_rate": lr}
    for r in [8, 16, 24]
    for dp in [0.1, 0.2]
    for lr in [5e-5, 1e-4, 2e-4]
]

# Smaller 8-combo grid — faster, covers the most promising neighborhood
SMALL_GRID = [
    {"lora_r": r, "lora_alpha": r * 2, "lora_dropout": dp, "learning_rate": lr}
    for r in [16, 24]
    for dp in [0.1, 0.2]
    for lr in [1e-4, 2e-4]
]


def combo_name(combo):
    return f"r{combo['lora_r']}_dp{combo['lora_dropout']}_lr{combo['learning_rate']:.0e}"


def write_config(base_cfg, combo):
    cfg = dict(base_cfg)
    cfg.update(combo)
    CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_train():
    print("[sweep] running train.py ...")
    subprocess.run([sys.executable, "scripts/train.py"], cwd=REPO_ROOT, check=True)


def run_teacher_test(out_csv):
    print("[sweep] running instructor test on val ...")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_teacher_test.py",
            "--test_data", str(VAL_PATH),
            "--out", str(out_csv),
        ],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    # Parse "Accuracy: XX.XX% (XXX/XXX)" from stdout
    print(result.stdout[-500:])
    for line in result.stdout.splitlines()[::-1]:
        if line.startswith("Accuracy:"):
            pct = float(line.split(":")[1].strip().split("%")[0])
            return pct
    raise RuntimeError("Could not parse accuracy from run_teacher_test output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Use 8-combo grid instead of 18")
    args = parser.parse_args()

    grid = SMALL_GRID if args.small else FULL_GRID
    print(f"[sweep] {len(grid)} combos to run")
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Snapshot base config (we'll mutate + restore)
    base_cfg = yaml.safe_load(CFG_PATH.read_text())
    original_cfg_text = CFG_PATH.read_text()

    results = []
    try:
        for i, combo in enumerate(grid, 1):
            name = combo_name(combo)
            print(f"\n{'='*70}\n[sweep] ({i}/{len(grid)}) {name}  config={combo}\n{'='*70}")

            write_config(base_cfg, combo)

            t0 = time.time()
            try:
                run_train()
                out_csv = REPO_ROOT / "output" / f"sweep_{name}.csv"
                acc = run_teacher_test(out_csv)
                elapsed = time.time() - t0

                # Save adapter copy
                dst = SWEEP_DIR / name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(ADAPTER_PATH, dst)

                results.append({"name": name, **combo, "val_accuracy": acc, "elapsed_sec": elapsed})
                print(f"[sweep] {name} => {acc:.2f}%  ({elapsed/60:.1f} min)")
            except Exception as e:
                print(f"[sweep] {name} FAILED: {e}")
                results.append({"name": name, **combo, "val_accuracy": None, "elapsed_sec": None})

            # Write results after every run so partial progress is saved
            with open(RESULTS_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "lora_r", "lora_alpha", "lora_dropout", "learning_rate", "val_accuracy", "elapsed_sec"])
                w.writeheader()
                for r in results:
                    w.writerow(r)
    finally:
        # Restore original config
        CFG_PATH.write_text(original_cfg_text)
        print("\n[sweep] restored original config")

    # Print sorted table
    print("\n" + "="*70)
    print("SWEEP RESULTS (sorted by val accuracy)")
    print("="*70)
    ok = [r for r in results if r["val_accuracy"] is not None]
    ok.sort(key=lambda r: r["val_accuracy"], reverse=True)
    print(f"{'name':<28} {'val_acc':>10} {'time':>8}")
    for r in ok:
        print(f"{r['name']:<28} {r['val_accuracy']:>9.2f}% {r['elapsed_sec']/60:>7.1f}m")
    failed = [r for r in results if r["val_accuracy"] is None]
    if failed:
        print(f"\n{len(failed)} failed: {[r['name'] for r in failed]}")
    print(f"\nResults also saved to {RESULTS_CSV}")
    print(f"Best adapters at {SWEEP_DIR}/<name>/")


if __name__ == "__main__":
    main()
