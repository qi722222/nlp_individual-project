"""Print all eval losses and highlight the best."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
log = json.loads((REPO_ROOT / "output" / "trainer_log.json").read_text())

evals = [(e.get("step"), e.get("epoch"), e["eval_loss"]) for e in log if "eval_loss" in e]
best = min(evals, key=lambda x: x[2])

print(f"Best: step={best[0]} epoch={best[1]:.2f} loss={best[2]:.4f}")
print("All evals:")
for s, ep, l in evals:
    marker = " <-- best" if (s, ep, l) == best else ""
    print(f"  step={s:4d} epoch={ep:.2f} loss={l:.4f}{marker}")
