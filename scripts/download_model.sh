#!/usr/bin/env bash
# Download Llama-2-7b from ModelScope to ./model/Llama-2-7b
# Run from the repo root:  bash scripts/download_model.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/model/Llama-2-7b"

mkdir -p "${MODEL_DIR}"

echo "[download_model] target: ${MODEL_DIR}"

# Ensure modelscope is installed
python3 -c "import modelscope" 2>/dev/null || {
  echo "[download_model] installing modelscope ..."
  pip install --quiet modelscope
}

python3 - <<PY
from modelscope import snapshot_download
path = snapshot_download(
    "shakechen/Llama-2-7b",
    cache_dir="${REPO_ROOT}/model/_cache",
    local_dir="${MODEL_DIR}",
)
print(f"[download_model] done -> {path}")
PY

echo "[download_model] files:"
ls -lh "${MODEL_DIR}"
