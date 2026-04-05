#!/usr/bin/env bash
# Convert Meta-format Llama-2-7b weights to HuggingFace format.
# Assumes original weights live at ./model/Llama-2-7b/ with ./model/Llama-2-7b/7B/ subdirectory.
# Output goes to ./model/Llama-2-7b-hf/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="${REPO_ROOT}/model/Llama-2-7b"
OUTPUT_DIR="${REPO_ROOT}/model/Llama-2-7b-hf"

# Try transformers' built-in conversion script first (available in transformers 4.x).
if python -c "import transformers.models.llama.convert_llama_weights_to_hf" 2>/dev/null; then
  echo "[convert] using transformers built-in conversion script"
  python -m transformers.models.llama.convert_llama_weights_to_hf --input_dir "${INPUT_DIR}" --model_size 7B --output_dir "${OUTPUT_DIR}" --llama_version 2
else
  echo "[convert] built-in not available (transformers 5.x?), downloading script"
  SCRIPT_PATH="/tmp/convert_llama.py"
  URL_PRIMARY="https://cdn.jsdelivr.net/gh/huggingface/transformers@v4.44.0/src/transformers/models/llama/convert_llama_weights_to_hf.py"
  URL_FALLBACK="https://raw.githubusercontent.com/huggingface/transformers/v4.44.0/src/transformers/models/llama/convert_llama_weights_to_hf.py"
  if ! wget -q -O "${SCRIPT_PATH}" "${URL_PRIMARY}"; then
    echo "[convert] primary URL failed, trying fallback ..."
    wget -q -O "${SCRIPT_PATH}" "${URL_FALLBACK}"
  fi
  if [ ! -s "${SCRIPT_PATH}" ]; then
    echo "[convert] ERROR: could not download conversion script" >&2
    exit 1
  fi
  python "${SCRIPT_PATH}" --input_dir "${INPUT_DIR}" --model_size 7B --output_dir "${OUTPUT_DIR}" --llama_version 2
fi

echo "[convert] done. Files:"
ls -lh "${OUTPUT_DIR}"
