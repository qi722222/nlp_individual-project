#!/usr/bin/env bash
# Backup current output/ to output_round{N}/ before starting a new training round.
# Usage: bash scripts/backup_round.sh 1

set -euo pipefail

ROUND="${1:?Usage: bash scripts/backup_round.sh <round_number>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/output"
DST="${REPO_ROOT}/output_round${ROUND}"

if [ ! -d "${SRC}" ]; then
  echo "No output/ directory to back up."
  exit 0
fi

if [ -d "${DST}" ]; then
  echo "${DST} already exists, skipping."
  exit 0
fi

mv "${SRC}" "${DST}"
echo "Backed up output/ -> output_round${ROUND}/"
