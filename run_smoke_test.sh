#!/usr/bin/env bash
# Runs a tiny end-to-end pass of nested_cv.py (12 patients, 1 epoch, reduced
# folds/grid) to catch wiring bugs before committing real GPU time.
# Usage (Git Bash on Windows, from the repo root): bash run_smoke_test.sh
set -e

echo "Copying a 12-patient subset into ./smoke_data ..."
mkdir -p smoke_data
for p in A_001 A_003 A_004 A_005 B_002 B_003 B_006 B_007 C_001 C_004 C_005 C_006; do
    cp -r "data/TrackRad/$p" "smoke_data/$p"
done

export MRI_DATA_DIR="$(pwd)/smoke_data"
export MRI_CHECKPOINT_DIR="$(pwd)/smoke_checkpoints"
export MRI_OUTPUTS_DIR="$(pwd)/smoke_outputs"

echo "Running smoke test..."
uv run python smoke_test.py
