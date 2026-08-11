"""
Self-contained smoke test for the Optuna-based nested_cv.py: copies a small
12-patient subset (4 per cohort), runs a tiny end-to-end pass (reduced
folds/trials/epochs), and removes everything it generated afterward -
whether the run succeeds or fails.

Usage:
    uv run python smoke_test_optuna.py
"""

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SMOKE_DATA = REPO_ROOT / "smoke_data"
SMOKE_CHECKPOINTS = REPO_ROOT / "smoke_checkpoints"
SMOKE_OUTPUTS = REPO_ROOT / "smoke_outputs"

PATIENTS = [
    "A_001", "A_003", "A_004", "A_005",
    "B_002", "B_003", "B_006", "B_007",
    "C_001", "C_004", "C_005", "C_006",
]


def setup_smoke_data():
    SMOKE_DATA.mkdir(exist_ok=True)
    for p in PATIENTS:
        src = REPO_ROOT / "data" / "TrackRad" / p
        dst = SMOKE_DATA / p
        if not dst.exists():
            shutil.copytree(src, dst)


def cleanup():
    for path in (SMOKE_DATA, SMOKE_CHECKPOINTS, SMOKE_OUTPUTS):
        shutil.rmtree(path, ignore_errors=True)
    print("Smoke test artifacts removed.")


if __name__ == "__main__":
    print("Copying a 12-patient subset into ./smoke_data ...")
    setup_smoke_data()

    os.environ["MRI_DATA_DIR"] = str(SMOKE_DATA)
    os.environ["MRI_CHECKPOINT_DIR"] = str(SMOKE_CHECKPOINTS)
    os.environ["MRI_OUTPUTS_DIR"] = str(SMOKE_OUTPUTS)

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import argparse

    import nested_cv as ncv

    ncv.OUTER_K = 2
    ncv.INNER_K = 2
    ncv.TRIAL_BUDGET = 4
    ncv.SEARCH_EPOCH_CAP = 2
    ncv.FINAL_EPOCHS = 1
    ncv.N_SEEDS = 1

    # transmorph, on purpose - also exercises the lambda_kl search dimension,
    # which voxelmorph doesn't have
    args = argparse.Namespace(model="transmorph", plot_only=False, devices=None)

    try:
        ncv.main(args)
        print("\nSMOKE TEST COMPLETE")
    finally:
        cleanup()
