"""
centralized configuration
"""

import os
from pathlib import Path

# --- Routes ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("MRI_DATA_DIR", PROJECT_ROOT / "data" / "TrackRad"))
CHECKPOINT_DIR = Path(os.environ.get("MRI_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
OUTPUTS_DIR = Path(os.environ.get("MRI_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Preprocessing ---
TARGET_SIZE = (256, 256)
EPSILON_BG = 1.0  # almost cero threshold to separate real background from anatomy

# --- Patients split ---
TEST_SIZE = 0.3        # train vs (val+test)
VAL_TEST_SPLIT = 0.5   # val vs. test within the remaining 30%
RANDOM_STATE = 42

# --- VoxelMorph architecture ---
VXM_INSHAPE = (256, 256)
VXM_INT_STEPS = 7
VXM_INT_DOWNSIZE = 2
VXM_SRC_FEATS = 1
VXM_TRG_FEATS = 1

# --- Training ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
N_EPOCHS = 100
PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
GRAD_CLIP_MAX_NORM = 1.0

# --- Loss ---
LAMBDA_DVF = 1.0
LAMBDA_SMOOTH = 0.1
CHARBONNIER_EPS = 1e-3

# --- Device ---
DEVICE = "cuda"  # it resolves automatially to cpu if theres no GPU (see src/utils.py)
