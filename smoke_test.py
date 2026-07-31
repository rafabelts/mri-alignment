"""
Runs nested_cv.py end to end with every design constant shrunk down, against
a tiny patient subset - a fast correctness check before committing real GPU
time to the full run. Not part of the pipeline itself; delete after use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import argparse
import nested_cv as ncv

ncv.OUTER_K = 2
ncv.INNER_K = 2
ncv.LAMBDA_SMOOTH_GRID = [0.1]
ncv.LEARNING_RATE_GRID = [1e-4]
ncv.SEARCH_EPOCH_CAP = 1
ncv.FINAL_EPOCHS = 1
ncv.N_SEEDS = 1

args = argparse.Namespace(model="voxelmorph", plot_only=False)
ncv.main(args)
print("\nSMOKE TEST COMPLETE")
