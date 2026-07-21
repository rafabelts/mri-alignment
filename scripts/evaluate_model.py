"""
Main evaluation script: loads a trained checkpoint, preprocesses the test split,
reconstructs full images from patches, and calculates EPE / % negative jacobian / SSIM / Dice / TRE

Use:
    uv run python scripts/evaluate_model.py --checkpoint best_vxm_model.pt
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

import config
from src.utils import get_device
from src.preprocessing import preprocess_dataset
from src.dataset import split_patients, MRICineDataset, build_lookup
from src.models import build_voxelmorph
from src.evaluate import inference_with_reconstruction, EvaluationMetric


def main(checkpoint_name):
    device = get_device()
    print(f"Using device: {device}")

    # Note: uses the same random_state as train_model.py, so
    # split_patients() reproduces exactly the same test split
    _, _, test_dirs = split_patients(config.DATA_DIR, verbose=False)

    print("Preprocessing test...")
    ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te = preprocess_dataset(config.DATA_DIR, test_dirs)

    test_dataset = MRICineDataset(ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = build_voxelmorph(device)
    checkpoint_path = config.CHECKPOINT_DIR / checkpoint_name
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    test_results = inference_with_reconstruction(model, test_loader, device=device)
    meta_lookup_te = build_lookup(ram_meta_te)

    eval_metrics = EvaluationMetric(test_results, ram_fixed_te, ram_moving_te, meta_lookup_te)
    eval_metrics.evaluate_reconstructed()
    eval_metrics.evaluate_segmentation(ram_meta_te)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_voxelmorph.pt")
    args = parser.parse_args()
    main(args.checkpoint)
