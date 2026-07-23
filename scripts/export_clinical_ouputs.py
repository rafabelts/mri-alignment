"""
Generate clinically usable ouputs (fixed, warped image, and predicted DVF)
for N pairs of test set, and reports the computational cost of the best-trained checkpoint.

Usage:
    uv run python scripts/export_clinical_outputs.py --model voxelmorph --n-samples 5
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import config
from src.utils import get_device
from src.preprocessing import preprocess_dataset
from src.dataset import split_patients
from src.models import build_voxelmorph, build_transmorph, load_weights_any_shape
from src.train import benchmark_model
from src.io_utils import run_inference_and_export


def build_model(model_name, device, inshape):
    if model_name == "voxelmorph":
        return build_voxelmorph(device, inshape=inshape)
    elif model_name == "transmorph":
        return build_transmorph(device, inshape=inshape)
    raise ValueError(f"Unknown model: {model_name}")


def main(model_name, n_samples, checkpoint_name):
    device = get_device()
    print(f"Using device : {device}")

    _, _, test_dirs = split_patients(config.DATA_DIR, verbose=False)

    print("Preprocessing test set...")
    ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te = preprocess_dataset(config.DATA_DIR, test_dirs)

    checkpoint_name = checkpoint_name or f"best_{model_name}_seed8506.pt"
    checkpoint_path = config.CHECKPOINT_DIR / checkpoint_name
    print(f"Checkpoint: {checkpoint_path}")

    export_dir = config.OUTPUTS_DIR / "clinical_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    n_samples = min(n_samples, len(ram_fixed_te))
    print(f"\nGenerating clinical exports for {n_samples} pairs of test set...")

    model_cache = {}

    for i in range(n_samples):
        img_fixed_np = ram_fixed_te[i]
        img_moving_np = ram_moving_te[i]
        meta = ram_meta_te[i]
        h, w = img_fixed_np.shape

        if (h, w) not in model_cache:
            model_i = build_model(model_name, device, inshape=(h, w))
            model_i = load_weights_any_shape(model_i, checkpoint_path, device)
            model_cache[(h, w)] = model_i
        model_i = model_cache[(h, w)]

        prefix = f"{meta['seq_id']}_{meta['frame_idx']}"

        fixed_clinical, warped_clinical, pred_dvf_np = run_inference_and_export(
            model_i, img_fixed_np, img_moving_np,
            meta["norm_stats"], meta["spatial_meta"],
            output_dir=export_dir, prefix=prefix,
            device=device,
        )

        print(f"  [{i + 1}/{n_samples}] {prefix} ({h}x{w}) -> "
              f"fixed_{prefix}.mha, warped_{prefix}.mha, dvf_{prefix}.mha")

    print(f"\Saved exports in: {export_dir}")

    print("\n--- Computational cost (256x256, batch_size=1, matching real-time frame-by-frame use) ---")
    model_std = build_model(model_name, device, inshape=config.VXM_INSHAPE)
    model_std = load_weights_any_shape(model_std, checkpoint_path, device)

    sample_fixed = torch.zeros(1, 1, *config.VXM_INSHAPE).to(device).float()
    sample_moving = torch.zeros(1, 1, *config.VXM_INSHAPE).to(device).float()
    benchmark_model(model_std, sample_fixed, sample_moving, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["voxelmorph", "transmorph"], default="voxelmorph")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Checkpoint name; default: best_<model>.pt")
    args = parser.parse_args()
    main(args.model, args.n_samples, args.checkpoint)
