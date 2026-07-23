"""
Usage:
    uv run python scripts/infer_new_images.py --fixed path/to/fixed.mha --moving path/to/moving.mha --model voxelmorph
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.utils import get_device
from src.io_utils import infer_on_external_images


def main(fixed_path, moving_path, model_name, checkpoint_name, prefix):
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_name = checkpoint_name or f"best_{model_name}.pt"
    checkpoint_path = config.CHECKPOINT_DIR / checkpoint_name
    print(f"Checkpoint: {checkpoint_path}")

    export_dir = config.OUTPUTS_DIR / "external_inference"
    export_dir.mkdir(parents=True, exist_ok=True)

    fixed_clinical, warped_clinical, pred_dvf_np = infer_on_external_images(
        model_name, checkpoint_path, fixed_path, moving_path,
        export_dir, prefix, device=device,
    )

    print(f"Ready -> {export_dir}/fixed_{prefix}.mha, warped_{prefix}.mha, dvf_{prefix}.mha")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=str, required=True)
    parser.add_argument("--moving", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["voxelmorph", "transmorph"], default="voxelmorph")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="external")
    args = parser.parse_args()
    main(args.fixed, args.moving, args.model, args.checkpoint, args.prefix)
