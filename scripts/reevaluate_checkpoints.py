"""
Reevaluate already-trained checkpoints with updated metrics (e.g. after
adding a new metric to EvaluationMetric).

Usage:
    uv run python scripts/reevaluate_checkpoints.py --model voxelmorph --seeds 2697 3078 5111 6369 8506
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.utils import get_device
from src.preprocessing import preprocess_dataset
from src.dataset import split_patients, MRICineDataset, build_lookup
from src.models import build_model 
from src.evaluate import inference_with_reconstruction, EvaluationMetric

def main(model_name, seeds):
    device = get_device()
    print(f"Usando dispositivo: {device}")

    # mismo split de siempre -> mismo test set que ya se uso para entrenar/evaluar
    _, _, test_dirs = split_patients(config.DATA_DIR, verbose=False)

    print("Preprocesando test set...")
    ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te = preprocess_dataset(config.DATA_DIR, test_dirs)

    test_dataset = MRICineDataset(ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    meta_lookup_te = build_lookup(ram_meta_te)

    per_run = {"epe": [], "jacobian": [], "ssim": [], "dice": [], "tre": [], "hausdorff": []}

    for seed in seeds:
        checkpoint_path = config.CHECKPOINT_DIR / f"best_{model_name}_seed{seed}.pt"
        print(f"\n{'=' * 60}\n Seed {seed} ({checkpoint_path.name})\n{'=' * 60}")

        model = build_model(model_name, device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        test_results = inference_with_reconstruction(model, test_loader, device=device)
        eval_metrics = EvaluationMetric(test_results, ram_fixed_te, ram_moving_te, meta_lookup_te)
        epe_list, jac_list, ssim_list = eval_metrics.evaluate_reconstructed()
        dice_list, tre_list, hd_list = eval_metrics.evaluate_segmentation(ram_meta_te)

        per_run["epe"].append(np.mean(epe_list))
        per_run["jacobian"].append(np.mean(jac_list))
        per_run["ssim"].append(np.mean(ssim_list))
        per_run["dice"].append(np.nanmean(dice_list))
        per_run["tre"].append(np.nanmean(tre_list))
        per_run["hausdorff"].append(np.nanmean(hd_list))

    print(f"\n{'=' * 60}\n Summary for {model_name} between {len(seeds)} runs (re-evaluated)\n{'=' * 60}")
    for metric_name, values in per_run.items():
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f} (values: {[round(v, 4) for v in values]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["voxelmorph", "transmorph"], default="voxelmorph")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                         help="Exact seeds used in training, ej: --seeds 2697 3078 5111 6369 8506")
    args = parser.parse_args()
    main(args.model, args.seeds)
