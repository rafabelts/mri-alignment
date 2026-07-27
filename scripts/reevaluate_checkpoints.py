"""
Reevaluate already-trained checkpoints with updated metrics (e.g. after
adding a new metric to EvaluationMetric).

Usage:
    uv run python scripts/reevaluate_checkpoints.py --model voxelmorph --seeds 8506 6369 5111 2697 3078
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from src.utils import get_device
from src.preprocessing import preprocess_dataset
from src.dataset import split_patients, MRICineDataset, build_lookup
from src.models import build_model 
from src.evaluate import inference_with_reconstruction, EvaluationMetric
from src.train import benchmark_model

def main(model_name, seeds):
    device = get_device()
    print(f"Using: {device}")
    
    # --- output folders ---
    run_dir = config.OUTPUTS_DIR / f"reevaluation_{model_name}"
    boxplots_dir = run_dir / "boxplots"
    per_case_dir = run_dir / "per_case"
    boxplots_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_dirs = split_patients(config.DATA_DIR, verbose=False)

    print("Preprocessing dataset...")
    ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te = preprocess_dataset(config.DATA_DIR, test_dirs)

    test_dataset = MRICineDataset(ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    meta_lookup_te = build_lookup(ram_meta_te)

    per_run = {"epe": [], "jacobian": [], "ssim": [], "dice": [], "tre": [], "hausdorff": []}
    summary_rows = [] # rows for metrics csv 

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

        # --- boxplot for each test set using checkpoint---
        metrics = {
            "EPE": epe_list, "% Neg. Jacobian": jac_list, "SSIM": ssim_list,
            "Dice": dice_list, "TRE": tre_list, "Hausdorff": hd_list
        }
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (name, values) in zip(axes.flat, metrics.items()):
            ax.boxplot([v for v in values if not np.isnan(v)])
            ax.set_title(name)
            ax.set_xticks([])
        plt.tight_layout()
        boxplot_path = boxplots_dir / f"seed_{seed}.png"
        plt.savefig(boxplot_path, dpi=150)
        plt.close(fig)
        print(f"Boxplots saved in: {boxplot_path}")

        # --- per case csv ---
        per_case_rows = []
        for key in test_results.keys():
            seq_id, frame_idx = key

            rec = eval_metrics.per_case_reconstructed.get(key, {})
            seg = eval_metrics.per_case_segmentation.get(key, {})

            per_case_rows.append({
                "seq_id": seq_id, "frame_idx": frame_idx,
                "epe": rec.get("epe"), "jacobian": rec.get("jacobian"), "ssim": rec.get("ssim"),
                "dice": seg.get("dice"), "tre": seg.get("tre"), "hausdorff": seg.get("hausdorff"),
                "tumor_area_px": seg.get("tumor_area_px"),
                "bounding_box_diag": seg.get("bbox_diag"),
                "hausdorff_norm_bounding_box": seg.get("hausdorff_norm_bbox"),
                "tre_norm_bounding_box": seg.get("tre_norm_bbox")
            })
        per_case_df = pd.DataFrame(per_case_rows)
        per_case_path = per_case_dir / f"seed_{seed}.csv"
        per_case_df.to_csv(per_case_path, index=False)
        print(f"CSV per case ({len(per_case_df)} rows) saved in: {per_case_path}")

        #  --- computational cost benchmark ---
        print(f"\n--- Computational cost (seed {seed}) ---")
        sample_fixed, sample_moving, _, _ = next(iter(test_loader))
        sample_fixed = sample_fixed[:1].to(device).float()
        sample_moving = sample_moving[:1].to(device).float()
        bench = benchmark_model(model, sample_fixed, sample_moving, device=device)

        # --- summary for metrics and row for csv ---
        summary_rows.append({
            "model": model_name, "seed": seed,
            "epe_mean": np.mean(epe_list), "epe_std": np.std(epe_list),
            "jacobian_mean": np.mean(jac_list), "jacobian_std": np.std(jac_list),
            "ssim_mean": np.mean(ssim_list), "ssim_std": np.std(ssim_list),
            "dice_mean": np.nanmean(dice_list), "dice_std": np.nanstd(dice_list),
            "tre_mean": np.nanmean(tre_list), "tre_std": np.nanstd(tre_list),
            "hausdorff_mean": np.nanmean(hd_list), "hausdorff_std": np.nanstd(hd_list),
            "n_params": bench["n_params"],
            "inference_time_ms_mean": bench["inference_time_ms_mean"],
            "fps": bench["fps"],
            "peak_memory_mb": bench["peak_memory_mb"],
        })

        per_run["epe"].append(np.mean(epe_list))
        per_run["jacobian"].append(np.mean(jac_list))
        per_run["ssim"].append(np.mean(ssim_list))
        per_run["dice"].append(np.nanmean(dice_list))
        per_run["tre"].append(np.nanmean(tre_list))
        per_run["hausdorff"].append(np.nanmean(hd_list))

    print(f"\n{'=' * 60}\n Summary for {model_name} between {len(seeds)} runs (re-evaluated)\n{'=' * 60}")
    for metric_name, values in per_run.items():
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f} (values: {[round(v, 4) for v in values]})")

    # --- summary.csv -> raiz de la carpeta del modelo ---
    summary_df = pd.DataFrame(summary_rows)
    summary_path = run_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved in: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["voxelmorph", "transmorph"], default="voxelmorph")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                         help="Exact seeds used in training, ej: --seeds 2697 3078 5111 6369 8506")
    args = parser.parse_args()
    main(args.model, args.seeds)
