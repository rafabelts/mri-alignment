"""
Re-evaluates already-trained checkpoints from nested_cv.py with the current
EvaluationMetric code, without retraining.

For each of the OUTER_K folds, evaluates that fold's N_SEEDS checkpoints
against that fold's own outer-test patients, overwrites the corresponding
metrics under outputs/nested_cv/<model>/final/, saves a boxplot and a
per-case CSV per checkpoint under outputs/nested_cv/<model>/reevaluation/,
runs a computational-cost benchmark, and regenerates the pooled plots.

Usage:
    uv run python scripts/reevaluate_checkpoints.py --model voxelmorph
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nested_cv import (
    INNER_K,
    N_SEEDS,
    OUTER_K,
    NestedCVRunner,
    load_json,
    make_loader,
    save_json,
    subset_by_patients,
)

import config
from src.dataset import build_lookup, cv_splits
from src.evaluate import EvaluationMetric, inference_with_reconstruction
from src.models import build_model
from src.train import benchmark_model
from src.utils import get_device


def main(model_name):
    device = get_device()
    print(f"Using: {device}")

    runner = NestedCVRunner(model_name, device)
    runner.preprocess_all()
    folds = cv_splits(config.DATA_DIR, OUTER_K, INNER_K)

    reeval_dir = runner.results_dir / "reevaluation"
    (reeval_dir / "boxplots").mkdir(parents=True, exist_ok=True)
    (reeval_dir / "per_case").mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for i, fold in enumerate(folds):
        ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te = subset_by_patients(
            *runner.ram, fold["outer_test"]
        )
        test_loader = make_loader(
            ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te, shuffle=False
        )
        meta_lookup_te = build_lookup(ram_meta_te)

        for seed_idx in range(N_SEEDS):
            result_path = runner.results_dir / "final" / f"outer{i}_seed{seed_idx}.json"
            if not result_path.exists():
                print(
                    f"outer{i} seed{seed_idx}: no existing result, skipping (run nested_cv.py first)"
                )
                continue

            old_result = load_json(result_path)
            ckpt_path = Path(old_result["checkpoint"])
            print(
                f"\n{'=' * 60}\n outer{i} seed{seed_idx} ({ckpt_path.name})\n{'=' * 60}"
            )

            model = build_model(model_name, device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            test_results = inference_with_reconstruction(
                model, test_loader, device=device
            )
            eval_metrics = EvaluationMetric(
                test_results, ram_fixed_te, ram_moving_te, meta_lookup_te
            )
            epe_list, jac_list, ssim_list = eval_metrics.evaluate_reconstructed()
            dice_list, tre_list, hd_list = eval_metrics.evaluate_segmentation(
                ram_meta_te
            )

            metrics = {
                "epe": float(np.nanmean(epe_list)),
                "jacobian": float(np.nanmean(jac_list)),
                "ssim": float(np.nanmean(ssim_list)),
                "dice": float(np.nanmean(dice_list)),
                "tre": float(np.nanmean(tre_list)),
                "hausdorff": float(np.nanmean(hd_list)),
            }
            old_result["metrics"] = metrics
            save_json(result_path, old_result)

            # boxplot for this checkpoint's outer-test cases
            plot_metrics = {
                "EPE": epe_list,
                "% Neg. Jacobian": jac_list,
                "SSIM": ssim_list,
                "Dice": dice_list,
                "TRE": tre_list,
                "Hausdorff": hd_list,
            }
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for ax, (name, values) in zip(axes.flat, plot_metrics.items()):
                ax.boxplot([v for v in values if not np.isnan(v)])
                ax.set_title(name)
                ax.set_xticks([])
            plt.tight_layout()
            plt.savefig(
                reeval_dir / "boxplots" / f"outer{i}_seed{seed_idx}.png", dpi=150
            )
            plt.close(fig)

            # per-case csv
            per_case_rows = []
            for key in test_results.keys():
                seq_id, frame_idx = key
                rec = eval_metrics.per_case_reconstructed.get(key, {})
                seg = eval_metrics.per_case_segmentation.get(key, {})
                per_case_rows.append(
                    {
                        "seq_id": seq_id,
                        "frame_idx": frame_idx,
                        "epe": rec.get("epe"),
                        "jacobian": rec.get("jacobian"),
                        "ssim": rec.get("ssim"),
                        "dice": seg.get("dice"),
                        "tre": seg.get("tre"),
                        "hausdorff": seg.get("hausdorff"),
                        "tumor_area_px": seg.get("tumor_area_px"),
                        "bounding_box_diag_mm": seg.get("bbox_diag_mm"),
                    }
                )
            pd.DataFrame(per_case_rows).to_csv(
                reeval_dir / "per_case" / f"outer{i}_seed{seed_idx}.csv", index=False
            )

            # computational cost benchmark
            sample_fixed, sample_moving, _, _ = next(iter(test_loader))
            sample_fixed = sample_fixed[:1].to(device).float()
            sample_moving = sample_moving[:1].to(device).float()
            bench = benchmark_model(model, sample_fixed, sample_moving, device=device)

            summary_rows.append(
                {
                    "outer": i,
                    "seed_idx": seed_idx,
                    **metrics,
                    "n_params": bench["n_params"],
                    "fps": bench["fps"],
                    "peak_memory_mb": bench["peak_memory_mb"],
                }
            )
            print(f"outer{i} seed{seed_idx} -> {metrics}")

    summary_path = reeval_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    print("\nRegenerating pooled plots with updated metrics...")
    runner.aggregate_and_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["voxelmorph", "transmorph"], required=True
    )
    args = parser.parse_args()
    main(args.model)
