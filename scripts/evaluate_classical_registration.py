"""
Evaluates ClassicalBRegistration (SimpleITK B-Spline) on the full TrackRad
dataset with the same metrics used for the deep-learning models (EPE, %
negative Jacobian, SSIM, Dice, TRE, Hausdorff) - a directly comparable
baseline number against the nested_cv.py pooled results.

Unlike the deep-learning pipeline, no patching/stitching is needed here:
classical registration has no fixed input-size constraint, so each frame is
registered at its full (already-preprocessed) size directly.

Usage:
    uv run python scripts/evaluate_classical_registration.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.classical_registration import ClassicalBRegistration
from src.dataset import build_lookup
from src.evaluate import EvaluationMetric
from src.preprocessing import preprocess_dataset


def main():
    all_subdirs = sorted(
        d for d in os.listdir(config.DATA_DIR) if os.path.isdir(os.path.join(config.DATA_DIR, d))
    )
    print(f"Preprocessing {len(all_subdirs)} patients...")
    ram_fixed, ram_moving, ram_dvf, ram_meta = preprocess_dataset(config.DATA_DIR, all_subdirs)
    print(f"{len(ram_fixed)} fixed/moving pairs to register.")

    results = {}
    reg_times = {}
    start = time.perf_counter()
    for i, meta in enumerate(ram_meta):
        key = (meta["seq_id"], meta["frame_idx"])
        case_start = time.perf_counter()
        pred_dvf = ClassicalBRegistration().register_arrays(ram_fixed[i], ram_moving[i])
        reg_times[key] = time.perf_counter() - case_start
        results[key] = {
            "pred_dvf": pred_dvf,
            "gt_dvf": ram_dvf[i],
            "anatomy_mask": meta["anatomy_mask"] > 0.5,
        }
        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - start
            print(f"{i + 1}/{len(ram_meta)} registered ({elapsed:.0f}s elapsed)")

    total_time = time.perf_counter() - start
    print(f"\nAll {len(results)} cases registered in {total_time:.0f}s")

    meta_lookup = build_lookup(ram_meta)
    metric = EvaluationMetric(results, ram_fixed, ram_moving, meta_lookup)
    epe_list, jac_list, ssim_list = metric.evaluate_reconstructed(ram_meta)
    dice_list, tre_list, hd_list = metric.evaluate_segmentation(ram_meta)

    out_dir = config.OUTPUTS_DIR / "classical_registration"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_case_rows = []
    for key in results:
        seq_id, frame_idx = key
        rec = metric.per_case_reconstructed.get(key, {})
        seg = metric.per_case_segmentation.get(key, {})
        per_case_rows.append({
            "seq_id": seq_id, "frame_idx": frame_idx,
            "epe": rec.get("epe"), "jacobian": rec.get("jacobian"), "ssim": rec.get("ssim"),
            "dice": seg.get("dice"), "tre": seg.get("tre"), "hausdorff": seg.get("hausdorff"),
            "reg_time_s": reg_times.get(key),
        })
    per_case_path = out_dir / "per_case.csv"
    pd.DataFrame(per_case_rows).to_csv(per_case_path, index=False)
    print(f"Per-case CSV saved: {per_case_path}")

    reg_time_ms = np.array(list(reg_times.values())) * 1000
    summary = {
        "n_cases": len(results),
        "dice_mean": float(np.nanmean(dice_list)), "dice_std": float(np.nanstd(dice_list)),
        "tre_mean": float(np.nanmean(tre_list)), "tre_std": float(np.nanstd(tre_list)),
        "hausdorff_mean": float(np.nanmean(hd_list)), "hausdorff_std": float(np.nanstd(hd_list)),
        "epe_mean": float(np.mean(epe_list)), "epe_std": float(np.std(epe_list)),
        "jacobian_mean": float(np.mean(jac_list)), "jacobian_std": float(np.std(jac_list)),
        "ssim_mean": float(np.mean(ssim_list)), "ssim_std": float(np.std(ssim_list)),
        # same field names as src/train.py's benchmark_model(), for a direct
        # side-by-side table with the DL models - "n_params"/"peak_memory_mb"
        # don't apply to a non-learned, CPU-only method, so left None/noted.
        "device": "cpu",
        "n_params": None,
        "inference_time_ms_mean": float(reg_time_ms.mean()),
        "inference_time_ms_std": float(reg_time_ms.std()),
        "fps": float(1000 / reg_time_ms.mean()),
        "peak_memory_mb": None,
        "total_time_s": float(total_time),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")
    print(summary)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, title in zip(axes, [dice_list, tre_list, hd_list], ["Dice", "TRE (mm)", "Hausdorff (mm)"]):
        clean = [v for v in values if not np.isnan(v)]
        ax.violinplot([clean], showmeans=True, showextrema=True)
        ax.set_xticks([1])
        ax.set_xticklabels(["classical"])
        ax.set_title(title)
    plt.tight_layout()
    plot_path = out_dir / "pooled_metrics_classical.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
