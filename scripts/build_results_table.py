"""
Pulls together the pooled quantitative results from classical B-Spline
registration, VoxelMorph, and TransMorph into one CSV table and one combined
violin plot.

Re-runs inference for all three methods (evaluate_classical_registration.py
and evaluate_checkpoints.py for each architecture) to get current
registration accuracy metrics (Dice, TRE, Hausdorff, EPE, Jacobian, SSIM),
then reads the resulting JSON/CSV files. Compute cost (inference time, FPS,
params) is benchmarked separately, fresh each run, per architecture.

Outputs:
    outputs/results_table.csv               - one row per method
    outputs/pooled_comparison_all_methods.png - Dice/TRE/Hausdorff violins,
        all three methods side by side (classical drawn from ~4977 per-case
        values, VoxelMorph/TransMorph from their 15 pooled outer-test values)
    outputs/paired_significance_test.json - Wilcoxon signed-rank test between
        VoxelMorph and TransMorph on per-outer-fold mean Dice/TRE/Hausdorff
        (paired because both share the same cv_splits outer-fold partition)

Result sources (read after re-running inference):
    outputs/classical_registration/summary.json, per_case.csv
    outputs/nested_cv/voxelmorph/final/outer*_seed*.json   (15 files, pooled)
    outputs/nested_cv/transmorph/final/outer*_seed*.json   (15 files, pooled)

Usage:
    uv run python scripts/build_results_table.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

import config
from nested_cv import load_json, save_json
from src.classical_registration import ClassicalBRegistration
from src.models import build_model
from src.preprocessing import preprocess_dataset
from src.train import benchmark_model
from src.utils import get_device

METRICS = ["dice", "tre", "hausdorff", "epe", "jacobian", "ssim"]


def classical_accuracy():
    path = config.OUTPUTS_DIR / "classical_registration" / "summary.json"
    if not path.exists():
        print(f"Skipping classical accuracy - not found: {path}")
        return None
    s = load_json(path)
    row = {"method": "Classical (B-Spline)", "n": s["n_cases"]}
    for m in METRICS:
        row[f"{m}_mean"] = s[f"{m}_mean"]
        row[f"{m}_std"] = s[f"{m}_std"]
    return row


def dl_accuracy(model_name, label):
    final_dir = config.OUTPUTS_DIR / "nested_cv" / model_name / "final"
    files = sorted(f for f in final_dir.glob("outer*_seed*.json") if "_history" not in f.name)
    if not files:
        print(f"Skipping {model_name} accuracy - no results under {final_dir}")
        return None

    records = [load_json(f) for f in files]
    row = {"method": label, "n": len(records)}
    for m in METRICS:
        values = [r["metrics"][m] for r in records]
        row[f"{m}_mean"] = float(np.nanmean(values))
        row[f"{m}_std"] = float(np.nanstd(values))
    return row


def dl_pooled_values(model_name, metric):
    final_dir = config.OUTPUTS_DIR / "nested_cv" / model_name / "final"
    files = sorted(f for f in final_dir.glob("outer*_seed*.json") if "_history" not in f.name)
    if not files:
        return None
    records = [load_json(f) for f in files]
    return [r["metrics"][metric] for r in records]


def dl_fold_means(model_name, metric):
    """Averages `metric` across the seeds within each outer fold -> one value per fold."""
    final_dir = config.OUTPUTS_DIR / "nested_cv" / model_name / "final"
    files = sorted(f for f in final_dir.glob("outer*_seed*.json") if "_history" not in f.name)
    if not files:
        return None
    records = [load_json(f) for f in files]
    outer_folds = sorted({r["outer"] for r in records})
    return [float(np.mean([r["metrics"][metric] for r in records if r["outer"] == i]))
            for i in outer_folds]


def paired_significance_tests():
    """
    Paired Wilcoxon signed-rank test between VoxelMorph and TransMorph on
    per-outer-fold mean Dice/TRE/Hausdorff - paired because both
    architectures share the same outer-fold patient partition (same
    cv_splits random_state), unlike classical, which has no fold structure
    to pair against.
    """
    results = {}
    for metric in ["dice", "tre", "hausdorff"]:
        vxm = dl_fold_means("voxelmorph", metric)
        tm = dl_fold_means("transmorph", metric)
        if vxm is None or tm is None or len(vxm) != len(tm):
            print(f"Skipping paired test on {metric} - missing or mismatched fold data.")
            continue

        stat, p_value = wilcoxon(vxm, tm)
        results[metric] = {
            "n_folds": len(vxm),
            "voxelmorph_fold_means": vxm,
            "transmorph_fold_means": tm,
            "statistic": float(stat),
            "p_value": float(p_value),
        }
        print(f"Paired Wilcoxon on per-fold mean {metric} (n={len(vxm)} folds): "
              f"stat={stat:.4f}, p={p_value:.4f}")
        if p_value >= 0.05:
            print(f"  (not significant at n=5 folds - report the raw means honestly "
                  f"rather than treating this as a hard gate)")

    if results:
        path = config.OUTPUTS_DIR / "paired_significance_test.json"
        save_json(path, results)
        print(f"Saved: {path}")
    return results


def classical_per_case_values(metric):
    path = config.OUTPUTS_DIR / "classical_registration" / "per_case.csv"
    if not path.exists():
        return None
    values = pd.read_csv(path)[metric].tolist()
    return [v for v in values if not np.isnan(v)]


def combined_violin_plot():
    """
    Saves one figure with Classical/VoxelMorph/TransMorph violins side by
    side for Dice, TRE, and Hausdorff. Classical's violin is drawn from all
    per-case values (~4977); VoxelMorph/TransMorph's from their 15 pooled
    outer-test values - different sample sizes are shown as-is, since each
    reflects what data is actually available for that method.
    """
    sources = {
        "Classical": lambda m: classical_per_case_values(m),
        "VoxelMorph": lambda m: dl_pooled_values("voxelmorph", m),
        "TransMorph": lambda m: dl_pooled_values("transmorph", m),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(axes, ["dice", "tre", "hausdorff"], ["Dice", "TRE (mm)", "Hausdorff (mm)"]):
        labels, data = [], []
        for name, getter in sources.items():
            values = getter(metric)
            if values is None:
                continue
            labels.append(name)
            data.append(values)
        if not data:
            continue
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(title)

    plt.tight_layout()
    path = config.OUTPUTS_DIR / "pooled_comparison_all_methods.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def classical_benchmark(n_runs=10):
    """Times ClassicalBRegistration.register_arrays on a few real cases - CPU only."""
    ram_fixed, ram_moving, _, _ = preprocess_dataset(config.DATA_DIR, ["A_001"])
    n_runs = min(n_runs, len(ram_fixed))

    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        ClassicalBRegistration().register_arrays(ram_fixed[i], ram_moving[i])
        times.append(time.perf_counter() - start)
    times_ms = np.array(times) * 1000

    return {
        "device": "cpu",
        "n_params": None,
        "inference_time_ms_mean": float(times_ms.mean()),
        "inference_time_ms_std": float(times_ms.std()),
        "fps": float(1000 / times_ms.mean()),
        "peak_memory_mb": None,
    }


def dl_benchmark(model_name, device):
    """Times one forward pass of a freshly-built model - cost depends only on
    the architecture, not on which seed's trained weights are loaded."""
    model = build_model(model_name, device)
    h, w = config.TARGET_SIZE
    sample_fixed = torch.randn(1, 1, h, w, device=device)
    sample_moving = torch.randn(1, 1, h, w, device=device)
    bench = benchmark_model(model, sample_fixed, sample_moving, device=device)
    bench["device"] = device
    return bench


def main():
    device = get_device()

    import evaluate_checkpoints
    import evaluate_classical_registration

    print("Running classical registration on the full dataset...")
    evaluate_classical_registration.main()

    print("\nRe-running inference for VoxelMorph's checkpoints...")
    evaluate_checkpoints.main("voxelmorph")

    print("\nRe-running inference for TransMorph's checkpoints...")
    evaluate_checkpoints.main("transmorph")

    print(f"\nBenchmarking compute cost on device: {device}\n")

    accuracy_rows = [
        classical_accuracy(),
        dl_accuracy("voxelmorph", "VoxelMorph"),
        dl_accuracy("transmorph", "TransMorph"),
    ]
    benchmarks = {
        "Classical (B-Spline)": classical_benchmark(),
        "VoxelMorph": dl_benchmark("voxelmorph", device),
        "TransMorph": dl_benchmark("transmorph", device),
    }

    rows = []
    for row in accuracy_rows:
        if row is None:
            continue
        bench = benchmarks.get(row["method"], {})
        rows.append({**row, **bench})

    if not rows:
        print("Nothing found to build a table from.")
        return

    df = pd.DataFrame(rows)
    out_path = config.OUTPUTS_DIR / "results_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}\n")

    display_cols = ["method", "n"] + [f"{m}_mean" for m in METRICS] + [
        "inference_time_ms_mean", "fps", "device"
    ]
    print(df[display_cols].to_string(index=False))

    combined_violin_plot()

    print()
    paired_significance_tests()


if __name__ == "__main__":
    main()
