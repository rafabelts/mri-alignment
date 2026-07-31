"""
Nested cross-validation with hyperparameter search over LAMBDA_SMOOTH and
LEARNING_RATE, for VoxelMorph/TransMorph.

Design:
    outer_k=5, inner_k=3
    grid = LAMBDA_SMOOTH{0.1,0.3,0.5} x LEARNING_RATE{1e-5,1e-4,1e-3} = 9 configs
    search stage capped at 9 epochs (low-fidelity ranking proxy)
    final refits: 3 seeds per outer fold, 50 epochs (diminishing-returns point
        read off the val-loss curves, not early stopping - it never fires here)
    tie-break metric: Dice, computed once after training (not per epoch)

Every unit of work (one search config, one final refit, one best-model
candidate) writes its own result file under outputs/nested_cv/<model>/ right
after it finishes. Re-running this script re-checks those files first and
skips anything already written, so an interrupted run (crash, reboot,
Ctrl+C) can just be restarted with the same command instead of starting over.

Usage:
    uv run python scripts/nested_cv.py --model voxelmorph
    uv run python scripts/nested_cv.py --model transmorph
    uv run python scripts/nested_cv.py --model voxelmorph --plot-only  # re-aggregate + re-plot without training
"""

import os
import sys
import json
import shutil
import argparse
import itertools
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import config
from src.utils import get_device, set_seed
from src.preprocessing import preprocess_dataset
from src.dataset import cv_splits, MRICineDataset, build_lookup
from src.models import build_model
from src.train import train_model
from src.evaluate import inference_with_reconstruction, EvaluationMetric

# --- Nested CV design ---
OUTER_K = 5
INNER_K = 3
LAMBDA_SMOOTH_GRID = [0.1, 0.3, 0.5]
LEARNING_RATE_GRID = [1e-5, 1e-4, 1e-3]
SEARCH_EPOCH_CAP = 9
FINAL_EPOCHS = 50
N_SEEDS = 3
SEARCH_SEED = 0
MASTER_SEED = 0
INTERNAL_VAL_FRACTION = 0.15


def grid_configs():
    return [{"lambda_smooth": ls, "lr": lr}
            for ls, lr in itertools.product(LAMBDA_SMOOTH_GRID, LEARNING_RATE_GRID)]


def seeds_for(n, master_seed=MASTER_SEED):
    rng = np.random.default_rng(seed=master_seed)
    return rng.integers(0, 10000, size=n).tolist()


def subset_by_patients(ram_fixed, ram_moving, ram_dvf, ram_meta, subdirs):
    """Filters the full preprocessed dataset down to the given patient folders."""
    subdirs = set(subdirs)
    idx = [i for i, m in enumerate(ram_meta) if m["seq_id"] in subdirs]
    return ([ram_fixed[i] for i in idx], [ram_moving[i] for i in idx],
             [ram_dvf[i] for i in idx], [ram_meta[i] for i in idx])


def make_loader(ram_fixed, ram_moving, ram_dvf, ram_meta, shuffle):
    dataset = MRICineDataset(ram_fixed, ram_moving, ram_dvf, ram_meta)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)


def internal_train_val_split(subdirs, val_fraction=INTERNAL_VAL_FRACTION, random_state=config.RANDOM_STATE):
    """
    Carves a small validation slice out of `subdirs`, stratified by cohort,
    used only for epoch-level checkpoint selection during training - never
    reported on, and never overlapping with any outer-test set.
    """
    groups = [s.split("_")[0] for s in subdirs]
    try:
        return train_test_split(subdirs, test_size=val_fraction, random_state=random_state, stratify=groups)
    except ValueError:
        # too few members in some cohort (or the resulting split is smaller than
        # the number of cohorts) to stratify; fall back to a plain random split
        return train_test_split(subdirs, test_size=val_fraction, random_state=random_state)


def evaluate_dice(model, loader, ram_fixed, ram_moving, ram_meta, device):
    results = inference_with_reconstruction(model, loader, device=device)
    metric = EvaluationMetric(results, ram_fixed, ram_moving, build_lookup(ram_meta))
    dice_list, _, _ = metric.evaluate_segmentation(ram_meta)
    return float(np.nanmean(dice_list))


def evaluate_full(model, loader, ram_fixed, ram_moving, ram_meta, device):
    results = inference_with_reconstruction(model, loader, device=device)
    metric = EvaluationMetric(results, ram_fixed, ram_moving, build_lookup(ram_meta))
    epe_list, jac_list, ssim_list = metric.evaluate_reconstructed()
    dice_list, tre_list, hd_list = metric.evaluate_segmentation(ram_meta)
    return {
        "epe": float(np.nanmean(epe_list)), "jacobian": float(np.nanmean(jac_list)),
        "ssim": float(np.nanmean(ssim_list)), "dice": float(np.nanmean(dice_list)),
        "tre": float(np.nanmean(tre_list)), "hausdorff": float(np.nanmean(hd_list)),
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _mode_with_tiebreak(values, default):
    counts = Counter(values)
    max_count = max(counts.values())
    tied = [v for v, c in counts.items() if c == max_count]
    return tied[0] if len(tied) == 1 else min(tied, key=lambda v: abs(v - default))


class NestedCVRunner:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.results_dir = config.OUTPUTS_DIR / "nested_cv" / model_name
        self.ram = None

        for sub in ["search", "theta", "final", "best_model", "plots"]:
            (self.results_dir / sub).mkdir(parents=True, exist_ok=True)
        for sub in ["search", "final", "best_model"]:
            (config.CHECKPOINT_DIR / "nested_cv" / model_name / sub).mkdir(parents=True, exist_ok=True)

    def preprocess_all(self):
        """Preprocesses every patient once; folds/configs index into this in RAM."""
        all_subdirs = sorted([d for d in os.listdir(config.DATA_DIR)
                               if os.path.isdir(os.path.join(config.DATA_DIR, d))])
        print(f"Preprocessing {len(all_subdirs)} patients once for the whole nested CV run...")
        self.ram = preprocess_dataset(config.DATA_DIR, all_subdirs)

    # --- search stage (inner CV, hyperparameter ranking) ---

    def run_search_unit(self, outer_i, inner_j, cfg_idx, cfg, inner_train, inner_val):
        assert self.ram is not None, "call preprocess_all() before running search units"
        result_path = self.results_dir / "search" / f"outer{outer_i}_inner{inner_j}_cfg{cfg_idx}.json"
        if result_path.exists():
            return load_json(result_path)["dice"]

        train_data = subset_by_patients(*self.ram, inner_train)
        val_data = subset_by_patients(*self.ram, inner_val)
        train_loader = make_loader(*train_data, shuffle=True)
        val_loader = make_loader(*val_data, shuffle=False)

        set_seed(SEARCH_SEED)
        model = build_model(self.model_name, self.device)
        ckpt_name = f"nested_cv/{self.model_name}/search/outer{outer_i}_inner{inner_j}_cfg{cfg_idx}.pt"
        _, ckpt_path = train_model(
            model, train_loader, val_loader, self.device,
            checkpoint_name=ckpt_name, n_epochs=SEARCH_EPOCH_CAP,
            lr=cfg["lr"], lambda_smooth=cfg["lambda_smooth"],
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model.eval()

        dice = evaluate_dice(model, val_loader, val_data[0], val_data[1], val_data[3], self.device)
        ckpt_path.unlink(missing_ok=True)  # only the score matters for search-stage runs

        save_json(result_path, {"outer": outer_i, "inner": inner_j, "config": cfg, "dice": dice})
        print(f"[search] outer{outer_i} inner{inner_j} cfg{cfg_idx} {cfg} -> dice={dice:.4f}")
        return dice

    def pick_theta(self, outer_i, configs):
        theta_path = self.results_dir / "theta" / f"outer{outer_i}.json"
        if theta_path.exists():
            return load_json(theta_path)["theta"]

        avg_dice = []
        for cfg_idx in range(len(configs)):
            dices = [load_json(self.results_dir / "search" / f"outer{outer_i}_inner{j}_cfg{cfg_idx}.json")["dice"]
                      for j in range(INNER_K)]
            avg_dice.append(float(np.mean(dices)))

        best_idx = int(np.argmax(avg_dice))
        theta = configs[best_idx]
        save_json(theta_path, {"outer": outer_i, "theta": theta,
                                "avg_inner_dice": avg_dice[best_idx], "all_avg_dice": avg_dice})
        print(f"[theta] outer{outer_i} winner: {theta} (avg inner dice={avg_dice[best_idx]:.4f})")
        return theta

    # --- final refit stage (outer CV, reported metrics) ---

    def run_final_refit(self, outer_i, seed_idx, seed, theta, outer_train_subdirs, outer_test_subdirs):
        assert self.ram is not None, "call preprocess_all() before running final refits"
        result_path = self.results_dir / "final" / f"outer{outer_i}_seed{seed_idx}.json"
        if result_path.exists():
            return load_json(result_path)

        refit_train, refit_val = internal_train_val_split(outer_train_subdirs)

        train_data = subset_by_patients(*self.ram, refit_train)
        val_data = subset_by_patients(*self.ram, refit_val)
        test_data = subset_by_patients(*self.ram, outer_test_subdirs)

        train_loader = make_loader(*train_data, shuffle=True)
        val_loader = make_loader(*val_data, shuffle=False)
        test_loader = make_loader(*test_data, shuffle=False)

        set_seed(seed)
        model = build_model(self.model_name, self.device)
        ckpt_name = f"nested_cv/{self.model_name}/final/outer{outer_i}_seed{seed_idx}.pt"
        history, ckpt_path = train_model(
            model, train_loader, val_loader, self.device,
            checkpoint_name=ckpt_name, n_epochs=FINAL_EPOCHS,
            lr=theta["lr"], lambda_smooth=theta["lambda_smooth"],
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model.eval()

        metrics = evaluate_full(model, test_loader, test_data[0], test_data[1], test_data[3], self.device)
        result = {"outer": outer_i, "seed_idx": seed_idx, "seed": seed, "theta": theta,
                  "metrics": metrics, "checkpoint": str(ckpt_path)}

        save_json(result_path, result)
        save_json(self.results_dir / "final" / f"outer{outer_i}_seed{seed_idx}_history.json", history)
        print(f"[final] outer{outer_i} seed{seed_idx}({seed}) -> {metrics}")
        return result

    def run_outer_fold(self, outer_i, fold):
        configs = grid_configs()
        for j, (inner_train, inner_val) in enumerate(fold["inner_folds"]):
            for c, cfg in enumerate(configs):
                self.run_search_unit(outer_i, j, c, cfg, inner_train, inner_val)

        theta = self.pick_theta(outer_i, configs)

        for seed_idx, seed in enumerate(seeds_for(N_SEEDS)):
            self.run_final_refit(outer_i, seed_idx, seed, theta, fold["outer_train"], fold["outer_test"])

    # --- best model (all data, no outer-test involved) ---

    def run_best_model(self, folds):
        assert self.ram is not None, "call preprocess_all() before running the best-model stage"
        result_path = self.results_dir / "best_model" / "summary.json"
        if result_path.exists():
            return load_json(result_path)

        thetas = [load_json(self.results_dir / "theta" / f"outer{i}.json")["theta"] for i in range(OUTER_K)]
        theta_final = {
            "lambda_smooth": _mode_with_tiebreak([t["lambda_smooth"] for t in thetas], default=config.LAMBDA_SMOOTH),
            "lr": _mode_with_tiebreak([t["lr"] for t in thetas], default=config.LEARNING_RATE),
        }
        print(f"[best_model] theta_final (majority vote across {OUTER_K} outer folds): {theta_final}")

        all_subdirs = folds[0]["outer_train"] + folds[0]["outer_test"]
        full_train, full_val = internal_train_val_split(all_subdirs)

        train_data = subset_by_patients(*self.ram, full_train)
        val_data = subset_by_patients(*self.ram, full_val)
        train_loader = make_loader(*train_data, shuffle=True)
        val_loader = make_loader(*val_data, shuffle=False)

        candidates = []
        for seed_idx, seed in enumerate(seeds_for(N_SEEDS)):
            cand_path = self.results_dir / "best_model" / f"seed{seed_idx}.json"
            if cand_path.exists():
                candidates.append(load_json(cand_path))
                continue

            set_seed(seed)
            model = build_model(self.model_name, self.device)
            ckpt_name = f"nested_cv/{self.model_name}/best_model/seed{seed_idx}.pt"
            history, ckpt_path = train_model(
                model, train_loader, val_loader, self.device,
                checkpoint_name=ckpt_name, n_epochs=FINAL_EPOCHS,
                lr=theta_final["lr"], lambda_smooth=theta_final["lambda_smooth"],
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            model.eval()
            dice = evaluate_dice(model, val_loader, val_data[0], val_data[1], val_data[3], self.device)

            cand = {"seed_idx": seed_idx, "seed": seed, "internal_val_dice": dice, "checkpoint": str(ckpt_path)}
            save_json(cand_path, cand)
            save_json(self.results_dir / "best_model" / f"seed{seed_idx}_history.json", history)
            candidates.append(cand)
            print(f"[best_model] seed{seed_idx}({seed}) -> internal val dice={dice:.4f}")

        best = max(candidates, key=lambda c: c["internal_val_dice"])
        selected_path = config.CHECKPOINT_DIR / "nested_cv" / self.model_name / "best_model" / "best_model.pt"
        shutil.copy2(best["checkpoint"], selected_path)
        best["selected_checkpoint"] = str(selected_path)

        result = {"theta_final": theta_final, "candidates": candidates, "chosen": best}
        save_json(result_path, result)
        print(f"[best_model] chosen seed{best['seed_idx']} -> checkpoint: {selected_path}")
        return result

    # --- reporting ---

    def aggregate_and_plot(self):
        final_dir = self.results_dir / "final"
        final_results = sorted(final_dir.glob("outer*_seed*.json"))
        final_results = [f for f in final_results if "_history" not in f.name]
        if not final_results:
            print("No final-refit results persisted yet - nothing to plot.")
            return

        records = [load_json(f) for f in final_results]
        dice = [r["metrics"]["dice"] for r in records]
        tre = [r["metrics"]["tre"] for r in records]
        hd = [r["metrics"]["hausdorff"] for r in records]

        print(f"\n=== Pooled outer-test metrics ({self.model_name}, n={len(records)}) ===")
        print(f"Dice: {np.mean(dice):.4f} +/- {np.std(dice):.4f}")
        print(f"TRE (mm): {np.mean(tre):.4f} +/- {np.std(tre):.4f}")
        print(f"Hausdorff (mm): {np.mean(hd):.4f} +/- {np.std(hd):.4f}")

        plots_dir = self.results_dir / "plots"
        self._violin_plot(dice, tre, hd, plots_dir)
        self._convergence_plot(records, plots_dir)
        self._best_model_plot(plots_dir)

    def _violin_plot(self, dice, tre, hd, plots_dir):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, values, title in zip(axes, [dice, tre, hd], ["Dice", "TRE (mm)", "Hausdorff (mm)"]):
            ax.violinplot([values], showmeans=True, showextrema=True)
            ax.set_xticks([1])
            ax.set_xticklabels([self.model_name])
            ax.set_title(title)
        plt.tight_layout()
        path = plots_dir / f"pooled_outer_test_{self.model_name}.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    def _convergence_plot(self, records, plots_dir):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        colors = plt.cm.tab10.colors
        for r in records:
            history = load_json(self.results_dir / "final" / f"outer{r['outer']}_seed{r['seed_idx']}_history.json")
            color = colors[r["outer"] % len(colors)]
            for ax, comp in zip(axes, ["loss", "epe", "smooth"]):
                ax.plot(history[f"train_{comp}"], color=color, alpha=0.5, linestyle="--", linewidth=0.8)
                ax.plot(history[f"val_{comp}"], color=color, alpha=0.8, linewidth=1.2)
        for ax, comp in zip(axes, ["LOSS", "EPE", "SMOOTH"]):
            ax.set_title(comp)
            ax.set_xlabel("epoch")
        handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], label=f"outer fold {i}")
                   for i in range(OUTER_K)]
        fig.legend(handles=handles, loc="upper center", ncol=OUTER_K, bbox_to_anchor=(0.5, 1.08))
        plt.tight_layout()
        path = plots_dir / f"final_refit_convergence_{self.model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    def _best_model_plot(self, plots_dir):
        best_model_dir = self.results_dir / "best_model"
        history_files = sorted(best_model_dir.glob("seed*_history.json"))
        if not history_files:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        colors = plt.cm.tab10.colors
        for i, hf in enumerate(history_files):
            history = load_json(hf)
            seed_label = hf.stem.split("_")[0]
            color = colors[i % len(colors)]
            for ax, comp in zip(axes, ["loss", "epe", "smooth"]):
                ax.plot(history[f"train_{comp}"], color=color, alpha=0.6, linestyle="--", linewidth=0.8,
                        label=f"{seed_label} train" if comp == "loss" else None)
                ax.plot(history[f"val_{comp}"], color=color, alpha=0.9, linewidth=1.2,
                        label=f"{seed_label} val" if comp == "loss" else None)
        for ax, comp in zip(axes, ["LOSS", "EPE", "SMOOTH"]):
            ax.set_title(comp)
            ax.set_xlabel("epoch")
        axes[0].legend(fontsize=8)
        plt.tight_layout()
        path = plots_dir / f"best_model_convergence_{self.model_name}.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def main(args):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Design: outer_k={OUTER_K} inner_k={INNER_K} grid={len(grid_configs())} "
          f"search_cap={SEARCH_EPOCH_CAP} seeds={N_SEEDS} final_epochs={FINAL_EPOCHS}")

    runner = NestedCVRunner(args.model, device)
    folds = cv_splits(config.DATA_DIR, OUTER_K, INNER_K)

    if not args.plot_only:
        runner.preprocess_all()
        for i, fold in enumerate(folds):
            print(f"\n{'='*60}\n Outer fold {i+1}/{OUTER_K}\n{'='*60}")
            runner.run_outer_fold(i, fold)
        runner.run_best_model(folds)

    runner.aggregate_and_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["voxelmorph", "transmorph"], required=True)
    parser.add_argument("--plot-only", action="store_true",
                         help="Skip training, only aggregate + plot already-persisted results")
    main(parser.parse_args())
