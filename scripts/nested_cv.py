"""
Nested cross-validation with an Optuna-driven hyperparameter search for
VoxelMorph and CNNTransformerSVF2D.

Design:
    outer_k=5, inner_k=3
    search space: learning_rate, lambda_smooth, int_steps;
                  lambda_dvf fixed at 1.0
    TRIAL_BUDGET Optuna trials per outer fold, pruned via inner_fold_0's
        per-epoch val loss (MedianPruner)
    final refits: 3 seeds per outer fold, 50 epochs (diminishing-returns
        point read off the val-loss curves, not early stopping)
    clinical selection metric: TRE (mm), computed once after training (not per epoch)

Every unit of work writes its own result file under outputs/nested_cv/<model>/
right after it finishes (the Optuna study itself is the persistence layer for
the search stage). Re-running this script skips anything already written, so
an interrupted run can just be restarted with the same command.

Usage:
    uv run python scripts/nested_cv.py --model voxelmorph
    uv run python scripts/nested_cv.py --model cnn_transformer_svf_2d --devices cuda:1
    uv run python scripts/nested_cv.py --model voxelmorph --plot-only
    uv run python scripts/nested_cv.py --model voxelmorph --devices cuda:0
"""

import argparse
import json
import os
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config
from src.dataset import MRICineDataset, build_lookup, cv_splits
from src.evaluate import EvaluationMetric, inference_with_reconstruction
from src.models import build_model
from src.preprocessing import preprocess_dataset
from src.train import benchmark_model, train_model
from src.utils import get_device, set_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)
LOGGER = logging.getLogger("mri_alignment")

# --- Nested CV design ---
OUTER_K = 5
INNER_K = 3
TRIAL_BUDGET = 15
SEARCH_EPOCH_CAP = 10
FINAL_EPOCHS = 50
N_SEEDS = 3
SEARCH_SEED = 0
MASTER_SEED = 0
INTERNAL_VAL_FRACTION = 0.15

# --- search space ---
LR_RANGE = (1e-5, 1e-3)
LAMBDA_SMOOTH_RANGE = (1e-3, 1e-1)
INT_STEPS_CHOICES = [5, 7]


def suggest_theta(trial, model_name):
    theta = {
        "learning_rate": trial.suggest_float("learning_rate", *LR_RANGE, log=True),
        "lambda_smooth": trial.suggest_float("lambda_smooth", *LAMBDA_SMOOTH_RANGE, log=True),
        "int_steps": trial.suggest_categorical("int_steps", INT_STEPS_CHOICES),
    }
    return theta


def seeds_for(n, master_seed=MASTER_SEED):
    rng = np.random.default_rng(seed=master_seed)
    return rng.integers(0, 10000, size=n).tolist()


def subset_by_patients(ram_fixed, ram_moving, ram_dvf, ram_meta, subdirs):
    """Filters the full preprocessed dataset down to the given patient folders."""
    subdirs = set(subdirs)
    idx = [i for i, m in enumerate(ram_meta) if m["seq_id"] in subdirs]
    return (
        [ram_fixed[i] for i in idx],
        [ram_moving[i] for i in idx],
        [ram_dvf[i] for i in idx],
        [ram_meta[i] for i in idx],
    )


def make_loader(ram_fixed, ram_moving, ram_dvf, ram_meta, shuffle, batch_size=config.BATCH_SIZE):
    dataset = MRICineDataset(ram_fixed, ram_moving, ram_dvf, ram_meta)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.NUM_WORKERS > 0,
    }
    if config.NUM_WORKERS > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def internal_train_val_split(
    subdirs, val_fraction=INTERNAL_VAL_FRACTION, random_state=config.RANDOM_STATE
):
    """
    Carves a small validation slice out of `subdirs`, stratified by cohort,
    used only for epoch-level checkpoint selection during training - never
    reported on, and never overlapping with any outer-test set.
    """
    groups = [s.split("_")[0] for s in subdirs]
    try:
        return train_test_split(
            subdirs, test_size=val_fraction, random_state=random_state, stratify=groups
        )
    except ValueError:
        # too few members in some cohort (or the resulting split is smaller than
        # the number of cohorts) to stratify; fall back to a plain random split
        return train_test_split(
            subdirs, test_size=val_fraction, random_state=random_state
        )


def evaluate_tre(model, loader, ram_fixed, ram_moving, ram_meta, device):
    """Return mean target TRE in physical mm, or +inf if none is finite."""
    results = inference_with_reconstruction(model, loader, device=device)
    metric = EvaluationMetric(results, ram_fixed, ram_moving, build_lookup(ram_meta))
    _, tre_list, _ = metric.evaluate_segmentation(ram_meta)
    finite_tre = np.asarray(tre_list, dtype=float)
    finite_tre = finite_tre[np.isfinite(finite_tre)]
    return float(np.nanmean(finite_tre)) if finite_tre.size else float("inf")


def evaluate_full(model, loader, ram_fixed, ram_moving, ram_meta, device):
    """Returns (metrics_summary, metric) - `metric` still holds the per-case
    detail (per_case_reconstructed/per_case_segmentation) for callers that
    want to save it, instead of only the pooled means."""
    results = inference_with_reconstruction(model, loader, device=device)
    metric = EvaluationMetric(results, ram_fixed, ram_moving, build_lookup(ram_meta))
    epe_list, jac_list, ssim_list = metric.evaluate_reconstructed(ram_meta)
    dice_list, tre_list, hd95_list = metric.evaluate_segmentation(ram_meta)
    summary = {
        "epe": float(np.nanmean(epe_list)),
        "jacobian": float(np.nanmean(jac_list)),
        "ssim": float(np.nanmean(ssim_list)),
        "dice": float(np.nanmean(dice_list)),
        "tre": float(np.nanmean(tre_list)),
        "hd95": float(np.nanmean(hd95_list)),
    }
    return summary, metric


def save_per_case_csv(metric, path):
    rows = []
    for key in metric.results:
        seq_id, frame_idx = key
        rec = metric.per_case_reconstructed.get(key, {})
        seg = metric.per_case_segmentation.get(key, {})
        rows.append({
            "seq_id": seq_id, "frame_idx": frame_idx,
            "epe": rec.get("epe"), "jacobian": rec.get("jacobian"), "ssim": rec.get("ssim"),
            "dice": seg.get("dice"), "tre": seg.get("tre"), "hd95": seg.get("hd95"),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.3f")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def configure_logging(model_name, device):
    """One console and one model/device-specific append-only file handler."""
    artifact = "proposed" if model_name == "cnn_transformer_svf_2d" else model_name
    device_slug = device.replace(":", "")
    log_dir = config.OUTPUTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{artifact}_{device_slug}.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    if not LOGGER.handlers:
        formatter = logging.Formatter("%(asctime)s %(message)s")
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(console)
        LOGGER.addHandler(file_handler)
    return log_path


class NestedCVRunner:
    def __init__(self, model_name, devices):
        """`devices` contains exactly one device assigned to this process."""
        self.model_name = model_name
        self.devices = devices
        self.device = devices[0]
        if len(devices) != 1:
            raise ValueError("nested_cv requires exactly one device per process")
        self.artifact_name = "proposed" if model_name == "cnn_transformer_svf_2d" else model_name
        self.tag = "PROPOSED" if self.artifact_name == "proposed" else "VXM"
        self.results_dir = config.OUTPUTS_DIR / "nested_cv" / self.artifact_name
        self.ram = None

        for sub in ["search", "theta", "final", "plots"]:
            (self.results_dir / sub).mkdir(parents=True, exist_ok=True)
        for sub in ["search", "final"]:
            (config.CHECKPOINT_DIR / "nested_cv" / self.artifact_name / sub).mkdir(
                parents=True, exist_ok=True
            )

    def preprocess_all(self):
        """Preprocesses every patient once; folds/configs index into this in RAM."""
        all_subdirs = sorted(
            [
                d
                for d in os.listdir(config.DATA_DIR)
                if os.path.isdir(os.path.join(config.DATA_DIR, d))
            ]
        )
        logging.getLogger("mri_alignment").info(
            f"Preprocessing {len(all_subdirs)} patients once for the whole nested CV run..."
        )
        self.ram = preprocess_dataset(config.DATA_DIR, all_subdirs)

    # --- search stage (inner CV, Optuna hyperparameter search) ---

    def _search_objective(self, trial, outer_i, fold):
        assert self.ram is not None, "call preprocess_all() before running search trials"
        theta = suggest_theta(trial, self.model_name)

        device = self.device
        try:
            tre_scores = []
            for j, (inner_train, inner_val) in enumerate(fold["inner_folds"]):
                train_data = subset_by_patients(*self.ram, inner_train)
                val_data = subset_by_patients(*self.ram, inner_val)
                train_loader = make_loader(*train_data, shuffle=True)
                val_loader = make_loader(*val_data, shuffle=False)

                # set_seed() reseeds torch's global RNG, which is shared across
                # threads - under concurrent search trials this means model init
                # isn't perfectly reproducible relative to other in-flight trials.
                # Acceptable here: search-stage only needs relative ranking
                # between configs, not bit-exact reproducibility (unlike the
                # final-refit/best-model seeds, which stay single-threaded).
                set_seed(SEARCH_SEED)
                model = build_model(self.model_name, device, int_steps=theta["int_steps"])

                def prune_callback(epoch, val_metrics, inner_j=j):
                    if inner_j == 0:
                        trial.report(val_metrics["loss"], step=epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                ckpt_name = (
                    f"nested_cv/{self.artifact_name}/search/"
                    f"tmp_outer{outer_i}_trial{trial.number}_inner{j}.pt"
                )
                _, ckpt_path = train_model(
                    model, train_loader, val_loader, device,
                    checkpoint_name=ckpt_name, n_epochs=SEARCH_EPOCH_CAP,
                    lr=theta["learning_rate"], lambda_smooth=theta["lambda_smooth"],
                    epoch_callback=prune_callback,
                    log_prefix=(f"[{self.tag}][{device}][search][outer={outer_i}]"
                                f"[trial={trial.number}][inner={j}] "),
                )
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True)
                )
                model.eval()

                tre = evaluate_tre(
                    model, val_loader, val_data[0], val_data[1], val_data[3], device
                )
                ckpt_path.unlink(missing_ok=True)  # only the score matters for search-stage runs
                tre_scores.append(tre)
        finally:
            torch.cuda.empty_cache()

        mean_tre = float(np.mean(tre_scores))
        logging.getLogger("mri_alignment").info(
            f"[search] outer{outer_i} trial{trial.number} (device={device}) {theta} "
            f"-> mean TRE = {mean_tre:.3f} mm "
            f"({len(tre_scores)}/{INNER_K} inner folds ran)"
        )
        return mean_tre

    def run_search_stage(self, outer_i, fold):
        theta_path = self.results_dir / "theta" / f"outer{outer_i}.json"
        if theta_path.exists():
            saved = load_json(theta_path)
            if saved.get("objective") != "mean_inner_validation_tre_mm":
                raise RuntimeError(
                    f"{theta_path} predates the TRE objective. Archive/remove the old "
                    "theta file before restarting TRE-based HPO."
                )
            expected = {"learning_rate", "lambda_smooth", "int_steps"}
            if set(saved.get("theta", {})) == expected:
                return saved["theta"]
            LOGGER.warning("Ignoring legacy theta schema at %s; new search will replace it", theta_path)

        storage = f"sqlite:///{self.results_dir / 'search' / f'outer{outer_i}.db'}"
        study = optuna.create_study(
            # A distinct name allows legacy Dice-maximization studies to remain
            # in the same SQLite database without a direction conflict.
            study_name=f"{self.artifact_name}_outer{outer_i}_tre_search_v2",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEARCH_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2),
        )
        remaining = TRIAL_BUDGET - len(study.trials)
        if remaining > 0:
            study.optimize(
                lambda trial: self._search_objective(trial, outer_i, fold),
                n_trials=remaining, n_jobs=1,
            )

        theta = study.best_params
        save_json(theta_path, {
            "outer": outer_i, "theta": theta,
            "objective": "mean_inner_validation_tre_mm",
            "best_value": study.best_value, "n_trials": len(study.trials),
        })
        states = Counter(t.state for t in study.trials)
        logging.getLogger("mri_alignment").info(
            "[%s][%s][search][outer=%d] COMPLETE completed_trials=%d "
            "pruned_trials=%d best_TRE=%.3f mm best_learning_rate=%.6g "
            "best_lambda_smooth=%.6g best_int_steps=%d",
            self.tag, self.device, outer_i,
            states[optuna.trial.TrialState.COMPLETE],
            states[optuna.trial.TrialState.PRUNED], study.best_value,
            theta["learning_rate"], theta["lambda_smooth"], theta["int_steps"],
        )
        return theta

    # --- final refit stage (outer CV, reported metrics) ---

    def run_final_refit(
        self, outer_i, seed_idx, seed, theta, outer_train_subdirs, outer_test_subdirs
    ):
        assert self.ram is not None, "call preprocess_all() before running final refits"
        result_path = self.results_dir / "final" / f"outer{outer_i}_seed{seed_idx}.json"
        if result_path.exists():
            saved = load_json(result_path)
            if "hd95" not in saved.get("metrics", {}):
                raise RuntimeError(
                    f"{result_path} uses the legacy Hausdorff schema. Re-evaluate "
                    "the checkpoint with scripts/evaluate_checkpoints.py."
                )
            return saved

        refit_train, refit_val = internal_train_val_split(outer_train_subdirs)

        train_data = subset_by_patients(*self.ram, refit_train)
        val_data = subset_by_patients(*self.ram, refit_val)
        test_data = subset_by_patients(*self.ram, outer_test_subdirs)

        train_loader = make_loader(*train_data, shuffle=True)
        val_loader = make_loader(*val_data, shuffle=False)
        test_loader = make_loader(*test_data, shuffle=False)

        set_seed(seed)
        model = build_model(self.model_name, self.device, int_steps=theta["int_steps"])
        ckpt_name = (
            f"nested_cv/{self.artifact_name}/final/outer{outer_i}_seed{seed_idx}.pt"
        )
        history, ckpt_path = train_model(
            model, train_loader, val_loader, self.device,
            checkpoint_name=ckpt_name, n_epochs=FINAL_EPOCHS,
            lr=theta["learning_rate"], lambda_smooth=theta["lambda_smooth"],
            patience=FINAL_EPOCHS + 1,
            log_prefix=f"[{self.tag}][{self.device}][final][outer={outer_i}][seed={seed_idx}] ",
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        model.eval()

        metrics, metric = evaluate_full(
            model, test_loader, test_data[0], test_data[1], test_data[3], self.device
        )

        # compute cost is architecture-dependent, not case-dependent (unlike
        # `metrics`), so it's measured once per refit here rather than per
        # row in the per-case CSV - but it DOES depend on this fold's own
        # theta (batch_size, vxm_int_steps can differ per outer fold), so
        # it's not redundant to measure it again for each one
        sample_fixed, sample_moving, _, _ = next(iter(test_loader))
        sample_fixed = sample_fixed[:1].to(self.device, non_blocking=True).float()
        sample_moving = sample_moving[:1].to(self.device, non_blocking=True).float()
        benchmark = benchmark_model(
            model, sample_fixed, sample_moving, device=self.device,
            log_prefix=f"[{self.tag}][{self.device}][final][outer={outer_i}][seed={seed_idx}] ",
        )

        result = {
            "outer": outer_i,
            "seed_idx": seed_idx,
            "seed": seed,
            "theta": theta,
            "metrics": metrics,
            "benchmark": benchmark,
            "checkpoint": str(ckpt_path),
        }

        save_json(result_path, result)
        save_json(
            self.results_dir / "final" / f"outer{outer_i}_seed{seed_idx}_history.json",
            history,
        )
        save_per_case_csv(
            metric, self.results_dir / "final" / f"outer{outer_i}_seed{seed_idx}_per_case.csv"
        )
        logging.getLogger("mri_alignment").info(
            "[%s][%s][final][outer=%d][seed=%d] metrics=%s",
            self.tag, self.device, outer_i, seed_idx, metrics,
        )
        return result

    def run_outer_fold(self, outer_i, fold):
        theta = self.run_search_stage(outer_i, fold)

        for seed_idx, seed in enumerate(seeds_for(N_SEEDS)):
            self.run_final_refit(
                outer_i, seed_idx, seed, theta, fold["outer_train"], fold["outer_test"]
            )

    # --- reporting ---

    def aggregate_and_plot(self):
        final_dir = self.results_dir / "final"
        final_results = sorted(final_dir.glob("outer*_seed*.json"))
        final_results = [f for f in final_results if "_history" not in f.name]
        if not final_results:
            print("No final-refit results persisted yet - nothing to plot.")
            return

        records = [load_json(f) for f in final_results]
        detail_rows = [{
            "model": self.model_name, "outer_fold": r["outer"], "seed": r["seed"],
            "tre_mm": r["metrics"]["tre"], "dvf_epe_mm": r["metrics"]["epe"],
            "dice": r["metrics"]["dice"], "hd95_mm": r["metrics"]["hd95"],
            "negative_jacobian_pct": r["metrics"]["jacobian"],
            "ssim": r["metrics"]["ssim"],
            "inference_time_ms": r["benchmark"]["inference_time_ms_mean"],
            "fps": r["benchmark"]["fps"],
        } for r in records]
        pd.DataFrame(detail_rows).to_csv(
            self.results_dir / "detailed_outer_fold_seed_results.csv",
            index=False, float_format="%.3f",
        )

        metric_columns = [
            "tre_mm", "dvf_epe_mm", "dice", "hd95_mm",
            "negative_jacobian_pct", "ssim", "inference_time_ms", "fps",
        ]
        detail_df = pd.DataFrame(detail_rows)
        fold_df = (
            detail_df.groupby(["model", "outer_fold"], as_index=False)
            .agg(
                n_seeds=("seed", "nunique"),
                **{column: (column, "mean") for column in metric_columns},
            )
            .sort_values("outer_fold")
        )
        fold_path = self.results_dir / "fold_summary.csv"
        fold_df.to_csv(fold_path, index=False, float_format="%.3f")

        global_row = {"model": self.model_name, "n_outer_folds": len(fold_df)}
        for column in metric_columns:
            global_row[f"{column}_mean"] = float(np.nanmean(fold_df[column]))
            global_row[f"{column}_std"] = float(np.nanstd(fold_df[column]))
        global_path = self.results_dir / "global_summary.csv"
        pd.DataFrame([global_row]).to_csv(
            global_path, index=False, float_format="%.3f"
        )
        print(f"Saved fold-level summary: {fold_path}")
        print(f"Saved global fold-aggregated summary: {global_path}")

        dice = fold_df["dice"].tolist()
        tre = fold_df["tre_mm"].tolist()
        hd95 = fold_df["hd95_mm"].tolist()

        print(
            f"\n=== Outer-test metrics ({self.model_name}; seeds averaged within fold, "
            f"n={len(dice)} folds) ==="
        )
        print(f"Dice: {np.mean(dice):.4f} +/- {np.std(dice):.4f}")
        print(f"TRE (mm): {np.mean(tre):.4f} +/- {np.std(tre):.4f}")
        print(f"HD95 (mm): {np.mean(hd95):.4f} +/- {np.std(hd95):.4f}")

        plots_dir = self.results_dir / "plots"
        self._violin_plot(dice, tre, hd95, plots_dir)
        self._convergence_plot(records, plots_dir)

    def _violin_plot(self, dice, tre, hd95, plots_dir):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, values, title in zip(
            axes, [dice, tre, hd95], ["Dice", "TRE (mm)", "HD95 (mm)"]
        ):
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
            history = load_json(
                self.results_dir
                / "final"
                / f"outer{r['outer']}_seed{r['seed_idx']}_history.json"
            )
            color = colors[r["outer"] % len(colors)]
            for ax, comp in zip(axes, ["loss", "epe", "smooth"]):
                ax.plot(
                    history[f"train_{comp}"],
                    color=color,
                    alpha=0.5,
                    linestyle="--",
                    linewidth=0.8,
                )
                ax.plot(history[f"val_{comp}"], color=color, alpha=0.8, linewidth=1.2)
        for ax, comp in zip(axes, ["LOSS", "EPE", "SMOOTH"]):
            ax.set_title(comp)
            ax.set_xlabel("epoch")
        handles = [
            plt.Line2D([0], [0], color=colors[i % len(colors)], label=f"outer fold {i}")
            for i in range(OUTER_K)
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=OUTER_K,
            bbox_to_anchor=(0.5, 1.08),
        )
        plt.tight_layout()
        path = plots_dir / f"final_refit_convergence_{self.model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

def main(args):
    devices = args.devices.split(",") if args.devices else [get_device()]
    if len(devices) != 1:
        raise SystemExit("Exactly one --devices value is required; internal GPU parallelism is disabled")
    device = devices[0]
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA is unavailable; cannot run on {device}")
        index = torch.device(device).index or 0
        if index >= torch.cuda.device_count():
            raise SystemExit(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) exist")
        torch.cuda.set_device(index)
        # Fixed-size patches benefit from cuDNN autotuning. This can cause small
        # numerical differences and therefore is not bit-exact reproducible.
        torch.backends.cudnn.benchmark = True
    log_path = configure_logging(args.model, device)
    LOGGER.info("[%s][%s][startup] outer_k=%d inner_k=%d trials=%d search_epochs=%d "
                "final_epochs=%d seeds=%d batch_size=%d num_workers=%d use_amp=%s log=%s",
                "PROPOSED" if args.model == "cnn_transformer_svf_2d" else "VXM",
                device, OUTER_K, INNER_K, TRIAL_BUDGET, SEARCH_EPOCH_CAP,
                FINAL_EPOCHS, N_SEEDS, config.BATCH_SIZE, config.NUM_WORKERS,
                config.USE_AMP, log_path)

    runner = NestedCVRunner(args.model, devices)
    folds = cv_splits(config.DATA_DIR, OUTER_K, INNER_K)

    if not args.plot_only:
        runner.preprocess_all()
        for i, fold in enumerate(folds):
            print(f"\n{'=' * 60}\n Outer fold {i + 1}/{OUTER_K}\n{'=' * 60}")
            runner.run_outer_fold(i, fold)

    runner.aggregate_and_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["voxelmorph", "cnn_transformer_svf_2d"],
        required=True,
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training, only aggregate + plot already-persisted results",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Exactly one process-local device, e.g. 'cuda:0', 'cuda:1', or 'cpu'.",
    )
    main(parser.parse_args())
