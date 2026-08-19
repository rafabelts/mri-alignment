"""
Training loop, and computational cost benchmark
"""

import logging
import time

import numpy as np
import torch
import torch.optim as optim

from config import (
    LEARNING_RATE, N_EPOCHS, PATIENCE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
    GRAD_CLIP_MAX_NORM, LAMBDA_DVF, LAMBDA_SMOOTH, CHECKPOINT_DIR,
)
from src.losses import Loss

LOGGER = logging.getLogger("mri_alignment")


def train_model(model, train_loader, val_loader, device,
                 checkpoint_name="best_model.pt",
                 n_epochs=N_EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,
                 lambda_dvf=LAMBDA_DVF, lambda_smooth=LAMBDA_SMOOTH,
                 scheduler_factor=SCHEDULER_FACTOR, scheduler_patience=SCHEDULER_PATIENCE,
                 grad_clip_max_norm=GRAD_CLIP_MAX_NORM, epoch_callback=None, log_prefix=""):
    """
    Trains 'model' with direct supervision (EPE + smoothness) against the
    real DVF, with early stopping and LR reduction in plateau.

    `epoch_callback(epoch, val_metrics)`, if given, is called after every
    epoch (e.g. for Optuna pruning) - raising from it stops training
    immediately, propagating out of this function as-is. Keeps this module
    unaware of what the callback actually does or raises.

    `log_prefix` is prepended to every printed line - since multiple calls
    to this function can run concurrently in different threads (e.g. Optuna
    trials on different GPUs), their epoch logs interleave in the shared
    console otherwise, and look like duplicated epochs instead of separate
    runs.

    Returns
    -------
    history: dict with train/val curves for epoch
    checkpoint_path: Path to the best saved checkpoint
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name

    history = {
        "train_loss": [], "train_epe": [], "train_smooth": [],
        "val_loss": [], "val_epe": [], "val_smooth": [], "epoch_time": [],
    }

    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()

        train_metrics = _run_epoch(model, train_loader, device, lambda_dvf, lambda_smooth,
                                    optimizer=optimizer, grad_clip_max_norm=grad_clip_max_norm)

        val_metrics = _run_epoch(model, val_loader, device, lambda_dvf, lambda_smooth, optimizer=None)

        epoch_time = time.perf_counter() - epoch_start

        scheduler.step(val_metrics["loss"])

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)
        history["epoch_time"].append(epoch_time)

        if epoch_callback is not None:
            epoch_callback(epoch, val_metrics)

        LOGGER.info(
            "%sepoch=%d/%d train_loss=%.4f val_loss=%.4f val_epe=%.4f "
            "lr=%.3g epoch_time=%.1fs",
            log_prefix, epoch + 1, n_epochs, train_metrics["loss"],
            val_metrics["loss"], val_metrics["epe"],
            optimizer.param_groups[0]["lr"], epoch_time,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOGGER.info("%searly_stopping patience=%d", log_prefix, patience)
                break

    return history, checkpoint_path


def _run_epoch(model, loader, device, lambda_dvf, lambda_smooth, optimizer=None, grad_clip_max_norm=None):
    """Runs a train or validation (if optimizer is None) epoch"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running = {"loss": 0.0, "epe": 0.0, "smooth": 0.0}
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for img_fixed, img_moving, gt_dvf, meta in loader:
            img_fixed = img_fixed.to(device, non_blocking=True).float()
            img_moving = img_moving.to(device, non_blocking=True).float()
            gt_dvf = gt_dvf.to(device, non_blocking=True).float()
            mask = meta["anatomy_mask"].to(device, non_blocking=True).float()

            if is_train:
                optimizer.zero_grad()

            # source=fixed, target=moving
            moved, pred_dvf = model(img_fixed, img_moving, registration=True)

            loss_fn = Loss(pred_dvf, gt_dvf, mask, lambda_dvf=lambda_dvf, lambda_smooth=lambda_smooth)
            loss, parts = loss_fn.total_loss()

            if is_train:
                if torch.isnan(loss):
                    LOGGER.warning("NaN loss; discarding batch")
                    continue
                loss.backward()
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                optimizer.step()

            running["loss"] += loss.item()
            running["epe"] += parts["epe"]
            running["smooth"] += parts["smooth"]

    n = len(loader)
    return {k: v / n for k, v in running.items()}


def benchmark_model(model, sample_fixed, sample_moving, device="cuda", n_warmup=10,
                    n_runs=50, log_prefix=""):
    """
    Measures the inference time, parameters number, and GPU memory peak of
    a trained model. `sample_fixed`/`sample_moving` must be tensors (1, 1, H, W)
    in the right device (batch_size=1 to reflect a frame-per-frame processing)
    """
    model.eval()

    # "cuda", "cuda:0", "cuda:1", etc. all count as CUDA - a plain "== cuda"
    # check would silently skip synchronization/memory tracking (and always
    # report peak_memory_mb=None) for any indexed device string
    is_cuda = torch.device(device).type == "cuda"

    n_params = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_fixed, sample_moving, registration=True)

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if is_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(sample_fixed, sample_moving, registration=True)
            if is_cuda:
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # ms

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if is_cuda else None

    results = {
        "n_params": n_params,
        "n_params_trainable": n_params_trainable,
        "inference_time_ms_mean": times.mean(),
        "inference_time_ms_std": times.std(),
        "fps": 1000 / times.mean(),
        "peak_memory_mb": peak_memory_mb,
    }

    LOGGER.info("%sparams=%d trainable_params=%d inference_time=%.2fms fps=%.2f",
                log_prefix, n_params, n_params_trainable,
                results["inference_time_ms_mean"], results["fps"])
    if peak_memory_mb is not None:
        LOGGER.info("%sgpu_peak_memory=%.1fMB", log_prefix, peak_memory_mb)

    return results
