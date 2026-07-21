"""
Training loop, and computational cost benchmark
"""

import time

import numpy as np
import torch
import torch.optim as optim

from config import (
    LEARNING_RATE, N_EPOCHS, PATIENCE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
    GRAD_CLIP_MAX_NORM, LAMBDA_DVF, LAMBDA_SMOOTH, CHECKPOINT_DIR,
)
from src.losses import Loss


def train_model(model, train_loader, val_loader, device,
                 checkpoint_name="best_model.pt",
                 n_epochs=N_EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,
                 lambda_dvf=LAMBDA_DVF, lambda_smooth=LAMBDA_SMOOTH,
                 scheduler_factor=SCHEDULER_FACTOR, scheduler_patience=SCHEDULER_PATIENCE,
                 grad_clip_max_norm=GRAD_CLIP_MAX_NORM):
    """
    Trains 'model' with direct supervision (EPE + smoothness) against the
    real DVF, with early stopping and LR reduction in plateau.

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
        "val_loss": [], "val_epe": [], "val_smooth": [],
    }

    for epoch in range(n_epochs):
        train_metrics = _run_epoch(model, train_loader, device, lambda_dvf, lambda_smooth,
                                    optimizer=optimizer, grad_clip_max_norm=grad_clip_max_norm)
        val_metrics = _run_epoch(model, val_loader, device, lambda_dvf, lambda_smooth, optimizer=None)

        scheduler.step(val_metrics["loss"])

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"  train -> loss: {train_metrics['loss']:.4f} | epe: {train_metrics['epe']:.4f} "
              f"| smooth: {train_metrics['smooth']:.4f}")
        print(f"  val   -> loss: {val_metrics['loss']:.4f} | epe: {val_metrics['epe']:.4f} "
              f"| smooth: {val_metrics['smooth']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> best model (val_loss={val_metrics['loss']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> early stopping (patience={patience})")
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
            img_fixed = img_fixed.to(device).float()
            img_moving = img_moving.to(device).float()
            gt_dvf = gt_dvf.to(device).float()
            mask = meta["anatomy_mask"].to(device).float()

            if is_train:
                optimizer.zero_grad()

            # source=fixed, target=moving
            moved, pred_dvf = model(img_fixed, img_moving, registration=True)

            loss_fn = Loss(pred_dvf, gt_dvf, mask, lambda_dvf=lambda_dvf, lambda_smooth=lambda_smooth)
            loss, parts = loss_fn.total_loss()

            if is_train:
                if torch.isnan(loss):
                    print("  NaN loss, discarding batch")
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


def benchmark_model(model, sample_fixed, sample_moving, device="cuda", n_warmup=10, n_runs=50):
    """
    Measures the inference time, parameters number, and GPU memory peak of
    a trained model. `sample_fixed`/`sample_moving` must be tensors (1, 1, H, W)
    in the right device (batch_size=1 to reflect a frame-per-frame processing)
    """
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_fixed, sample_moving, registration=True)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample_fixed, sample_moving, registration=True)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # ms

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else None

    results = {
        "n_params": n_params,
        "n_params_trainable": n_params_trainable,
        "inference_time_ms_mean": times.mean(),
        "inference_time_ms_std": times.std(),
        "fps": 1000 / times.mean(),
        "peak_memory_mb": peak_memory_mb,
    }

    print(f"Total params: {n_params:,} ({n_params_trainable:,} trainable)")
    print(f"Inference time: {results['inference_time_ms_mean']:.2f} ± {results['inference_time_ms_std']:.2f} ms")
    print(f"FPS equivalente: {results['fps']:.2f}")
    if peak_memory_mb is not None:
        print(f"GPU memory peak: {peak_memory_mb:.1f} MB")

    return results
