"""
    Trains VoxelMorph with different independent seeds (same split, same loaders),
    and report metrics averaged across runs.

    Usage:
        uv run python scripts/train_multi_seed.py --master-seed 0 --n-runs 5
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.utils import get_device, set_seed
from src.preprocessing import preprocess_dataset
from src.dataset import split_patients, MRICineDataset, build_lookup
from src.models import build_model 
from src.train import train_model
from src.evaluate import inference_with_reconstruction, EvaluationMetric

def main(seeds):
    device = get_device()
    print(f"Using: {device}")

    train_dirs, val_dirs, test_dirs = split_patients(config.DATA_DIR)

    print("\n--- Preprocessing data ---")

    ram_fixed_tr, ram_moving_tr, ram_dvf_tr, ram_meta_tr = preprocess_dataset(config.DATA_DIR, train_dirs)
    ram_fixed_val, ram_moving_val, ram_dvf_val, ram_meta_val = preprocess_dataset(config.DATA_DIR, val_dirs)
    ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te = preprocess_dataset(config.DATA_DIR, test_dirs)

    train_dataset = MRICineDataset(ram_fixed_tr, ram_moving_tr, ram_dvf_tr, ram_meta_tr)
    val_dataset = MRICineDataset(ram_fixed_val, ram_moving_val, ram_dvf_val, ram_meta_val)
    test_dataset = MRICineDataset(ram_fixed_te, ram_moving_te, ram_dvf_te, ram_meta_te)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    meta_lookup_te = build_lookup(ram_meta_te)
    
    per_run = {"epe": [], "jacobian": [], "ssim": [], "dice": [], "tre": [], "hausdorff": []}

    for seed in seeds:
        print(f"\n{'='*60}\n Seed {seed}\n{'='*60}")
        set_seed(seed)

        model = build_model(args.model, device)
        checkpoint_name = f"best_{args.model}_seed{seed}.pt"
        history, checkpoint_path = train_model(model, train_loader, val_loader, device, checkpoint_name=checkpoint_name)
        
        print(f"\n Best checkpoint saved in: {checkpoint_path}")

        # training curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for ax, comp in zip(axes, ["loss", "epe", "smooth"]):
            ax.plot(history[f"train_{comp}"], label="train")
            ax.plot(history[f"val_{comp}"], label="val")
            ax.set_title(comp.upper())
            ax.set_xlabel("epoch")
            ax.legend()
        plt.tight_layout()
        fig_path = config.OUTPUTS_DIR / f"training_curves_{args.model}_seed{seed}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Curves saved in: {fig_path}")


        # load the best checkpoint from run before evaluating
        model.load_state_dict(torch.load(config.CHECKPOINT_DIR / checkpoint_name, map_location=device))
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
    
    print(f"\n{'='*60} \n Summary for {args.model} between {len(seeds)} independent runs (seeds: {seeds})\n{'='*60}")
    
    for metric_name, values in per_run.items():
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f} (values: {[round(v, 4) for v in values]})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=['voxelmorph', 'transmorph'], default='voxelmorph', help='Architecture to train') 

    parser.add_argument("--master-seed", type=int, default=0, help="Master seed: generates N seeds") 
    parser.add_argument("--n-runs", type=int, default=5, help="Number of independent runs")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=args.master_seed)
    seeds = rng.integers(0, 10000, size=args.n_runs).tolist()
 
    print(f"Master seed: {args.master_seed} -> generated seeds: {seeds}")

    main(seeds)
