"""
Principal script, preprocesses the dataset, generates the splits/loades, trains the model,
and saves the checkpoint + training curves

Use:
    uv run python scripts/train_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from src.utils import get_device
from src.preprocessing import preprocess_dataset
from src.dataset import split_patients, MRICineDataset
from src.models import build_voxelmorph
from src.train import train_model


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_dirs, val_dirs, test_dirs = split_patients(config.DATA_DIR)

    print("\Preprocessing train...")
    ram_fixed_tr, ram_moving_tr, ram_dvf_tr, ram_meta_tr = preprocess_dataset(config.DATA_DIR, train_dirs)
    print("Preprocessing val...")
    ram_fixed_val, ram_moving_val, ram_dvf_val, ram_meta_val = preprocess_dataset(config.DATA_DIR, val_dirs)

    train_dataset = MRICineDataset(ram_fixed_tr, ram_moving_tr, ram_dvf_tr, ram_meta_tr)
    val_dataset = MRICineDataset(ram_fixed_val, ram_moving_val, ram_dvf_val, ram_meta_val)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"\nTotal train pairs: {len(ram_meta_tr)}")
    print(f"Total val pairs:   {len(ram_meta_val)}")

    model = build_voxelmorph(device)

    history, checkpoint_path = train_model(
        model, train_loader, val_loader, device,
        checkpoint_name="best_voxelmorph.pt",
    )
    print(f"\nBest checkpoint saved in: {checkpoint_path}")

    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, comp in zip(axes, ["loss", "epe", "smooth"]):
        ax.plot(history[f"train_{comp}"], label="train")
        ax.plot(history[f"val_{comp}"], label="val")
        ax.set_title(comp.upper())
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    fig_path = config.OUTPUTS_DIR / "training_curves.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Curves saved in: {fig_path}")


if __name__ == "__main__":
    main()
