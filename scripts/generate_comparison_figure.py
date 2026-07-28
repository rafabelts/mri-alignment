import sys
import argparse
import pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours
from torch.utils.data import DataLoader

import config
from src.utils import get_device
from src.preprocessing import preprocess_dataset
from src.dataset import MRICineDataset, build_lookup
from src.models import build_model
from src.evaluate import inference_with_reconstruction
from src.io_utils import save_as_mha, denormalize

DEFAULT_CASES = ["A_024:095", "B_021:017", "C_008:007"]

def warp_image(img_fixed_np, pred_dvf):
    h, w = img_fixed_np.shape
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    new_y = grid_y + pred_dvf[..., 1]
    new_x = grid_x + pred_dvf[..., 0]

    return map_coordinates(img_fixed_np, [new_y, new_x], order=1, mode="constant")

def propagate_segmentation(seg_fixed, pred_dvf):
    h, w = seg_fixed.shape
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    new_y = grid_y + pred_dvf[..., 1]
    new_x = grid_x + pred_dvf[..., 0]
    return map_coordinates(seg_fixed, [new_y, new_x], order=0, mode="constant")

def epe_map(pred_dvf, gt_dvf, mask):
    diff = pred_dvf - gt_dvf
    epe = np.sqrt((diff ** 2).sum(axis=1))
    return epe * mask

def plot_contour(ax, mask, color, linewidth=1.5):
    for contour in find_contours(mask.astype(float), level=0.5):
        ax.plot(contour[:, 1], contour[:,0], color=color, linewidth=linewidth)

def get_single_case(seq_id, frame_idx):
    ram_fixed, ram_moving, ram_dvf, ram_meta = preprocess_dataset(config.DATA_DIR, [seq_id])
    lookup = build_lookup(ram_meta)
    key = (seq_id, frame_idx)

    if key not in lookup:
        raise ValueError(f"Case {key} not found")
    
    idx = lookup[key]
    meta = ram_meta[idx]

    dataset = MRICineDataset([ram_fixed[idx]], [ram_moving[idx]], [ram_dvf[idx]], [meta])
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return {
        "fixed_np": ram_fixed[idx], "moving_np": ram_moving[idx],
        "gt_dvf": ram_dvf[idx], "meta": meta, "loader": loader,
    }

def run_dl_model(model_name, checkpoint_name, loader, device):
    checkpoint_path = config.CHECKPOINT_DIR / checkpoint_name
    model = build_model(model_name, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    results = inference_with_reconstruction(model, loader, device=device)
    key = list(results.keys())[0] # singe case in loader
    return results[key]["pred_dvf"], results[key]["anatomy_mask"]

def generate_figure_for_case(patient, frame, vxm_checkpoint, tm_checkpoint, device):
    print(f"{'=' * 60} Case: {patient} frame {frame}\n{'=' * 60}")
    
    ase = get_single_case(patient, frame)
    fixed_np, moving_np, gt_dvf = case["fixed_np"], case["moving_np"], case["gt_dvf"]
    meta = case["meta"]
    seg_fixed, seg_moving = meta["seg_fixed"], meta["seg_moving"]
    norm_stats, spatial_meta = meta["norm_stats"], meta["spatial_meta"]

    export_dir = config.OUTPUTS_DIR / "comparison_exports" / f"{patient}_{frame}"
    export_dir.mkdir(parents=True, exist_ok=True)

    fixed_clinical = denormalize(fixed_np, norm_stats["mean_fixed"], norm_stats["std_fixed"])
    moving_clinical = denormalize(moving_np, norm_stats["mean_moving"], norm_stats["std_moving"])
    save_as_mha(fixed_clinical, spatial_meta, export_dir / "fixed.mha")
    save_as_mha(moving_clinical, spatial_meta, export_dir / "moving.mha")

    print("Running VoxelMorph")
    pred_dvf_vxm, mask_vxm = run_dl_model("voxelmorph", vxm_checkpoint, case["loader"], device)

    print("Running TransMorph")
    pred_dvf_tm, mask_tm = run_dl_model("transmorph", tm_checkpoint, case["loader"], device)

    methods = [
        ("voxelmorph", "VoxelMorph (CNN)", pred_dvf_vxm, mask_vxm),
        ("transmorph", "TransMorph (ViT)", pred_dvf_tm, mask_tm),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    axes[0, 0].imshow(fixed_np, cmap="gray")
    plot_contour(axes[0, 0], seg_fixed, color="lime")
    axes[0, 0].set_title(f"Reference / Target\n{patient} frame {frame}")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for col, (slug, name, pred_dvf, mask) in enumerate(methods, start=1):
        warped = warp_image(fixed_np, pred_dvf)
        propagated_seg = propagate_segmentation(seg_fixed, pred_dvf)
        error_map = epe_map(pred_dvf, gt_dvf, mask)

        warped_clinical = denormalize(warped, norm_stats["mean_fixed"], norm_stats["std_fixed"])
        save_as_mha(warped_clinical, spatial_meta, export_dir / f"warped_{slug}.mha")
        save_as_mha(pred_dvf, spatial_meta, export_dir / f"dvf_{slug}.mha")
        save_as_mha(propagated_seg.astype(np.uint8), spatial_meta, export_dir / f"seg_propagated_{slug}.mha")

        axes[0, col].imshow(warped, cmap="gray")
        plot_contour(axes[0, col], seg_moving, color="lime")
        plot_contour(axes[0, col], propagated_seg, color="red")
        axes[0, col].set_title(name)
        axes[0, col].axis("off")

        im = axes[1, col].imshow(error_map, cmap="inferno", vmin=0, vmax=10)
        axes[1, col].axis("off")
        divider = make_axes_locatable(axes[1, col])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label="EPE")

    plt.tight_layout()
    output_path = config.OUTPUTS_DIR / f"comparison_{patient}_{frame}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Figure saved in: {output_path}")
    print(f"Exports .mha saved in {export_dir}")

def main(cases, vxm_checkpoint, tm_checkpoint):
    device = get_device()
    print(f"Usando dispositivo: {device}")

    for case_str in cases:
        patient, frame = case_str.split(":")
        generate_figure_for_case(patient, frame, vxm_checkpoint, tm_checkpoint, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, nargs="+", default=DEFAULT_CASES,
                         help="Formato patient:frame, ej: A_024:095 B_021:017 C_008:007")
    parser.add_argument("--vxm-checkpoint", type=str, default="best_voxelmorph.pt")
    parser.add_argument("--tm-checkpoint", type=str, default="best_transmorph.pt")
    args = parser.parse_args()

    main(args.cases, args.vxm_checkpoint, args.tm_checkpoint)
