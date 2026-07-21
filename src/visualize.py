"""
Funciones de visualización: patches individuales por cohorte (para
inspección de preprocesamiento), y comparación fixed/moving/warped/DVF
sobre imágenes reconstruidas (para inspección cualitativa del modelo).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_patches_per_cohort(dataloader, n_per_cohort=2, cohorts=("A", "B", "C"),
                                  stride=20, min_magnitude=0.3, cmap="viridis"):
    """Muestra `n_per_cohort` patches de cada cohorte (fixed, moving, DVF)."""
    counts = {c: 0 for c in cohorts}
    total_shown = 0
    total_target = n_per_cohort * len(cohorts)

    for batch_data in dataloader:
        images_fixed, images_moving, dvfs, meta_dict = batch_data

        for i in range(images_fixed.size(0)):
            if total_shown >= total_target:
                return

            patient_id = meta_dict["seq_id"][i]
            cohort = patient_id[0]
            if cohort not in counts or counts[cohort] >= n_per_cohort:
                continue

            counts[cohort] += 1
            total_shown += 1

            img_fixed = torch.nan_to_num(images_fixed[i].squeeze(0)).cpu().numpy()
            img_moving = torch.nan_to_num(images_moving[i].squeeze(0)).cpu().numpy()

            dvf = dvfs[i].permute(1, 2, 0).cpu().numpy()
            ux, uy = dvf[:, :, 0], dvf[:, :, 1]
            dvf_magnitude = np.sqrt(ux ** 2 + uy ** 2)

            anatomy_mask = meta_dict["anatomy_mask"][i].cpu().numpy()
            dvf_magnitude = dvf_magnitude * anatomy_mask
            ux_masked, uy_masked = ux * anatomy_mask, uy * anatomy_mask

            try:
                coords = (meta_dict["patch_coords"][0][i].item(), meta_dict["patch_coords"][1][i].item())
                title_text = f"Patch #{total_shown} | {patient_id} (cohort {cohort})\nCoords: {coords}"
            except Exception:
                title_text = f"Patch #{total_shown} | {patient_id} (cohort {cohort})"

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].imshow(img_fixed, cmap="gray"); axes[0].set_title(title_text); axes[0].axis("off")
            axes[1].imshow(img_moving, cmap="gray"); axes[1].set_title("Moving Patch"); axes[1].axis("off")

            im_mag = axes[2].imshow(dvf_magnitude, cmap=cmap, vmin=0, interpolation="none")
            axes[2].set_title("DVF"); fig.colorbar(im_mag, ax=axes[2], shrink=0.7, label="Displacement")

            y_grid, x_grid = np.mgrid[0:dvf.shape[0]:stride, 0:dvf.shape[1]:stride]
            ux_s = ux_masked[0:dvf.shape[0]:stride, 0:dvf.shape[1]:stride]
            uy_s = uy_masked[0:dvf.shape[0]:stride, 0:dvf.shape[1]:stride]
            mag_s = dvf_magnitude[0:dvf.shape[0]:stride, 0:dvf.shape[1]:stride]

            valid = mag_s > min_magnitude
            ux_dir, uy_dir = np.zeros_like(ux_s), np.zeros_like(uy_s)
            ux_dir[valid] = ux_s[valid] / mag_s[valid]
            uy_dir[valid] = uy_s[valid] / mag_s[valid]

            axes[2].quiver(x_grid, y_grid, ux_dir, -uy_dir, color="white", pivot="middle",
                            scale=25, scale_units="inches", width=0.004, headwidth=4,
                            headlength=5, alpha=0.85)
            axes[2].axis("off")
            plt.tight_layout()
            plt.show()


def visualize_reconstruction(test_results, ram_fixed, ram_moving, meta_lookup,
                              n_show=3, keys=None, one_per_cohort=False,
                              show_profiles=True, profile_row=210, profile_col=210,
                              overlap_start=167, overlap_end=256):
    """
    Muestra fixed/moving/warped/DVF para casos de test ya reconstruidos, con
    métricas de MSE (warped vs moving, y baseline sin warp), y opcionalmente
    perfiles de línea cruzando la zona de traslape de patches (para
    verificar que la reconstrucción no introduce discontinuidades).
    """
    if keys is None:
        if one_per_cohort:
            keys, seen_cohorts = [], set()
            for k in test_results.keys():
                cohort = k[0][0]
                if cohort not in seen_cohorts:
                    keys.append(k)
                    seen_cohorts.add(cohort)
                if len(seen_cohorts) >= 3:
                    break
        else:
            keys = list(test_results.keys())[:n_show]

    for key in keys:
        seq_id, frame_idx = key
        idx = meta_lookup.get(key)
        if idx is None:
            print(f"No se encontró imagen completa para {key}")
            continue

        img_fixed_np = ram_fixed[idx]
        img_moving_np = ram_moving[idx]
        pred_dvf = test_results[key]["pred_dvf"]
        mask = test_results[key]["anatomy_mask"]

        h, w = img_fixed_np.shape
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        new_y = grid_y + pred_dvf[..., 1]
        new_x = grid_x + pred_dvf[..., 0]

        # warp fixed -> aproxima moving (misma convención que el GT)
        warped = map_coordinates(img_fixed_np, [new_y, new_x], order=1, mode="constant")
        dvf_mag = np.sqrt((pred_dvf ** 2).sum(axis=-1)) * mask

        diff_sq = (img_moving_np - warped) ** 2
        mse_masked = diff_sq[mask > 0].mean() if mask.sum() > 0 else float("nan")

        diff_sq_baseline = (img_fixed_np - img_moving_np) ** 2
        mse_baseline_masked = diff_sq_baseline[mask > 0].mean() if mask.sum() > 0 else float("nan")

        print(f"\n=== {seq_id} frame {frame_idx} (cohort {seq_id[0]}) ===")
        print(f"  MSE (moving vs warped)         -> masked: {mse_masked:.4f}")
        print(f"  MSE (fixed vs moving, no warp) -> masked: {mse_baseline_masked:.4f}")
        print(f"  Relative upgrade (warped): {100 * (1 - mse_masked / mse_baseline_masked):.1f}%")

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_fixed_np, cmap="gray")
        axes[0].set_title(f"Fixed\n{seq_id} frame {frame_idx} (cohort {seq_id[0]})"); axes[0].axis("off")
        axes[1].imshow(img_moving_np, cmap="gray"); axes[1].set_title("Moving"); axes[1].axis("off")
        axes[2].imshow(warped, cmap="gray")
        axes[2].set_title(f"Warped (fixed→moving)\nMSE={mse_masked:.4f}"); axes[2].axis("off")

        im = axes[3].imshow(dvf_mag, cmap="viridis", aspect="auto")
        axes[3].set_title("Predicted DVF magnitude"); axes[3].axis("off")
        divider = make_axes_locatable(axes[3])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.show()

        if show_profiles:
            dvf_mag_full = np.sqrt((pred_dvf ** 2).sum(axis=-1))
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(dvf_mag_full[profile_row, :])
            axes[0].axvline(overlap_start, color="red", linestyle="--", label="start overlap")
            axes[0].axvline(overlap_end, color="red", linestyle="--", label="end overlap")
            axes[0].set_title(f"Horizontal profile (row {profile_row}) - {seq_id}"); axes[0].legend()

            axes[1].plot(dvf_mag_full[:, profile_col])
            axes[1].axvline(overlap_start, color="red", linestyle="--")
            axes[1].axvline(overlap_end, color="red", linestyle="--")
            axes[1].set_title(f"Vertical profile (column {profile_col}) - {seq_id}")
            plt.tight_layout()
            plt.show()
