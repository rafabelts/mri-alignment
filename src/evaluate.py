"""
Evaluation for trainned model: full image reconstruction from patches (averaging overlap zones),
and compute of metrics (EPE, % Jacobian negative, SSIM, Dice, Tre)
"""

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity as ssim


def inference_with_reconstruction(model, loader, device="cuda"):
    """
    Runs inference over all 'loader' patches and reconstructs them into
    complete images by (seq_id, frame_idx) averaging the areas where multiple
    patches overlap
    """
    model.eval()
    acc = {}

    with torch.no_grad():
        for img_fixed, img_moving, gt_dvf, meta in loader:
            img_fixed = img_fixed.to(device).float()
            img_moving = img_moving.to(device).float()

            moved, pred_dvf = model(img_fixed, img_moving, registration=True)

            pred_dvf_np = pred_dvf.permute(0, 2, 3, 1).cpu().numpy()
            gt_dvf_np = gt_dvf.permute(0, 2, 3, 1).cpu().numpy()
            mask_np = meta["anatomy_mask"].cpu().numpy()

            bsz = pred_dvf_np.shape[0]
            for i in range(bsz):
                key = (meta["seq_id"][i], meta["frame_idx"][i])
                h = int(meta["original_shape"][0][i].item())
                w = int(meta["original_shape"][1][i].item())
                py = int(meta["patch_coords"][0][i].item())
                px = int(meta["patch_coords"][1][i].item())

                if key not in acc:
                    acc[key] = {
                        "pred_sum": np.zeros((h, w, 2), dtype=np.float32),
                        "gt_sum": np.zeros((h, w, 2), dtype=np.float32),
                        "mask_sum": np.zeros((h, w), dtype=np.float32),
                        "count": np.zeros((h, w), dtype=np.float32),
                    }

                ph, pw = pred_dvf_np.shape[1], pred_dvf_np.shape[2]
                ph_eff = min(ph, h - py)
                pw_eff = min(pw, w - px)

                acc[key]["pred_sum"][py:py + ph_eff, px:px + pw_eff, :] += pred_dvf_np[i, :ph_eff, :pw_eff, :]
                acc[key]["gt_sum"][py:py + ph_eff, px:px + pw_eff, :] += gt_dvf_np[i, :ph_eff, :pw_eff, :]
                acc[key]["mask_sum"][py:py + ph_eff, px:px + pw_eff] += mask_np[i, :ph_eff, :pw_eff]
                acc[key]["count"][py:py + ph_eff, px:px + pw_eff] += 1

    results = {}
    for key, data in acc.items():
        count_safe = np.maximum(data["count"], 1)
        results[key] = {
            "pred_dvf": data["pred_sum"] / count_safe[..., None],
            "gt_dvf": data["gt_sum"] / count_safe[..., None],
            "anatomy_mask": (data["mask_sum"] / count_safe) > 0.5,
        }
    return results


class EvaluationMetric:
    """Computes EPE, % Jacobian negative, SSIM, Dice and TRE over `results`
    (ouput of `inference_with_reconstruction`)."""

    def __init__(self, results, ram_fixed=None, ram_moving=None, meta_lookup=None):
        self.results = results
        self.ram_fixed = ram_fixed
        self.ram_moving = ram_moving
        self.meta_lookup = meta_lookup

    @staticmethod
    def jacobian_determinant(dvf):
        dy_dy, dy_dx = np.gradient(dvf[..., 1])
        dx_dy, dx_dx = np.gradient(dvf[..., 0])
        return ((dx_dx + 1) * (dy_dy + 1)) - (dx_dy * dy_dx)

    def compute_ssim(self, key, pred_dvf):
        if self.ram_fixed is None or self.meta_lookup is None:
            return None

        idx = self.meta_lookup.get(key)
        if idx is None:
            return None

        img_fixed_np = self.ram_fixed[idx]
        img_moving_np = self.ram_moving[idx]

        h, w = img_fixed_np.shape
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        new_y = grid_y + pred_dvf[..., 1]
        new_x = grid_x + pred_dvf[..., 0]

        # warp fixed -> approximate moving (same convention as GT)
        warped = map_coordinates(img_fixed_np, [new_y, new_x], order=1, mode="constant")

        data_range = img_moving_np.max() - img_moving_np.min()
        if data_range == 0:
            return None

        return ssim(img_moving_np, warped, data_range=data_range)

    def evaluate_reconstructed(self):
        epe_list, pct_neg_jac_list, ssim_list = [], [], []

        for key, r in self.results.items():
            pred_dvf, gt_dvf, mask = r["pred_dvf"], r["gt_dvf"], r["anatomy_mask"]

            diff = pred_dvf - gt_dvf
            epe_map = np.sqrt((diff ** 2).sum(axis=-1))
            if mask.sum() > 0:
                epe_list.append(epe_map[mask].mean())

            jac = self.jacobian_determinant(pred_dvf)
            pct_neg_jac_list.append((jac < 0).sum() / jac.size * 100)

            ssim_val = self.compute_ssim(key, pred_dvf)
            if ssim_val is not None:
                ssim_list.append(ssim_val)

        print(f"EPE average (reconstructed): {np.mean(epe_list):.4f} ± {np.std(epe_list):.4f}")
        print(f"% negative jacobian (reconstructed): {np.mean(pct_neg_jac_list):.4f} ± {np.std(pct_neg_jac_list):.4f}")
        print(f"SSIM (moving vs warped): {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")

        return epe_list, pct_neg_jac_list, ssim_list

    def evaluate_segmentation(self, ram_meta):
        """
        Propagates the tumor segmentation of fixed to moving using the predicted
        DVF (nearest-neighbor, is binary mask), and computes Dice + TRE (distance between
        centroids) against the real segmentation from 'moving'.
        """
        dice_list, tre_list = [], []

        for key, r in self.results.items():
            idx = self.meta_lookup.get(key)
            if idx is None:
                continue

            meta = ram_meta[idx]
            seg_fixed = meta.get("seg_fixed")
            seg_moving = meta.get("seg_moving")
            if seg_fixed is None or seg_moving is None:
                continue

            pred_dvf = r["pred_dvf"]
            h, w = seg_fixed.shape
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            new_y = grid_y + pred_dvf[..., 1]
            new_x = grid_x + pred_dvf[..., 0]

            warped_seg = map_coordinates(seg_fixed, [new_y, new_x], order=0, mode="constant")

            intersection = ((warped_seg > 0) & (seg_moving > 0)).sum()
            denom = (warped_seg > 0).sum() + (seg_moving > 0).sum()
            dice_list.append(2 * intersection / (denom + 1e-8) if denom > 0 else np.nan)

            c_pred = self._centroid(warped_seg)
            c_real = self._centroid(seg_moving)
            if c_pred is not None and c_real is not None:
                tre_list.append(np.linalg.norm(c_pred - c_real))

        print(f"Dice: {np.nanmean(dice_list):.4f} ± {np.nanstd(dice_list):.4f}")
        print(f"TRE: {np.nanmean(tre_list):.4f} ± {np.nanstd(tre_list):.4f}")

        return dice_list, tre_list

    @staticmethod
    def _centroid(mask):
        ys, xs = np.where(mask > 0)
        return np.array([ys.mean(), xs.mean()]) if len(ys) > 0 else None
