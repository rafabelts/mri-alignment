"""
    Evaluation for trainned model: full image reconstruction from patches (averaging overlap zones),
    and compute of metrics (EPE, % Jacobian negative, SSIM, Dice, Tre)
"""

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity as ssim
from src.io_utils import hann_weight_map

def inference_with_reconstruction(model, loader, device="cuda"):
    """
        Runs inference over all 'loader' patches and reconstructs them into
        complete images by (seq_id, frame_idx) averaging the areas where multiple
        patches overlap
        """
    model.eval()
    acc = {}
    weight_map = hann_weight_map((256, 256))

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

                w_patch = weight_map[:ph_eff, :pw_eff]

                acc[key]["pred_sum"][py:py+ph_eff, px:px+pw_eff, :] += pred_dvf_np[i, :ph_eff, :pw_eff, :] * w_patch[..., None]
                acc[key]["gt_sum"][py:py+ph_eff, px:px+pw_eff, :] += gt_dvf_np[i, :ph_eff, :pw_eff, :] * w_patch[..., None]
                acc[key]["mask_sum"][py:py+ph_eff, px:px+pw_eff] += mask_np[i, :ph_eff, :pw_eff] * w_patch
                acc[key]["count"][py:py+ph_eff, px:px+pw_eff] += w_patch

        results = {}
        for key, data in acc.items():
            count_safe = np.maximum(data["count"], 1e-6)
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
        """
        Determinant of the Jacobian of the deformation field `x + dvf` at every
        pixel. Values <= 0 indicate local folding (a non-physical, non-invertible
        deformation), used to compute the '% negative Jacobian' metric.
        """
        dy_dy, dy_dx = np.gradient(dvf[..., 1])
        dx_dy, dx_dx = np.gradient(dvf[..., 0])
        return ((dx_dx + 1) * (dy_dy + 1)) - (dx_dy * dy_dx)

    def compute_ssim(self, key, pred_dvf):
        """
        Warps the fixed image with `pred_dvf` and computes SSIM against the real
        moving image, as a reference-based proxy for registration quality
        (independent of the GT DVF). Returns None if the raw images/lookup
        weren't provided to this instance, or if the moving image has zero
        dynamic range.
        """
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

    def evaluate_reconstructed(self, ram_meta):
        """
        Computes EPE (mm, against GT DVF, masked to anatomy), % negative
        Jacobian, and SSIM for every case in `self.results`, printing the
        mean ± std across cases and storing per-case values in
        `self.per_case_reconstructed`.

        Returns
        -------
        epe_list, pct_neg_jac_list, ssim_list : list[float]
        """
        epe_list, pct_neg_jac_list, ssim_list = [], [], []
        self.per_case_reconstructed = {}

        for key, r in self.results.items():
            idx = self.meta_lookup.get(key)
            if idx is None:
                continue

            pred_dvf, gt_dvf, mask = r["pred_dvf"], r["gt_dvf"], r["anatomy_mask"]
            spatial_meta = ram_meta[idx]["spatial_meta"]

            diff_mm = self._pixel_vector_to_physical(pred_dvf - gt_dvf, spatial_meta)
            epe_map = np.linalg.norm(diff_mm, axis=-1)
            epe_val = epe_map[mask].mean() if mask.sum() > 0 else np.nan
            if mask.sum() > 0:
                epe_list.append(epe_val)

            jac = self.jacobian_determinant(pred_dvf)
            jac_val = (jac < 0).sum() / jac.size * 100
            pct_neg_jac_list.append(jac_val)

            ssim_val = self.compute_ssim(key, pred_dvf)
            if ssim_val is not None:
                ssim_list.append(ssim_val)

            self.per_case_reconstructed[key] = {"epe": epe_val, "jacobian": jac_val, "ssim": ssim_val}

        print(f"EPE average (mm): {np.mean(epe_list):.4f} ± {np.std(epe_list):.4f}")
        print(f"% negative jacobian (reconstructed): {np.mean(pct_neg_jac_list):.4f} ± {np.std(pct_neg_jac_list):.4f}")
        print(f"SSIM (moving vs warped): {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")

        return epe_list, pct_neg_jac_list, ssim_list

    @staticmethod
    def _pixel_vector_to_physical(diff, spatial_meta):
        """
        Converts a (H, W, 2) field of (dx, dy) pixel-space displacement
        DIFFERENCES into physical (mm) vectors, via the direction/spacing
        linear map. Unlike `_physical_points`, this has no origin term - a
        vector difference has no absolute position for origin to apply to,
        it cancels out exactly like it does between two transformed points.
        """
        direction = np.array(spatial_meta["direction"]).reshape(3, 3)
        spacing = np.array(spatial_meta["spacing"])

        h, w = diff.shape[:2]
        vec_index = np.zeros((h, w, 3))
        vec_index[..., 0] = diff[..., 0]  # dx, index axis 0
        vec_index[..., 1] = diff[..., 1]  # dy, index axis 1

        return (vec_index * spacing) @ direction.T

    def evaluate_segmentation(self, ram_meta):
        """
        Propagates the tumor segmentation of fixed to moving using the predicted
        DVF (nearest-neighbor, is binary mask), and computes Dice + TRE (distance between
        centroids) against the real segmentation from 'moving'.
        """
        dice_list, tre_list, hd_list = [], [], []
        self.per_case_segmentation = {}

        for key, r in self.results.items():
            idx = self.meta_lookup.get(key)
            if idx is None:
                continue

            meta = ram_meta[idx]
            seg_fixed = meta.get("seg_fixed")
            seg_moving = meta.get("seg_moving")

            if seg_fixed is None or seg_moving is None:
                continue

            spatial_meta = meta["spatial_meta"]

            # segmentation bounding box
            ys, xs = np.where(seg_moving > 0)
            if len(ys) > 0:
                bbox_h = ys.max() - ys.min() + 1
                bbox_w = xs.max() - xs.min() + 1
                bbox_diag = np.sqrt(bbox_h ** 2 + bbox_w ** 2)
                p_min, p_max = self._physical_points([(ys.min(), xs.min()), (ys.max(), xs.max())], spatial_meta)
                bbox_diag_mm = float(np.linalg.norm(p_max - p_min))
            else:
                bbox_h = bbox_w = bbox_diag = bbox_diag_mm = np.nan

            pred_dvf = r["pred_dvf"]
            h, w = seg_fixed.shape
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            new_y = grid_y + pred_dvf[..., 1]
            new_x = grid_x + pred_dvf[..., 0]

            warped_seg = map_coordinates(seg_fixed, [new_y, new_x], order=0, mode="constant")

            intersection = ((warped_seg > 0) & (seg_moving > 0)).sum()
            denom = (warped_seg > 0).sum() + (seg_moving > 0).sum()
            dice_val = 2 * intersection / (denom + 1e-8) if denom > 0 else np.nan
            dice_list.append(dice_val)

            hd_val = np.nan
            # Hausdorff Distance (mm)
            if (warped_seg > 0).any() and (seg_moving > 0).any():
                hd_val = self._hausdorff_mm(warped_seg > 0, seg_moving > 0, spatial_meta)
            hd_list.append(hd_val)

            c_pred = self._centroid(warped_seg)
            c_real = self._centroid(seg_moving)
            tre_val = np.nan
            if c_pred is not None and c_real is not None:
                p_pred, p_real = self._physical_points([c_pred, c_real], spatial_meta)
                tre_val = float(np.linalg.norm(p_pred - p_real))  # mm
            tre_list.append(tre_val)

            self.per_case_segmentation[key] = {
                "dice": dice_val, "tre": tre_val, "hausdorff": hd_val,
                "tumor_area_px": int((seg_moving > 0).sum()),
                "bbox_diag": bbox_diag,
                "bbox_diag_mm": bbox_diag_mm,
                "hausdorff_norm_bbox": hd_val / bbox_diag_mm if bbox_diag_mm else np.nan,
                "tre_norm_bbox": tre_val / bbox_diag_mm if bbox_diag_mm else np.nan,
            }

        print(f"Dice: {np.nanmean(dice_list):.4f} ± {np.nanstd(dice_list):.4f}")
        print(f"Hausdorff Distance (mm): {np.nanmean(hd_list):.4f} ± {np.nanstd(hd_list):.4f}")
        print(f"TRE (mm): {np.nanmean(tre_list):.4f} ± {np.nanstd(tre_list):.4f}")

        return dice_list, tre_list, hd_list

    @staticmethod
    def _hausdorff_mm(mask_a, mask_b, spatial_meta):
        """Standard (symmetric) Hausdorff distance in mm between nonzero pixels
        of two masks, converting pixel coordinates to physical space via the
        image's actual spacing/origin/direction before the nearest-neighbor
        search (correct under anisotropic spacing and non-identity direction,
        not just axis-aligned isotropic spacing)."""
        a_points = EvaluationMetric._physical_points(np.transpose(np.nonzero(mask_a)), spatial_meta)
        b_points = EvaluationMetric._physical_points(np.transpose(np.nonzero(mask_b)), spatial_meta)
        if len(a_points) == 0:
            return 0.0 if len(b_points) == 0 else np.inf
        if len(b_points) == 0:
            return np.inf
        fwd = cKDTree(a_points).query(b_points, k=1)[0]
        bwd = cKDTree(b_points).query(a_points, k=1)[0]
        return max(fwd.max(), bwd.max())

    @staticmethod
    def _centroid(mask):
        """Pixel-coordinate (row, col) centroid of a binary mask, or None if empty."""
        ys, xs = np.where(mask > 0)
        return np.array([ys.mean(), xs.mean()]) if len(ys) > 0 else None

    @staticmethod
    def _physical_points(points_yx, spatial_meta):
        """
        Converts an array of (row, col) pixel coordinates into physical-space
        (mm) coordinates, via a throwaway SimpleITK image carrying the case's
        actual spacing/origin/direction. Delegates the index-to-physical math
        to ITK itself instead of assuming axis-aligned isotropic spacing.
        """
        points_yx = list(points_yx)
        if len(points_yx) == 0:
            return np.empty((0, 3))

        ref = sitk.Image(1, 1, 1, sitk.sitkUInt8)
        ref.SetSpacing(spatial_meta["spacing"])
        ref.SetOrigin(spatial_meta["origin"])
        ref.SetDirection(spatial_meta["direction"])

        # sitk index order is (x=col, y=row, z); every case is a single 2D
        # slice, so z is always 0 - it contributes an identical constant to
        # every point here and cancels out in any distance between them.
        return np.array([
            ref.TransformContinuousIndexToPhysicalPoint((float(x), float(y), 0.0))
            for y, x in points_yx
        ])
