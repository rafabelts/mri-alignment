"""
Data preprocessing: .mha file lecture, normalization, anatomy mask generation,
padding, and patches extraction for images bigger than model objective size. 
"""

import os
import glob

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_closing, binary_fill_holes

from config import TARGET_SIZE, EPSILON_BG


def preprocess_dataset(root_dir, subdirs, target_size=TARGET_SIZE, epsilon_bg=EPSILON_BG):
    """
    Reads the .mha files of each patient and returns numpy array lists with its
    metadata, ready to be wrapped in a PyTorch Dataset.

    Parameters
    ----------
    root_dir : str o Path
        Dataset root folder (ej. .../TrackRad).
    subdirs : list[str]
        Patient folder names to include (ej. ['A_001', 'A_004']).
    target_size : tuple[int, int]
        Minimum image size; applies padding if image is smaller.
    epsilon_bg : float
        A near-zero thresold to distinguish real background from real
        anatomy.

    Returns
    -------
    fixed_list, moving_list, dvf_list, metadata_list
    """
    fixed_list, moving_list, dvf_list, metadata_list = [], [], [], []

    root_dir = str(root_dir)
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Folder not found: {root_dir}")

    for subdir in subdirs:
        cine_dir = os.path.join(root_dir, subdir, "SynthesizedCine")
        dvf_dir = os.path.join(root_dir, subdir, "DVFReverse")

        if not os.path.exists(cine_dir) or not os.path.exists(dvf_dir):
            continue

        fix_img_path = os.path.join(cine_dir, "img_000.mha")
        if not os.path.exists(fix_img_path):
            continue

        mov_imgs = sorted(glob.glob(os.path.join(cine_dir, "img_*.mha")))

        for moving_img_path in mov_imgs:
            filename = os.path.basename(moving_img_path)
            frame_idx = filename.split("_")[1].split(".")[0]

            if frame_idx == "000":
                continue

            dvf_path = os.path.join(dvf_dir, f"dvfReverse{frame_idx}.mha")
            if not os.path.exists(dvf_path):
                continue

            (img_fixed_np, img_moving_np, dvf_np, seg_fixed_np, seg_moving_np, h, w, pad_meta,
                anatomy_mask_raw, norm_stats, spatial_meta) = _read_and_process_frame(
                root_dir, subdir, fix_img_path, moving_img_path, dvf_path, frame_idx, target_size, epsilon_bg
            )

            fixed_list.append(img_fixed_np)
            moving_list.append(img_moving_np)
            dvf_list.append(dvf_np)

            metadata_list.append({
                "seq_id": subdir,
                "frame_idx": frame_idx,
                "original_shape": (h, w),
                "padding_info": pad_meta,
                "anatomy_mask": anatomy_mask_raw,
                "seg_fixed": seg_fixed_np,
                "seg_moving": seg_moving_np,
                "norm_stats": norm_stats,
                "spatial_meta": spatial_meta,
            })

    return fixed_list, moving_list, dvf_list, metadata_list


def _read_and_process_frame(root_dir, subdir, fix_img_path, moving_img_path, dvf_path,
                             frame_idx, target_size, epsilon_bg):
    """Reads and preprocesses a single pair (fixed, moving, dvf) + its segmentation"""

    sitk_fixed = sitk.ReadImage(fix_img_path)
    sitk_moving = sitk.ReadImage(moving_img_path)
    sitk_dvf = sitk.ReadImage(dvf_path)

    # Sitk metadata
    spacing = sitk_fixed.GetSpacing()
    origin = sitk_fixed.GetOrigin()
    direction = sitk_fixed.GetDirection()

    img_fixed_np = sitk.GetArrayFromImage(sitk_fixed).squeeze(0).astype(np.float32)
    img_moving_np = sitk.GetArrayFromImage(sitk_moving).squeeze(0).astype(np.float32)
    dvf_np = sitk.GetArrayFromImage(sitk_dvf).squeeze(0).astype(np.float32)

    # tumor segmentation
    seg_dir = os.path.join(root_dir, subdir, "SynthesizedSegmentations")
    seg_fixed_path = os.path.join(seg_dir, "seg_000.mha")
    seg_moving_path = os.path.join(seg_dir, f"seg_{frame_idx}.mha")

    if os.path.exists(seg_fixed_path) and os.path.exists(seg_moving_path):
        seg_fixed_np = sitk.GetArrayFromImage(sitk.ReadImage(seg_fixed_path)).squeeze(0).astype(np.uint8)
        seg_moving_np = sitk.GetArrayFromImage(sitk.ReadImage(seg_moving_path)).squeeze(0).astype(np.uint8)
    else:
        seg_fixed_np, seg_moving_np = None, None

    # anatomy mask: only excludes real background (~0), not actually dark tissues
    anatomy_mask_raw = (img_fixed_np > epsilon_bg).astype(np.uint8)
    anatomy_mask_raw = binary_closing(anatomy_mask_raw, structure=np.ones((5, 5))).astype(np.uint8)
    anatomy_mask_raw = binary_fill_holes(anatomy_mask_raw).astype(np.uint8)

    # anatomy mask: only excludes real background (~0), not actually dark tissues
    anatomy_mask_raw = (img_fixed_np > epsilon_bg).astype(np.uint8)
    anatomy_mask_raw = binary_closing(anatomy_mask_raw, structure=np.ones((5, 5))).astype(np.uint8)
    anatomy_mask_raw = binary_fill_holes(anatomy_mask_raw).astype(np.uint8)

    # normalization stats before applying it
    mean_fixed = float(np.average(img_fixed_np))
    std_fixed = float(np.std(img_fixed_np))

    mean_moving = float(np.average(img_moving_np))
    std_moving = float(np.std(img_moving_np))

    # z-score normalization
    if img_fixed_np.max() - img_fixed_np.min() > 0:
        img_fixed_np = (img_fixed_np - mean_fixed) / std_fixed
    if img_moving_np.max() - img_moving_np.min() > 0:
        img_moving_np = (img_moving_np - mean_moving) / std_moving

    h, w = img_fixed_np.shape
    pad_meta = {"top": 0, "bottom": 0, "left": 0, "right": 0, "padded": False}

    if h < target_size[0] or w < target_size[1]:
        pad_h = max(0, target_size[0] - h)
        pad_w = max(0, target_size[1] - w)
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        img_fixed_np = np.pad(img_fixed_np, ((top, bottom), (left, right)), mode="constant")
        img_moving_np = np.pad(img_moving_np, ((top, bottom), (left, right)), mode="constant")
        dvf_np = np.pad(dvf_np, ((top, bottom), (left, right), (0, 0)), mode="constant")
        anatomy_mask_raw = np.pad(anatomy_mask_raw, ((top, bottom), (left, right)), mode="constant")

        pad_meta = {"top": top, "bottom": bottom, "left": left, "right": right, "padded": True}
    
    norm_stats = {
        "mean_fixed": mean_fixed, "std_fixed": std_fixed,
        "mean_moving": mean_moving, "std_moving": std_moving
    }

    spatial_meta = {"spacing": spacing, "origin": origin, "direction": direction}

    return (img_fixed_np, img_moving_np, dvf_np, seg_fixed_np, seg_moving_np, h, w, pad_meta,
            anatomy_mask_raw, norm_stats, spatial_meta)


def extract_patches(image, patch_size=TARGET_SIZE):
    """
    Extracts 2D patches from a numpy array using a linspace grid with overlap
    to cover the entire image.

    It works the same way for grayscale images (H, W) as it does for DVF (H, W, 2),
    anatomatically detecting the dimensionality.

    Returns
    -------
    patches : list[np.ndarray]
    coordinates : list[tuple[int, int]]  # (y, x) of the top-left corner
    """
    h, w = image.shape[0], image.shape[1]
    ph, pw = patch_size

    num_patches_h = int(np.ceil(h / ph)) if h > ph else 1
    num_patches_w = int(np.ceil(w / pw)) if w > pw else 1

    starts_y = np.linspace(0, h - ph, num_patches_h, dtype=int) if h > ph else [0]
    starts_x = np.linspace(0, w - pw, num_patches_w, dtype=int) if w > pw else [0]

    patches, coordinates = [], []
    for y in starts_y:
        for x in starts_x:
            if len(image.shape) == 3:
                patch = image[y:y + ph, x:x + pw, :]
            else:
                patch = image[y:y + ph, x:x + pw]
            patches.append(patch)
            coordinates.append((y, x))

    return patches, coordinates
