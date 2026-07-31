"""
Patients split in train/val/test, and the PyTorch Dataset that wraps the preprocessed
images, spliting them in 256 x 256 patches when original image is bigger.
"""

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import TEST_SIZE, VAL_TEST_SPLIT, RANDOM_STATE, TARGET_SIZE
from src.preprocessing import extract_patches


def split_patients(data_dir, test_size=TEST_SIZE, val_test_split=VAL_TEST_SPLIT,
                    random_state=RANDOM_STATE, verbose=True):
    """
    Divide the patients into train/val/test sets, stratified by cohort (based on the
    first letter of the folder name, e.g., 'A_001' -> 'A'), to prevent data leakage between
    splits (all pairs from the same patient stays in the same split)
    """
    data_dir = str(data_dir)
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    groups = [s.split("_")[0] for s in subdirs]

    train_dirs, temp_dirs, train_groups, temp_groups = train_test_split(
        subdirs, groups, test_size=test_size, random_state=random_state, stratify=groups
    )
    val_dirs, test_dirs = train_test_split(
        temp_dirs, test_size=val_test_split, random_state=random_state, stratify=temp_groups
    )

    if verbose:
        print(f"Train: {len(train_dirs)} | Val: {len(val_dirs)} | Test: {len(test_dirs)}")
        print("Train:", Counter(s.split("_")[0] for s in train_dirs))
        print("Val:  ", Counter(s.split("_")[0] for s in val_dirs))
        print("Test: ", Counter(s.split("_")[0] for s in test_dirs))

    return train_dirs, val_dirs, test_dirs


def cv_splits(data_dir, outer_k, inner_k, random_state=RANDOM_STATE):
    """
    Builds a nested cross-validation split at the patient level, stratified by
    cohort (first letter of the folder name), with no patient leakage (each
    patient folder is an atomic unit).

    Returns
    -------
    list[dict], one entry per outer fold, each with:
        "outer_train": list[str], "outer_test": list[str]
        "inner_folds": list[tuple[list[str], list[str]]] of length `inner_k`,
            each a (inner_train, inner_val) pair drawn from "outer_train"
    """
    data_dir = str(data_dir)
    subdirs = np.array(sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]))
    groups = np.array([s.split("_")[0] for s in subdirs])

    outer_skf = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    folds = []

    for outer_train_idx, outer_test_idx in outer_skf.split(subdirs, groups):
        outer_train = subdirs[outer_train_idx].tolist()
        outer_test = subdirs[outer_test_idx].tolist()
        outer_train_groups = groups[outer_train_idx]

        inner_skf = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        inner_folds = []
        for inner_train_idx, inner_val_idx in inner_skf.split(outer_train, outer_train_groups):
            inner_train = [outer_train[i] for i in inner_train_idx]
            inner_val = [outer_train[i] for i in inner_val_idx]
            inner_folds.append((inner_train, inner_val))

        folds.append({"outer_train": outer_train, "outer_test": outer_test, "inner_folds": inner_folds})

    return folds


class MRICineDataset(Dataset):
    """
    Wraps the preprocessed array lists (see src/preprocessing.py) and converts them into
    individual 256 x 256 samples, splitting the originally larger images into overlapping patches.

    Note: 'seg_fixed' / 'seg_moving' are excluded deliberately from per-patch metadata because
    its full size varies from patient to patient and breaks the DataLoader collate when building
    a batch; they still available in the original metadata (ram_meta_*) for evaluation of Dice/TRE
    in the full image
    """

    def __init__(self, fixed_arrays, moving_arrays, dvf_arrays, meta_list, patch_size=TARGET_SIZE):
        self.patch_size = patch_size
        self.samples = []

        for f_img, m_img, dvf, meta in zip(fixed_arrays, moving_arrays, dvf_arrays, meta_list):
            if meta["padding_info"]["padded"]:
                patch_meta = meta.copy()
                patch_meta.pop("seg_fixed", None)
                patch_meta.pop("seg_moving", None)
                patch_meta["patch_coords"] = (0, 0)
                patch_meta["is_patchified"] = False
                self.samples.append((f_img, m_img, dvf, patch_meta))
            else:
                patches_f, coordinates = extract_patches(f_img, patch_size)
                patches_m, _ = extract_patches(m_img, patch_size)
                patches_dvf, _ = extract_patches(dvf, patch_size)
                patches_mask, _ = extract_patches(meta["anatomy_mask"], patch_size)

                for pf, pm, pdvf, pmask, (y, x) in zip(patches_f, patches_m, patches_dvf,
                                                        patches_mask, coordinates):
                    patch_meta = meta.copy()
                    patch_meta.pop("seg_fixed", None)
                    patch_meta.pop("seg_moving", None)
                    patch_meta["patch_coords"] = (y, x)
                    patch_meta["is_patchified"] = True
                    patch_meta["anatomy_mask"] = pmask
                    self.samples.append((pf, pm, pdvf, patch_meta))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (fixed, moving, dvf, meta) as (1, H, W) / (2, H, W) tensors, channels-first."""
        img_fixed_np, img_moving_np, dvf_np, meta = self.samples[idx]

        img_fixed_pt = torch.from_numpy(img_fixed_np).unsqueeze(0)
        img_moving_pt = torch.from_numpy(img_moving_np).unsqueeze(0)
        dvf_pt = torch.from_numpy(dvf_np).permute(2, 0, 1)

        return img_fixed_pt, img_moving_pt, dvf_pt, meta


def build_lookup(ram_meta):
    """Maps (seq_id, frame_idx) -> original ram_meta list index"""
    return {(m["seq_id"], m["frame_idx"]): idx for idx, m in enumerate(ram_meta)}
