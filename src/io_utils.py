import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import map_coordinates
from src.models import build_model, load_weights_any_shape

def denormalize(array_normalized, mean, std):
    """Reverts z-score normalization: x = x_norm * std + mean."""
    array_normalized * std + mean

def padded_shape(h, w, multiple=16):
    # calculates the nearest larger size (h, w) that is a multiple of multiple param
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return h + pad_h, w + pad_w

def pad_to_multiple(array, multiple=16):
    # fills w zero until the closest multiple of 16
    h, w = array.shape[:2]
    ph, pw = padded_shape(h, w, multiple)
    pad_spec = ((0, ph - h), (0, pw - w)) if array.ndim == 2 else ((0, ph - h), (0, pw - w), (0, 0))
    return np.pad(array, pad_spec, mode="constant")

def crop_to_original(array, original_shape):
    # crop it back to its original size, discarding the added padding
    h, w = original_shape
    return array[:h, :w, ...] if array.ndim > 2 else array[:h, :w]

def infer_on_external_images(model_name, checkpoint_path, fixed_path, moving_path,
                              output_dir, prefix, device="cuda"):
    """
        Runs inference over 2 images, normalizes, fills to multiple of 16, infers, crops and exports
        fixed/warped/dvf as .mha
    """
    sitk_fixed = sitk.ReadImage(str(fixed_path))
    sitk_moving = sitk.ReadImage(str(moving_path))

    spatial_meta = {
        "spacing": sitk_fixed.GetSpacing(),
        "origin": sitk_fixed.GetOrigin(),
        "direction": sitk_fixed.GetDirection(),
    }

    img_fixed_raw = sitk.GetArrayFromImage(sitk_fixed).squeeze(0).astype(np.float32)
    img_moving_raw = sitk.GetArrayFromImage(sitk_moving).squeeze(0).astype(np.float32)
    original_shape = img_fixed_raw.shape

    mean_fixed, std_fixed = float(img_fixed_raw.mean()), float(img_fixed_raw.std())
    img_fixed_norm = (img_fixed_raw - mean_fixed) / std_fixed if std_fixed > 0 else img_fixed_raw
    img_moving_norm = (img_moving_raw - img_moving_raw.mean()) / img_moving_raw.std()

    fixed_padded = pad_to_multiple(img_fixed_norm, multiple=16)
    moving_padded = pad_to_multiple(img_moving_norm, multiple=16)

    ph, pw = fixed_padded.shape

    model = build_model(model_name, device, inshape=(ph, pw))
    model = load_weights_any_shape(model, checkpoint_path, device)

    model.eval()

    fixed_t = torch.from_numpy(fixed_padded).unsqueeze(0).unsqueeze(0).to(device).float()
    moving_t = torch.from_numpy(moving_padded).unsqueeze(0).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        moved, pred_dvf = model(fixed_t, moving_t, registration=True)

    # computational cost
    print(f"--- Computational cost over image size ({ph} x {pw}) ---")
    benchmark_model(model, fixed_t, moving_t, device=device)

    pred_dvf_np = pred_dvf.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_dvf_np = crop_to_original(pred_dvf_np, original_shape)

    h, w = original_shape
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    new_y = grid_y + pred_dvf_np[..., 1]
    new_x = grid_x + pred_dvf_np[..., 0]
    warped_normalized = map_coordinates(img_fixed_norm, [new_y, new_x], order=1, mode="constant")

    fixed_clinical = img_fixed_raw
    warped_clinical = denormalize(warped_normalized, mean_fixed, std_fixed)

    save_as_mha(fixed_clinical, spatial_meta, output_dir / f"fixed_{prefix}.mha")
    save_as_mha(warped_clinical, spatial_meta, output_dir / f"warped_{prefix}.mha")
    save_as_mha(pred_dvf_np, spatial_meta, output_dir / f"dvf_{prefix}.mha")

    return fixed_clinical, warped_clinical, pred_dvf_np
