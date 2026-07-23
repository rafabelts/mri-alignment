import numpy as np
import SimpleITK as sitk


def denormalize(array_normalized, mean, std):
    """Reverts z-score normalization: x = x_norm * std + mean."""
    array_normalized * std + mean

def save_as_mha(array, spatial_meta, output_path):
    """
    Saves 2D numpy array as .mha, preserving spacing/origin/direction from the correspondant original
    file.

    Parameters
    ----------
    array : np.ndarray (H, W) — in original intesity units
    spatial_meta : dict with 'spacing', 'origin', 'direction'
    output_path: str or Path
    """

    array = array.astype(np.float32)
    sitk_array = array[np.newaxis, ...]

    img = sitk.GetImageFromArray(sitk_array, isVector=(array.ndim == 3))
    img.SetSpacing(spatial_meta["spacing"])
    img.SetOrigin(spatial_meta["origin"])
    img.SetDirection(spatial_meta["direction"])
    sitk.WriteImage(img, str(output_path))

def run_inference_and_export(model, img_fixed_np, img_moving_np, norm_stats, spatial_meta,
                             output_path, device="cuda"):
    """
    Runs full inference over a processed image and exports fixed, warped and predicted dvf as .mha
    
    Parameters:
    -----------
    output_dir: Path - file where the files are stored
    prefix: str - identifies the case (e.g. "C_002")

    Returns:
    --------
    fixed_clinical, warped_clinical : np.ndarray
    pred_dvf_np : np.ndarray (H, W, 2)
    """

    model.eval()
    fixed_t = torch.from_numpy(img_fixed_np).unsqueeze(0).unsqueeze(0).to(device).float()
    moving_t = torch.from_numpy(img_moving_np).unsqueeze(0).unsqueeze(0).to(device).float()

    with torch.no_grad():
        moved, pred_dvf = model(fixed_t, moving_t, registration=True)

    pred_dvf_np = pred_dvf.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
   
    h, w = img_fixed_np.shape
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    new_y = grid_y + pred_dvf_np[..., 1]
    new_x = grid_x + pred_dvf_np[..., 0]
    warped_normalized = map_coordinates(img_fixed_np, [new_y, new_x], order=1, mode="constant")

    fixed_clinical = denormalize(img_fixed_np, norm_stats["mean_fixed"], norm_stats["std_fixed"])
    warped_clinical = denormalize(warped_normalized, norm_stats["mean_fixed"], norm_stats["std_fixed"])

    save_as_mha(fixed_clinical, spatial_meta, output_dir / f"fixed_{prefix}.mha")
    save_as_mha(warped_clinical, spatial_meta, output_dir / f"warped_{prefix}.mha")
    save_as_mha(pred_dvf_np, spatial_meta, output_dir / f"dvf_{prefix}.mha")

    return fixed_clinical, warped_clinical, pred_dvf_np
