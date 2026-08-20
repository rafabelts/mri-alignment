"""Verify that in-place reconstruction matches the original implementation."""

import numpy as np


def original_reconstruction(pred_sum, gt_sum, mask_sum, count):
    count_safe = np.maximum(count, 1e-6)
    return {
        "pred_dvf": pred_sum / count_safe[..., None],
        "gt_dvf": gt_sum / count_safe[..., None],
        "anatomy_mask": (mask_sum / count_safe) > 0.5,
    }


def inplace_reconstruction(pred_sum, gt_sum, mask_sum, count):
    count_safe = count
    np.maximum(count_safe, 1e-6, out=count_safe)
    np.divide(pred_sum, count_safe[..., None], out=pred_sum)
    np.divide(gt_sum, count_safe[..., None], out=gt_sum)
    np.divide(mask_sum, count_safe, out=mask_sum)
    return {
        "pred_dvf": pred_sum,
        "gt_dvf": gt_sum,
        "anatomy_mask": mask_sum > 0.5,
    }


def main():
    rng = np.random.default_rng(0)
    shape = (423, 423)

    pred_sum = rng.normal(size=(*shape, 2)).astype(np.float32)
    gt_sum = rng.normal(size=(*shape, 2)).astype(np.float32)
    mask_sum = rng.random(shape, dtype=np.float32)
    count = rng.random(shape, dtype=np.float32)

    # Exercise the division-by-zero protection explicitly.
    count[0, 0] = 0.0
    count[100, 200] = 0.0

    original = original_reconstruction(pred_sum, gt_sum, mask_sum, count)
    inplace = inplace_reconstruction(
        pred_sum.copy(), gt_sum.copy(), mask_sum.copy(), count.copy()
    )

    checks = {
        name: np.array_equal(original[name], inplace[name])
        for name in original
    }

    print(f"Input shape: {shape}; dtype: {pred_sum.dtype}")
    for name, passed in checks.items():
        print(f"{name}: {'EXACT MATCH' if passed else 'MISMATCH'}")

    if not all(checks.values()):
        raise SystemExit("FAILED: the implementations produced different results")

    print("PASS: in-place reconstruction is exactly equivalent")


if __name__ == "__main__":
    main()
