"""
Compatibility with the 'voxelmorph' library (v0.2), which was written for a version of Python earlier than
3.11 and uses 'inspect.getargspec', that was removed from Python 3.11+


IMPORTANT: This module must be imported BEFORE `voxelmorph`/`neurite` in any file that uses them, since
environment variables only take effect if they are defined before the import.
"""

import inspect
import os
from collections import namedtuple

# vxm/neurite supports a pythorch or tensorflow backend, here we use pytorch.
# Note: "VXM_BACKEND" and "NEURITE_BACKEND" are the exact names required
# the Voxelmorph library itself — they cannot be configured by this project,
# unlike the MRI_* variables used in config.py.

os.environ.setdefault("VXM_BACKEND", "pytorch")
os.environ.setdefault("NEURITE_BACKEND", "pytorch")

if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "varkw", "defaults"])

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec


def patch_voxelmorph_spatial_transformer():
    """Use an explicit meshgrid indexing mode with the legacy VoxelMorph layer.

    VoxelMorph 0.2 calls ``torch.meshgrid`` without ``indexing``.  PyTorch's
    current default is ``"ij"``, so specifying it preserves VoxelMorph's grid
    layout while avoiding the deprecation warning (and a future behaviour
    change when PyTorch makes the argument mandatory).
    """
    import torch
    import torch.nn.functional as nnf
    from torch import nn
    from voxelmorph.torch import layers

    if getattr(layers.SpatialTransformer, "_mri_alignment_compatible", False):
        return

    class SpatialTransformer(nn.Module):
        _mri_alignment_compatible = True

        def __init__(self, size, mode="bilinear"):
            super().__init__()
            self.mode = mode

            vectors = [torch.arange(0, dimension) for dimension in size]
            grid = torch.stack(torch.meshgrid(*vectors, indexing="ij"))
            grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)
            self.register_buffer("grid", grid)

        def forward(self, src, flow):
            new_locs = self.grid + flow
            shape = flow.shape[2:]

            for axis in range(len(shape)):
                new_locs[:, axis, ...] = 2 * (
                    new_locs[:, axis, ...] / (shape[axis] - 1) - 0.5
                )

            if len(shape) == 2:
                new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            return nnf.grid_sample(
                src, new_locs, align_corners=True, mode=self.mode
            )

    layers.SpatialTransformer = SpatialTransformer
