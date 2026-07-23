"""
Definition/Construction for register models. 
"""

import src.compat
import voxelmorph as vxm

from config import VXM_INSHAPE, VXM_INT_STEPS, VXM_INT_DOWNSIZE, VXM_SRC_FEATS, VXM_TRG_FEATS

def build_model(model_name, device, inshape=None):
    if model_name == 'voxelmorph':
        return _build_voxelmorph(device, inshape=inshape) if inshape else _build_voxelmorph(device)
    elif model_name == 'transmorph':
        return _build_transmorph(device, inshape=inshape) if inshape else _build_transmorph(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _build_voxelmorph(device, inshape=VXM_INSHAPE, int_steps=VXM_INT_STEPS,
                      int_downsize=VXM_INT_DOWNSIZE, src_feats=VXM_SRC_FEATS,
                      trg_feats=VXM_TRG_FEATS, nb_unet_features=None):
    """
    Builds a VxmDense instance with diffeomorphic integration.
    """
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=nb_unet_features,
        int_steps=int_steps,
        int_downsize=int_downsize,
        src_feats=src_feats,
        trg_feats=trg_feats,
    ).to(device)
    return model

def _build_transmorph(device, inshape=VXM_INSHAPE, int_steps=VXM_INT_STEPS, int_downsize=VXM_INT_DOWNSIZE):
    from src.transmorph import TransMorphDiff
    model = TransMorphDiff(inshape=inshape, int_steps=int_steps, int_downsize=int_downsize).to(device)
    return model

def load_weights_any_shape(model, state_dict_path, device):
    """
    Load the trainned weights into a VxmDense instace with a different 
    'inshape' than the one used during training. The only size-dependent parameters
    are the SpatialTransformer's grid buffers (which are not trainable weights), so
    they are excluded before loading.

    Useful for running direct inference on full-size images of variable dimensions without
    going through patches.
    """
    import torch

    state_dict = torch.load(state_dict_path, map_location=device)
    filtered = {k: v for k, v in state_dict.items() if "grid" not in k}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model
