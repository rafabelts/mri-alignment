"""
Definición/construcción de modelos de registro. Por ahora solo VoxelMorph;
TransMorph se agregará aquí siguiendo la misma interfaz (build_transmorph)
para que train.py no tenga que cambiar entre arquitecturas.
"""

import src.compat  # noqa: F401  (debe importarse antes que voxelmorph)
import voxelmorph as vxm

from config import VXM_INSHAPE, VXM_INT_STEPS, VXM_INT_DOWNSIZE, VXM_SRC_FEATS, VXM_TRG_FEATS


def build_voxelmorph(device, inshape=VXM_INSHAPE, int_steps=VXM_INT_STEPS,
                      int_downsize=VXM_INT_DOWNSIZE, src_feats=VXM_SRC_FEATS,
                      trg_feats=VXM_TRG_FEATS, nb_unet_features=None):
    """
    Construye una instancia de VxmDense con integración diffeomórfica.

    Nota sobre `inshape`: es fijo al tamaño de entrada usado en entrenamiento
    (normalmente 256x256, los patches). Para inferir sobre una imagen de
    tamaño distinto sin repatchificar, se debe construir una nueva instancia
    con el `inshape` correspondiente y cargar los pesos entrenados excluyendo
    los buffers de rejilla (ver load_weights_any_shape en este mismo módulo).
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


def load_weights_any_shape(model, state_dict_path, device):
    """
    Carga pesos entrenados en una instancia de VxmDense con OTRO `inshape`
    al que se usó en entrenamiento. Los únicos parámetros dependientes del
    tamaño son los buffers de la rejilla del SpatialTransformer (no son
    pesos entrenables), así que se excluyen antes de cargar.

    Útil para correr inferencia directa sobre imágenes completas de tamaño
    variable (270x270, 425x425, etc.) sin pasar por patches.
    """
    import torch

    state_dict = torch.load(state_dict_path, map_location=device)
    filtered = {k: v for k, v in state_dict.items() if "grid" not in k}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model
