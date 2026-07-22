"""
A probabilistc variant inspired by Transmorph-diff (Chen et al., 2021)
and the probabilistc framework of Dalca et al. (2018) which that paper
cites as its basis. Transformer Encoder + CNN decoder, with probabilistc
head (mean + log-variance of the velocity field).


Note: this is a 2D adaptation, not a port of the original code by Chen et al.
(which is 3D, uses a full Swin Transformer encoder, and its own spatial transformer
in normalized coordinates [-1, 1]). Here, we reuse the diffeomorphic integration 
(VecInt) and warping (SpatialTransformer) from voxelmorph-which have already
been validated in this project—to maintain a single, consistent 
coordinate convention throughout the entire pipeline.

Mantains the same VoxelMorph interface:
    model(source, target, registration=True) -> (moved, pos_flow)
in that way train and eval function doesnt need to be modified.

The term KL (variance regularization, simplified version against an N(0, I) prior) 
is displayed as the 'model.last_kl_loss' attribute after each forward pass.
"""

import src.compat
import torch
import torch.nn as nn
from voxelmorph.torch import layers as vxm_layers

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))

class TransformerBottleneck(nn.Module):
    """ Standard self-attention on the most compressed feature map  """

    def __init__(self, embed_dim, num_tokens, depth=4, num_heads=4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model = embed_dim, nhead = num_heads,
                dim_feedforward=embed_dim * 4, batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)





