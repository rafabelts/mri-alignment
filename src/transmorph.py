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

class TransMorphDiff(nn.Module):
    """
        Predicts mean and log-variance of velocity field. During training, it samples via reparameterization
        (which allows backpropagation trough the samples); in inference it uses the mean (deterministic).
    """
    def __init__(self, inshape=(256, 256), int_steps=7, int_downsize=2,
                 embed_dim=96, depth=4, num_heads=4):
        super().__init__()
        ndims = len(inshape)

        # -- Encoder --
        self.down1 = ConvBlock(2, 16, stride=2) # 256 -> 128
        self.down2 = ConvBlock(16, 32, stride=2) # 128 -> 64
        self.down3 = ConvBlock(32, 64, stride=2) # 64 -> 32
        self.down4 = ConvBlock(64, embed_dim, stride=2) # 32 -> 16

        n_tokens = (inshape[0] // 16) * (inshape[1] // 16)
        self.bottleneck = TransformerBottleneck(embed_dim, n_tokens, depth, num_heads)

        # -- Decoder (U-Net style, with skip connections) --
        self.up1 = ConvBlock(embed_dim + 64, 64)
        self.up2 = ConvBlock(64 + 32, 32)
        self.up3 = ConvBlock(32 + 16, 16)
        self.up4 = ConvBlock(16 + 2, 16)
        
        # -- Probabilistic head: mean and variance-log --
        self.flow_mean_head = nn.Conv2d(16, ndims, kernel_size=3, padding=1)
        self.flow_logvar_head = nn.Conv2d(16, ndims, kernel_size=3, padding=1)

        # starts almost in 0: when beginning, predicts deformation ~nil
        # and small variance (avoids chaotic in the early epochs)
        nn.init.constant_(self.flow_mean_head.weight, 1e-5)
        nn.init.constant_(self.flow_mean_head.bias, 0.0)
        nn.init.constant_(self.flow_logvar_head.weight, 1e-5)
        nn.init.constant_(self.flow_logvar_head.bias, -5.0) # low initial logvar -> small std

        # -- Difeomorphic integration and wrapping, recicled from voxelmorph --
        down_shape = [int(d / int_downsize) for d in inshape]
        self.resize = vxm_layers.ResizeTransform(int_downsize, ndims) if int_downsize > 1 else None
        self.fullsize = vxm_layers.ResizeTransform(1 / int_downsize, ndims) if int_downsize > 1 else None
        self.integrate = vxm_layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.transformer = vxm_layers.SpatialTransformer(inshape)

        # filled in on every `forward()` call; available to anyone who wants to add it to the loss function
        self.last_kl_loss = None
    
    def forward(self, source, target, registration=False):
        x = torch.cat([source, target], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        u1 = nn.functional.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, d3], dim=1))

        u2 = nn.functional.interpolate(u1, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, d2], dim=1))

        u3 = nn.functional.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, d1], dim=1))

        u4 = nn.functional.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = self.up4(torch.cat([u4, x], dim=1))

        flow_mean = self.flow_mean_head(u4)
        flow_logvar = self.flow_logvar_head(u4)

        self.last_kl_loss = -0.5 * torch.mean(1 + flow_logvar - flow_mean.pow(2) - flow_logvar.exp())

        if self.training:
            std = torch.exp(0.5 * flow_logvar)
            eps = torch.randn_like(std)
            preint_flow = flow_mean + eps * std
        else:
            preint_flow = flow_mean

        pos_flow = preint_flow
        if self.resize:
            pos_flow = self.resize(pos_flow)
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        y_source = self.transformer(source, pos_flow)

        if not registration:
            return y_source, preint_flow
        return y_source, pos_flow
