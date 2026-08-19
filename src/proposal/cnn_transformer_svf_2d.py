"""Lightweight CNN--Transformer model for supervised 2D DVF prediction.

The network uses a convolutional encoder, global self-attention at the
bottleneck, and a convolutional decoder with skip connections. A deterministic
head predicts a two-channel stationary velocity field (SVF), which is integrated
into a displacement vector field (DVF) before warping the source image.

The use of Transformer attention for image registration is inspired by TransMorph
(Chen et al., 2021, "TransMorph: Transformer for unsupervised medical image
registration"), but this is not a 2D port of TransMorph: it does not implement
its hierarchical Swin Transformer. The stationary-velocity formulation follows Dalca
et al. (2018, "Unsupervised Learning of Probabilistic Diffeomorphic
Registration for Images and Surfaces"), which TransMorph's diffeomorphic
variant itself builds on.

It reuses three building blocks from `voxelmorph` (Balakrishnan et al., 2019):
`ResizeTransform` (resolution changes with displacement rescaling),
`VecInt` (scaling-and-squaring integration, in the sense of Ashburner 2007 /
Dalca et al. 2018) and `SpatialTransformer` (warping), instead of writing
them from scratch, so that this model resamples images and integrates flow
fields with the exact same numerics and pixel-coordinate convention as the
other (VoxelMorph) model in this project.

Exposes the same interface used elsewhere for the other model:
    model(source, target, registration=True) -> (moved, pos_flow)
so the training/eval code doesn't need a separate code path per model.

References
---------
- Chen et al. (2021). TransMorph: Transformer for unsupervised medical
  image registration. Medical Image Analysis.
- Dalca et al. (2018). Unsupervised Learning of Probabilistic Diffeomorphic
  Registration for Images and Surfaces. MICCAI.
- Balakrishnan et al. (2019). VoxelMorph: A Learning Framework for
  Deformable Medical Image Registration. IEEE TMI.
- Ashburner (2007). A fast diffeomorphic image registration algorithm.
  NeuroImage. (scaling-and-squaring integration of a stationary velocity field)
"""

import torch
from torch import nn
from voxelmorph.torch import layers as vxm_layers


class ConvBlock(nn.Module):
    """3x3 conv + LeakyReLU(0.2). `stride=2` halves spatial resolution (used in the encoder)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))


class TransformerBottleneck(nn.Module):
    """Standard self-attention on the most compressed feature map.

    This is dense global self-attention at one resolution. It is inspired by
    Transformer-based registration, but is not TransMorph's hierarchical Swin
    Transformer encoder.
    """

    def __init__(self, embed_dim, num_tokens, depth=4, num_heads=4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class CNNTransformerSVF2D(nn.Module):
    """Predicts and integrates a deterministic 2D stationary velocity field."""

    def __init__(
        self,
        inshape=(256, 256),
        int_steps=7,
        int_downsize=2,
        embed_dim=96,
        depth=4,
        num_heads=4,
    ):
        super().__init__()
        ndims = len(inshape)

        # -- Encoder --
        self.down1 = ConvBlock(2, 16, stride=2)  # 256 -> 128
        self.down2 = ConvBlock(16, 32, stride=2)  # 128 -> 64
        self.down3 = ConvBlock(32, 64, stride=2)  # 64 -> 32
        self.down4 = ConvBlock(64, embed_dim, stride=2)  # 32 -> 16

        n_tokens = (inshape[0] // 16) * (inshape[1] // 16)
        self.bottleneck = TransformerBottleneck(embed_dim, n_tokens, depth, num_heads)

        # -- Decoder (U-Net style, with skip connections) --
        self.up1 = ConvBlock(embed_dim + 64, 64)
        self.up2 = ConvBlock(64 + 32, 32)
        self.up3 = ConvBlock(32 + 16, 16)
        self.up4 = ConvBlock(16 + 2, 16)

        # -- Deterministic stationary velocity field head --
        # Near-zero initialization starts training close to the identity warp.
        self.velocity_head = nn.Conv2d(16, ndims, kernel_size=3, padding=1)
        nn.init.normal_(self.velocity_head.weight, mean=0.0, std=1e-5)
        nn.init.constant_(self.velocity_head.bias, 0.0)

        # -- Convert the predicted SVF into a displacement field --
        # (all three layers reused from voxelmorph, Balakrishnan et al. 2019)
        # `resize`/`fullsize` down/upsample the field around the integration step
        # (cheaper and more regular at lower resolution); `integrate` runs
        # scaling-and-squaring (Ashburner 2007 / Dalca et al. 2018) to turn the
        # stationary velocity field into a discrete approximation of its
        # diffeomorphic flow. Numerical discretization can still introduce local
        # foldings, which are assessed with the Jacobian metric. `transformer`
        # resamples `source` with the resulting DVF via bilinear grid sampling.
        self.int_downsize = int_downsize
        down_shape = [int(d / int_downsize) for d in inshape]
        self.resize = (
            vxm_layers.ResizeTransform(int_downsize, ndims)
            if int_downsize > 1
            else None
        )
        self.fullsize = (
            vxm_layers.ResizeTransform(1 / int_downsize, ndims)
            if int_downsize > 1
            else None
        )
        self.integrate = (
            vxm_layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        )
        self.transformer = vxm_layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        """
        Predicts the DVF that warps `source` onto `target` and applies it.

        Parameters
        ----------
        source, target : torch.Tensor
            (B, 1, H, W) images.
        registration : bool
            If False, returns the deterministic pre-integration velocity
            (`preint_flow`) at the integration resolution
            (`inshape / int_downsize`). If True (training/eval), returns the
            final integrated flow (`pos_flow`) at full resolution, i.e.
            the actual displacement field used to produce `y_source`.

        Returns
        -------
        y_source : torch.Tensor
            `source` warped by the predicted (fully integrated) flow.
        flow : torch.Tensor
            `preint_flow` or `pos_flow` depending on `registration`.
        """
        x = torch.cat([source, target], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        u1 = nn.functional.interpolate(
            bottleneck, scale_factor=2, mode="bilinear", align_corners=False
        )
        u1 = self.up1(torch.cat([u1, d3], dim=1))

        u2 = nn.functional.interpolate(
            u1, scale_factor=2, mode="bilinear", align_corners=False
        )
        u2 = self.up2(torch.cat([u2, d2], dim=1))

        u3 = nn.functional.interpolate(
            u2, scale_factor=2, mode="bilinear", align_corners=False
        )
        u3 = self.up3(torch.cat([u3, d1], dim=1))

        u4 = nn.functional.interpolate(
            u3, scale_factor=2, mode="bilinear", align_corners=False
        )
        u4 = self.up4(torch.cat([u4, x], dim=1))

        velocity = self.velocity_head(u4)
        preint_flow = self.resize(velocity) if self.resize else velocity

        pos_flow = preint_flow
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        y_source = self.transformer(source, pos_flow)

        if not registration:
            return y_source, preint_flow
        return y_source, pos_flow
