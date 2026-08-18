"""
2D probabilistic diffeomorphic registration model: CNN encoder + Transformer
self-attention bottleneck + CNN decoder, with a probabilistic head that
predicts the mean and log-variance of a stationary velocity field instead of
a single deterministic deformation. During training the field is sampled via
reparameterization (so gradients flow through the sample); at inference the
mean is used directly, so the model is deterministic in production.

The Transformer-in-a-registration-network idea follows TransMorph
(Chen et al., 2021, "TransMorph: Transformer for unsupervised medical image
registration"); the probabilistic mean/log-variance head, the KL term, and
the diffeomorphic velocity-field formulation follow the probabilistic
diffeomorphic framework of Dalca et al. (2018, "Unsupervised Learning of
Probabilistic Diffeomorphic Registration for Images and Surfaces"), which
TransMorph's diffeomorphic variant itself builds on.

This is a 2D, from-scratch design, not a port of any specific 3D
implementation - it does not use TransMorph's hierarchical/windowed (Swin)
attention encoder, and its spatial transformer works in pixel coordinates.
It reuses two building blocks from `voxelmorph` (Balakrishnan et al., 2019):
`VecInt` (scaling-and-squaring integration, in the sense of Ashburner 2007 /
Dalca et al. 2018) and `SpatialTransformer` (warping), instead of writing
them from scratch, so that this model resamples images and integrates flow
fields with the exact same numerics and pixel-coordinate convention as the
other (VoxelMorph) model in this project.

Exposes the same interface used elsewhere for the other model:
    model(source, target, registration=True) -> (moved, pos_flow)
so the training/eval code doesn't need a separate code path per model.

Each forward pass also sets `self.last_kl_loss`: a regularization term on
the predicted mean/log-variance of the velocity field, pulling it toward
low-variance, spatially smooth predictions, for the caller to add to the
training loss.

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

import math

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

    The idea of putting a Transformer inside a registration network follows
    TransMorph (Chen et al., 2021); unlike TransMorph's Swin encoder, this
    is plain dense self-attention applied only at the bottleneck.
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


class TransMorphDiff(nn.Module):
    """
    Predicts mean and log-variance of velocity field. During training, it samples via reparameterization
    (which allows backpropagation trough the samples); in inference it uses the mean (deterministic).
    """

    def __init__(
        self,
        inshape=(256, 256),
        int_steps=7,
        int_downsize=2,
        embed_dim=96,
        depth=4,
        num_heads=4,
        prior_lambda=25.0,
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

        # -- Probabilistic head: mean and variance-log (Dalca et al., 2018) --
        # predicts q(velocity | fixed, moving) = N(mean, diag(exp(logvar)))
        self.flow_mean_head = nn.Conv2d(16, ndims, kernel_size=3, padding=1)
        self.flow_logvar_head = nn.Conv2d(16, ndims, kernel_size=3, padding=1)

        # starts almost in 0: when beginning, predicts deformation ~nil
        # and small variance (avoids chaotic in the early epochs). Same
        # near-identity init trick as Dalca et al. (2018) / voxelmorph.
        nn.init.normal_(self.flow_mean_head.weight, mean=0.0, std=1e-5)
        nn.init.constant_(self.flow_mean_head.bias, 0.0)
        nn.init.normal_(self.flow_logvar_head.weight, mean=0.0, std=1e-10)
        nn.init.constant_(self.flow_logvar_head.bias, -10.0)

        # -- Turns the predicted velocity field into an invertible displacement --
        # (all three layers reused from voxelmorph, Balakrishnan et al. 2019)
        # `resize`/`fullsize` down/upsample the field around the integration step
        # (cheaper and more regular at lower resolution); `integrate` runs
        # scaling-and-squaring (Ashburner 2007 / Dalca et al. 2018) to turn the
        # (stationary) velocity field into an actual, guaranteed-invertible
        # displacement field; `transformer` resamples `source` with that
        # displacement field via bilinear grid sampling.
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

        self.prior_lambda = prior_lambda
        # filled in on every `forward()` call; available to anyone who wants to add it to the loss function
        self.last_kl_loss = None

    def _diffusion_penalty(self, field):
        """Mean squared spatial gradient of `field` (used as the KL precision term)."""
        dy = (field[:, :, 1:, :] - field[:, :, :-1, :]) ** 2
        dx = (field[:, :, :, 1:] - field[:, :, :, :-1]) ** 2

        return (dy.mean() + dx.mean()) / 2.0

    def forward(self, source, target, registration=False):
        """
        Predicts the DVF that warps `source` onto `target` and applies it.

        Also sets `self.last_kl_loss` as a side effect (mean + smoothness
        regularization on the predicted velocity field, weighted by
        `prior_lambda`), for the caller to add to the training loss.

        Parameters
        ----------
        source, target : torch.Tensor
            (B, 1, H, W) images.
        registration : bool
            If False (training), returns the pre-integration flow
            (`preint_flow`), sampled from the mean/log-variance heads
            *after* they've been resized to the integration resolution
            (`inshape / int_downsize`). If True (inference/eval), returns
            the final integrated flow (`pos_flow`) at full resolution, i.e.
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

        flow_mean = self.flow_mean_head(u4)
        flow_logvar = self.flow_logvar_head(u4)

        # KL against the smooth velocity-field prior (Dalca et al., 2018, eq. for
        # KL(q||p) with a Markov-graph precision; here approximated on the 2D pixel
        # grid). sigma_term penalizes large predicted variances (scaled by prior_lambda
        # and the 2D 4-neighbour degree) and rewards informative ones (the -flow_logvar
        # term keeps it from collapsing the variance to zero for free); prec_term
        # penalizes a non-smooth mean field, also scaled by prior_lambda so both terms
        # carry the same prior strength. Together they pull the distribution toward
        # "small, confident, spatially smooth deformation".
        degree = 4.0  # neighbour count in a 2D 4-connected pixel grid
        sigma_term = torch.mean(
            self.prior_lambda * degree * torch.exp(flow_logvar) - flow_logvar
        )
        prec_term = self.prior_lambda * self._diffusion_penalty(flow_mean)
        self.last_kl_loss = 0.5 * (sigma_term + prec_term)

        # Move mean/logvar to the integration grid (inshape / int_downsize) before
        # sampling, so the noise is drawn where VecInt integrates. flow_mean is a pixel
        # displacement field, so it goes through ResizeTransform (interpolate + rescale
        # magnitude by factor). flow_logvar must NOT: ResizeTransform would multiply it
        # by factor, but a log-variance rescales *additively* - since std' = factor*std,
        # logvar' = logvar + 2*ln(factor) (= -1.386 nats for factor=0.5). Passing it
        # through ResizeTransform instead would give factor*logvar, injecting the wrong
        # amount of noise (~24x too much std at the -10 init bias). So interpolate it
        # and add the 2*ln(factor) shift by hand - do not "simplify" this back to
        # self.resize(flow_logvar).
        mean_for_int = flow_mean
        logvar_for_int = flow_logvar
        if self.resize:
            factor = 1.0 / self.int_downsize
            mean_for_int = self.resize(flow_mean)
            logvar_for_int = nn.functional.interpolate(
                flow_logvar, scale_factor=factor, mode="bilinear", align_corners=True
            ) + 2.0 * math.log(factor)

        if self.training:
            std = torch.exp(0.5 * logvar_for_int)
            eps = torch.randn_like(std)
            preint_flow = mean_for_int + eps * std
        else:
            preint_flow = mean_for_int

        pos_flow = preint_flow
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        y_source = self.transformer(source, pos_flow)

        if not registration:
            return y_source, preint_flow
        return y_source, pos_flow
