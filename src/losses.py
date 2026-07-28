"""
Supervised Loss Function: Charbonnier-EPE against the real DVF + 2D smooth regularization.
"""

import torch

from config import LAMBDA_DVF, LAMBDA_SMOOTH, CHARBONNIER_EPS


class Loss:
    """
    Supervised registration loss: Charbonnier EPE against the ground-truth DVF
    (masked to the anatomy region if `mask` is given) plus a 2D smoothness
    regularizer on the predicted DVF.
    """

    def __init__(self, pred_dvf, gt_dvf, mask=None,
                 lambda_dvf=LAMBDA_DVF, lambda_smooth=LAMBDA_SMOOTH, eps=CHARBONNIER_EPS):
        self.pred_dvf = pred_dvf
        self.gt_dvf = gt_dvf
        self.mask = mask
        self.lambda_dvf = lambda_dvf
        self.lambda_smooth = lambda_smooth
        self.eps = eps

    def charbonnier_epe_loss(self):
        """Charbonnier: L2 for small mistakes, L1 big errors."""
        diff_sq = ((self.pred_dvf - self.gt_dvf) ** 2).sum(dim=1)
        charbonnier = torch.sqrt(diff_sq + self.eps ** 2)

        if self.mask is not None:
            charbonnier = charbonnier * self.mask
            return charbonnier.sum() / (self.mask.sum() + 1e-8)
        return charbonnier.mean()

    def smoothness_loss(self, penalty="l2"):
        """2D smoothness regularization: penalizes abrupt spatial gradients."""
        dy = torch.abs(self.pred_dvf[:, :, 1:, :] - self.pred_dvf[:, :, :-1, :])
        dx = torch.abs(self.pred_dvf[:, :, :, 1:] - self.pred_dvf[:, :, :, :-1])

        if penalty == "l2":
            dy = dy * dy
            dx = dx * dx

        return (torch.mean(dx) + torch.mean(dy)) / 2.0

    def total_loss(self):
        """Weighted sum of the EPE and smoothness terms.

        Returns
        -------
        l_total : torch.Tensor
            Scalar loss to call `.backward()` on.
        parts : dict
            {'epe': float, 'smooth': float} unweighted component values, for logging.
        """
        l_epe = self.charbonnier_epe_loss()
        l_smooth = self.smoothness_loss()
        l_total = self.lambda_dvf * l_epe + self.lambda_smooth * l_smooth
        return l_total, {"epe": l_epe.item(), "smooth": l_smooth.item()}
