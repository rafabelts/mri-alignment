"""
Función de pérdida supervisada: Charbonnier-EPE (robusta) contra el DVF
real + regularización de suavidad 2D.

Nota histórica (ver reporte semanal): originalmente incluía un término de
similitud de imagen (LNCC) que se retiró tras causar divergencia de
entrenamiento en regiones de fondo con varianza casi cero, además de
contribuir de forma marginal al gradiente por un desajuste de escala con
el término EPE.
"""

import torch

from config import LAMBDA_DVF, LAMBDA_SMOOTH, CHARBONNIER_EPS


class Loss:
    def __init__(self, pred_dvf, gt_dvf, mask=None,
                 lambda_dvf=LAMBDA_DVF, lambda_smooth=LAMBDA_SMOOTH, eps=CHARBONNIER_EPS):
        self.pred_dvf = pred_dvf
        self.gt_dvf = gt_dvf
        self.mask = mask
        self.lambda_dvf = lambda_dvf
        self.lambda_smooth = lambda_smooth
        self.eps = eps

    def charbonnier_epe_loss(self):
        """Charbonnier: L2 para errores pequeños, L1 para errores grandes."""
        diff_sq = ((self.pred_dvf - self.gt_dvf) ** 2).sum(dim=1)
        charbonnier = torch.sqrt(diff_sq + self.eps ** 2)

        if self.mask is not None:
            charbonnier = charbonnier * self.mask
            return charbonnier.sum() / (self.mask.sum() + 1e-8)
        return charbonnier.mean()

    def smoothness_loss(self, penalty="l2"):
        """Regularización de difusión 2D: penaliza gradientes espaciales abruptos."""
        dy = torch.abs(self.pred_dvf[:, :, 1:, :] - self.pred_dvf[:, :, :-1, :])
        dx = torch.abs(self.pred_dvf[:, :, :, 1:] - self.pred_dvf[:, :, :, :-1])

        if penalty == "l2":
            dy = dy * dy
            dx = dx * dx

        return (torch.mean(dx) + torch.mean(dy)) / 2.0

    def total_loss(self):
        l_epe = self.charbonnier_epe_loss()
        l_smooth = self.smoothness_loss()
        l_total = self.lambda_dvf * l_epe + self.lambda_smooth * l_smooth
        return l_total, {"epe": l_epe.item(), "smooth": l_smooth.item()}
