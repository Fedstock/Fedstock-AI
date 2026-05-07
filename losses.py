import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_scaler_center_and_scale(target_scaler):
    if target_scaler is None:
        return None, None

    center = getattr(target_scaler, "center_", None)
    if center is None:
        center = getattr(target_scaler, "mean_", None)
    scale = getattr(target_scaler, "scale_", None)
    if center is None or scale is None:
        return None, None

    return float(center[0]), max(float(scale[0]), 1e-8)


class HuberSMAPELoss(nn.Module):
    def __init__(self, target_scaler=None, huber_delta=1.0, smape_weight=0.1, eps=1e-8):
        super().__init__()
        self.huber_delta = huber_delta
        self.smape_weight = smape_weight
        self.eps = eps

        center, scale = _get_scaler_center_and_scale(target_scaler)
        if center is None:
            self.target_center = None
            self.target_scale = None
        else:
            self.register_buffer("target_center", torch.tensor(center, dtype=torch.float32))
            self.register_buffer("target_scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, pred, target):
        huber = F.huber_loss(pred, target, delta=self.huber_delta)
        if self.smape_weight <= 0:
            return huber

        if self.target_center is not None:
            pred_for_smape = pred * self.target_scale + self.target_center
            target_for_smape = target * self.target_scale + self.target_center
        else:
            pred_for_smape = pred
            target_for_smape = target

        smape = torch.mean(
            2.0
            * torch.abs(pred_for_smape - target_for_smape)
            / (torch.abs(target_for_smape) + torch.abs(pred_for_smape) + self.eps)
        )
        return huber + self.smape_weight * smape
