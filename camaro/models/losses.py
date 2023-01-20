import torch
import torch.nn as nn

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_mask: torch.Tensor, mixup_coeffs=None):
        pad_mask = pad_mask.bool()
        if pad_mask.sum() == 0:
            return torch.tensor(0.0).to(y_pred)

        y_pred = y_pred[pad_mask]
        y_true = y_true[pad_mask]
        loss = self.bce(y_pred, y_true)

        if mixup_coeffs is not None:
            mixup_coeffs = mixup_coeffs[:, None].expand_as(pad_mask)
            mixup_coeffs = mixup_coeffs[pad_mask]
            loss = loss * mixup_coeffs

        return loss.mean()