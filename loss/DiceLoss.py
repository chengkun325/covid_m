import torch
import torch.nn as nn


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        def activation_fn(x): return x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N


class BinaryDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation=None, *args, **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt, activation=self.activation)

class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self, Beta=0.5, *args, **kwargs):
        super().__init__()
        self.dice = BinaryDiceLoss()
        self.cross_entropy = nn.BCEWithLogitsLoss()
        self.beta = Beta

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)ss
        cross_entropy = self.cross_entropy(y_pred, y_true)
        return self.beta * dice + (1-self.beta) * cross_entropy