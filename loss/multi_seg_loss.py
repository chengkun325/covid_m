# loss.py
import torch.nn as nn
import torch

def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)

    return loss.sum() / N

class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None, reduction='mean', *args, **kwargs):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(0, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice



class SoftDiceCELoss(nn.Module):

    def __init__(self, num_classes, activation=None, reduction='mean', *args, **kwargs):
        super(SoftDiceCELoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(0, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        dice_loss = 1 - mean_dice
        true_mask = torch.argmax(y_true, dim=1, keepdim=True)
        ce_loss = self.ce(y_pred, true_mask.squeeze(1).long())
        
        return 0.5 * dice_loss + 0.5 * ce_loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean', *args, **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses
