# metircs.py

import torch
import torch.nn as nn
import numpy as np
from medpy.metric import binary
from sklearn.metrics import *

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


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
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

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N




def hd(predict, label):
    try:
        hd_val = binary.hd(predict, label)
        
    except Exception:
        hd_val = 0.0
    return hd_val

def hd95(predict, label):
    try:
        hd_val = binary.hd95(predict, label)
    except Exception:
        hd_val = 0.0
    return hd_val

def jc(predict, label):
    try:
        jc_val = binary.jc(predict, label)
    except Exception:
        jc_val = 0.0
    return jc_val

def f1_score(predict, label):
    sensitivity = binary.sensitivity(predict, label)
    specificity = binary.specificity(predict, label)
    f1 = 2 * (sensitivity * specificity) / (sensitivity + specificity)
    return f1

def pa(pred, target):
    assert pred.shape == target.shape, \
        "预测结果和真实标签的形状不一致：{} vs {}".format(pred.shape, target.shape)
    
    # 将预测结果和真实标签转换为一维数组
    pred = pred.ravel()
    target = target.ravel()
    
    # 统计预测正确的像素数量
    correct_pixels = np.sum(pred == target)
    
    # 计算像素准确率
    accuracy = correct_pixels / len(target)
    
    return accuracy

BINARY_METRIC_MAP = {
    "Dice": binary.dc,
    "IoU": jc,
    "Sen": binary.sensitivity,
    "Prec": binary.precision,
    "Spec": binary.specificity,
    "HD": hd,
    "HD95": hd95,
    "PA": pa,
    "F1": f1_score
}



def gen_confusion_matrix(ground_truth:np.array, predict_label:np.array, class_num:int=4):
    # 生成混淆矩阵
#     print(ground_truth.shape, predict_label.shape)
    assert ground_truth.shape == predict_label.shape, "预测图与ground truth尺寸不一致"
    ground_truth = ground_truth.flatten()
    predict_label = predict_label.flatten()
    return confusion_matrix(ground_truth, predict_label, labels=[i for i in range(class_num)])


def pixel_accuracy(cm:np.array):
    # comment: 预测正确的像素占总像素的比例
    # PA = acc = (TP+TN) / (TP + TN + FP + TN)
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    return acc


def class_pixel_accuracy(cm:np.array):
    # 每个类别预测正确的像素占比
    # class acc = TP / (TP + FP)
    class_acc = np.nan_to_num(np.diag(cm) / np.sum(cm, axis=0))
    return class_acc


def mean_pixel_accuracy(cm:np.array, rm_bg=True):
    # MPA 均像素精度，计算每个类内被正确分类的像素比例，求平均值
    if rm_bg:
        return np.nanmean(class_pixel_accuracy(cm)[1:])
    return np.nanmean(class_pixel_accuracy(cm))


def intersection_over_union(cm:np.array):
    # intersection = TP union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    # 返回一个列表，表示每个类的IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    
    IoU = np.nan_to_num(intersection / union)
    return IoU


def mean_intersection_over_union(cm:np.array, rm_bg=True):
    # comment: 求各个类别的IoU的平均值
    if rm_bg:
        np.nanmean(intersection_over_union(cm)[1:])
    return np.nanmean(intersection_over_union(cm))


def DSC(cm:np.array):
    # dice = (2 * TP) / (2 * TP + FN + FP)
    dice_val = (2 * np.diag(cm)) / (np.sum(cm, axis=0) + np.sum(cm, axis=1))
    return np.nan_to_num(dice_val)


def mean_dice(cm:np.array, rm_bg=True):
    if rm_bg:
        return np.nanmean(DSC(cm)[1:])
    return np.nanmean(DSC(cm))


def Sen(cm:np.array):
    # 查全率，召回率：预测正确的数据占总的比例
    # TP / (TP + FN)
    
    sen = np.diag(cm) / np.sum(cm, axis=1)
    return np.nan_to_num(sen)


def Spec(cm:np.array):
    # comment: 特异性
    # TN / (TN + FP) 
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    spec = TN / (TN + FP)
    return np.nan_to_num(spec)


CONFUSION_MATRIX_METRIC_MAP = {
    "PA": pixel_accuracy,
    "CPA": class_pixel_accuracy,
    "MPA": mean_pixel_accuracy,
    "IoU": intersection_over_union,
    "MIoU": mean_intersection_over_union,
    "Dice": DSC,
    "mDice": mean_dice,
    "Sen": Sen,
    "Spec": Spec
}


def cal_binary_metric(ground_truth:torch.Tensor, predict_label:torch.Tensor, metric_list:list=["Dice"]):
    ground_truth = ground_truth.cpu().detach().numpy()
    predict_label = predict_label.cpu().detach().numpy()
    result_map = {}
    
    for metric in metric_list:
        try:
            metric_val = BINARY_METRIC_MAP[metric](ground_truth, predict_label)
        except:
            metric_val = 0
        result_map[metric] = metric_val
    
    return result_map


def cal_metric_by_cm(ground_truth, predict_label, class_num=4, metric_list:list=["Dice"], rm_bg=True):
    ground_truth = ground_truth.cpu().detach().numpy()
    predict_label = predict_label.cpu().detach().numpy()
    result_map = {}
    
    cm = gen_confusion_matrix(ground_truth, predict_label, class_num=class_num)
    
    for metric in metric_list:
        metric_val = 0.0
        if metric[0] in ["m", "M"]:
            metric_val = CONFUSION_MATRIX_METRIC_MAP[metric](cm, rm_bg=rm_bg)
        else:
            metric_val = CONFUSION_MATRIX_METRIC_MAP[metric](cm)
        result_map[metric] = metric_val
    
    return result_map



class AvgMetricPool:
    def __init__(self) -> None:
        self._num_len = 0
        self._metric = dict()

    @property
    def avg_metric(self):
        # comment: 
        return self._metric
    
    
    def add_batch(self, metric_map:dict):
        if not self._metric:
            self._metric = metric_map
        else:
            assert set(self._metric.keys()) ==  set(metric_map.keys()), "The keys in metric_map are mismatch with the previous"
            for key in metric_map:
                self._metric[key] = (self._metric[key] * self._num_len + metric_map[key]) / (self._num_len + 1)
        self._num_len += 1
    
    def reset(self):
        self._metric = dict()
        self._num_len = 0




class SegmentationMetric:
    
    def __init__(self, class_num, extra_metric_num=0) -> None:
        self.class_num = class_num
        self.confusion_matrix = np.zeros((class_num, class_num))
        
        self._metric_arr = np.array([0.0] * extra_metric_num)
        self.num_len = 0
        
    
    # np.diag 取对角线元素， np.sum 求和，可以指定维度
    def pixel_accuracy(self):
        # comment: 预测正确的像素占总像素的比例
        # PA = acc = (TP+TN) / (TP + TN + FP + TN)
        acc = np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)
        return acc
    
    def class_pixel_accuracy(self):
        # 每个类别预测正确的像素占比
        # class acc = TP / (TP + FP)
        class_acc = np.nan_to_num(np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0))
        return class_acc
    
    def mean_pixel_accuracy(self, rm_bg=True):
        # MPA 均像素精度，计算每个类内被正确分类的像素比例，求平均值
        if rm_bg:
            return np.nanmean(self.class_pixel_accuracy()[1:])
        return np.nanmean(self.class_pixel_accuracy())
    
    def intersection_over_union(self):
        # intersection = TP union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        # 返回一个列表，表示每个类的IoU
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
        
        IoU = np.nan_to_num(intersection / union)
        
        return IoU

    def mean_intersection_over_union(self, rm_bg=True):
        # comment: 求各个类别的IoU的平均值
        if rm_bg:
            np.nanmean(self.intersection_over_union()[1:])
        return np.nanmean(self.intersection_over_union())

    def frequency_weighted_intersection_over_union(self):
        # 这个不能确定是否正确, 已确定
        # comment: FWIoU，频权交并⽐:为MIoU的⼀种提升，这种⽅法根据每个类出现的频率为其设置权重。
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)

        IoU = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * IoU[freq > 0]).sum()
        return FWIoU
    
    def sensitivity(self):
        # 查全率，召回率：预测正确的数据占总的比例
        # TP / (TP + FN)
        
        sen = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        return np.nan_to_num(sen)

    def specifity(self):
        # comment: 特异性
        # TN / (TN + FP) 
        FP = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)  
        FN = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        TP = np.diag(self.confusion_matrix)
        TN = self.confusion_matrix.sum() - (FP + FN + TP)
        spec = TN / (TN + FP)
        return np.nan_to_num(spec)
    
    def dice(self):
        # dice = (2 * TP) / (2 * TP + FN + FP)
        dice = (2 * np.diag(self.confusion_matrix)) / (np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=1))
        return np.nan_to_num(dice)
    
    def mean_dice(self, rm_bg=True):
        if rm_bg:
            return np.nanmean(self.dice()[1:])
        return np.nanmean(self.dice())
    
    def gen_confusion_matrix(self, gt:np.ndarray, pred:np.ndarray):
        # 生成混淆矩阵
        assert gt.shape == pred.shape, "预测图与ground truth尺寸不一致"
        gt = gt.flatten()
        pred = pred.flatten()
        return confusion_matrix(gt, pred, labels=[i for i in range(self.class_num)])
    
    def add_batch(self, gt, pred):
        # 添加一个batch的数据到混淆矩阵中
        self.confusion_matrix += self.gen_confusion_matrix(gt, pred)
        
    def add(self, metrics):
        if isinstance(metrics, list) or isinstance(metrics, tuple):
            metrics = np.array(metrics)
        self._metric_arr = (self._metric_arr * self.num_len + metrics) / (self.num_len + 1)
        self.num_len += 1
        
    @property
    def avg_metric(self):
        return self._metric_arr
    
    def reset(self):
        # 重置混淆矩阵为0
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))
        self._metric_arr *= 0.0
        self.num_len = 0

