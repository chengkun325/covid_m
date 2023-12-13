import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader

from torchvision.transforms import transforms as T

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from utils.metrics import *
from utils.data_format import print_metric
from utils.factory import model_factory, loss_factory, dataset_factory


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_net(
    net: str,
    loss_func: str,
    dataset_cls: str,
    train_data: str,
    valid_data: str,
    extra: str,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    class_num: int = 1,
    *args, **kwargs
):

    file_name = "{}_{}_{}_{}_{}_{}".format(
        net, loss_func, batch_size, epochs, learning_rate, extra)
    # 定义模型
    class_num = class_num if class_num > 2 else 1
    model_cls = model_factory(net)
    net = model_cls(n_channel=1, n_classes=class_num)

    if torch.cuda.is_available():
        device_ids = [i for i in range(torch.cuda.device_count())]
        # 多GPU并行训练
        net = nn.DataParallel(net, device_ids=device_ids)
        net = net.to(device)

    if class_num <= 2:
        metric_list = list(BINARY_METRIC_MAP.keys())
    else:
        metric_list = list(CONFUSION_MATRIX_METRIC_MAP.keys())

    # 加载数据
    if valid_data == "":
        with open(train_data, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        train_data, valid_data = train_test_split(
            dataset, test_size=0.2, random_state=101)
    else:
        with open(train_data, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(valid_data, 'r', encoding='utf-8') as f:
            valid_data = json.load(f)

    # 加载数据
    image_transform = T.Compose([
        # T.Resize((256, 256)),
        T.ToTensor(),
    ])
    mask_transform = T.Compose([
        # T.Resize((256, 256)),
        T.ToTensor(),
    ])
    DatasetCls = dataset_factory(dataset_cls)
    train_dataset = DatasetCls(
        train_data, image_transform, mask_transform, is_file=False)
    valid_dataset = DatasetCls(
        valid_data, image_transform, mask_transform, is_file=False)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 损失函数定义
    loss_cls = loss_factory(loss_func)
    criterion = loss_cls(num_classes=class_num)

    # 优化器定义
    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=1e-8)

    scheduler = ReduceLROnPlateau(optimizer,mode='max', factor=0.5, patience=10,min_lr=1e-6, verbose=False)

    print("start train model".center(50, "="))

    best_score = float("-inf")
    best_score_metric = "Dice" if class_num <= 2 else "mDice"

    # 开始训练
    for epoch in range(1, epochs + 1):
        print("Starting train epoch {}/{}, lr: {}".format(epoch, epochs, str(optimizer.param_groups[0]["lr"])).center(50, "="))

        # 模型训练阶段
        net.train()
        epoch_metric = AvgMetricPool()
        start_train = time.time()
        for batch_idx, (x, y) in enumerate(train_dataloader):
            image = x.to(device)
            true_mask = y.to(device)

            pred_mask = net(image)
            if class_num > 2:
                pred_mask = torch.softmax(pred_mask, dim=1)
            else:
                pred_mask = torch.sigmoid(pred_mask)
            loss = criterion(pred_mask, true_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if class_num > 2:
                pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
                true_mask = torch.argmax(true_mask, dim=1, keepdim=True)
                batch_metric = cal_metric_by_cm(
                    true_mask, pred_mask, class_num=class_num, metric_list=metric_list, rm_bg=True)
            else:
                pred_mask = torch.where(pred_mask >= 0.5, torch.ones_like(
                    pred_mask), torch.zeros_like(pred_mask))
                batch_metric = cal_binary_metric(
                    true_mask, pred_mask, metric_list=metric_list)
            batch_metric["loss"] = loss.item()
            epoch_metric.add_batch(batch_metric)
            # print("Train Epoch {}/{} batch {}/{} loss: {:.4f}{}".format(epoch, epochs, batch_idx+1, len(train_dataloader), loss.item(), print_metric(batch_metric)))

            torch.cuda.empty_cache()
            del x, y, image, true_mask, pred_mask, loss

        end_train = time.time()
        print("Train Epoch {}/{}{}".format(epoch, epochs,
              print_metric(epoch_metric.avg_metric)))

        print("Train Epoch {}/{} duration: {:.2f}s".format(epoch, epochs, end_train - start_train))
        # 模型验证
        net.eval()
        epoch_metric = AvgMetricPool()
        start_valid = time.time()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(valid_dataloader):
                image = x.to(device)
                true_mask = y.to(device)

                pred_mask = net(image)

                if class_num > 2:
                    pred_mask = torch.softmax(pred_mask, dim=1)
                else:
                    pred_mask = torch.sigmoid(pred_mask)
                loss = criterion(pred_mask, true_mask)
                if class_num > 2:
                    pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
                    true_mask = torch.argmax(true_mask, dim=1, keepdim=True)
                    batch_metric = cal_metric_by_cm(
                        true_mask, pred_mask, class_num=class_num, metric_list=metric_list, rm_bg=True)
                else:
                    pred_mask = torch.where(pred_mask >= 0.5, torch.ones_like(
                        pred_mask), torch.zeros_like(pred_mask))
                    batch_metric = cal_binary_metric(
                        true_mask, pred_mask, metric_list=metric_list)
                batch_metric["loss"] = loss.item()
                epoch_metric.add_batch(batch_metric)

                # print("Valid Epoch {}/{} batch {}/{} loss: {:.4f}{}".format(epoch, epochs, batch_idx+1, len(valid_dataloader), loss.item(), print_metric(batch_metric)))

                torch.cuda.empty_cache()
                del x, y, image, true_mask, pred_mask, loss
        end_valid = time.time()
        scheduler.step(epoch_metric.avg_metric["IoU"] if class_num <= 2 else epoch_metric.avg_metric["mIoU"])
        print("Valid Epoch {}/{}{}".format(epoch, epochs,
              print_metric(epoch_metric.avg_metric)))
        print("Valid Epoch {}/{} duration: {:.2f}s".format(epoch, epochs, end_valid - start_valid))
        if epoch_metric.avg_metric[best_score_metric] > best_score and epoch > 50:
            best_score = epoch_metric.avg_metric[best_score_metric]
            torch.save(net.state_dict(),
                       "./checkpoints/{}.pth".format(file_name))
    return "{}.pth".format(file_name)

def get_args():
    parser = argparse.ArgumentParser(
        description='Train model on images and target masks')

    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('--class_num', '-c', type=int,
                        default=2, help='Number of classes')

    parser.add_argument('--net', '-m', type=str, default="UNet",
                        help='Model used to start a train process')
    parser.add_argument('--loss_func', '-l', type=str, default="BinaryDiceLoss",
                        help='Loss function used in the train process')
    parser.add_argument('--dataset_cls', '-dcls', type=str,
                        default="LungSegDataset", help='The class used to load train_data')

    # parser.add_argument("--train_data", '-td', type=str, required=True, help="Data used to train the model")
    parser.add_argument("--train_data", '-td', type=str,
                        help="Data used to train the model")
    parser.add_argument("--valid_data", '-vd', type=str,
                        default="", help='Data used to valid the trained model')

    parser.add_argument(
        "--extra", type=str, default=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    # not used in current experiments
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float,
                        default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true',
                        default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    model_file = train_net(**vars(args))
    
    
    with open("test.sh", "a", encoding="utf-8") as f:
        f.write("python test.py --net {} --dataset_cls {} --class_num {} --test_data {} --checkpoints {}\n".format(args.net, args.dataset_cls, args.class_num if args.class_num != 2 else 1, str(os.path.basename(args.train_data)).replace("train", "test"), model_file))
# end main
