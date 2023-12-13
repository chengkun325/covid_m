import argparse
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from tqdm import tqdm
from torchvision.transforms import transforms as T

from utils.factory import model_factory, dataset_factory
from utils.data_format import print_metric
# from utils.onehot_convert import *
from utils.metrics import *

import matplotlib.pyplot as plt


def get_model(model_path, model_name, class_num: int = 1):
    model = model_factory(model_name)(n_channel=1, n_classes=class_num)
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(torch.device('cuda:0'))
    return model


def test(model, test_data_path, device=torch.device('cpu'), dataset_class: str = "Covid19Dataset", class_num: int = 1):
    if class_num <= 2:
        metric_list = list(BINARY_METRIC_MAP.keys())
    else:
        metric_list = list(CONFUSION_MATRIX_METRIC_MAP.keys())
    image_transform = T.Compose([
        #     T.Resize((224, 224)),
        T.ToTensor(),
    ])

    mask_transform = T.Compose([
        #     T.Resize((224, 224), Image.NEAREST),
        T.ToTensor(),
    ])
    dataset_cls = dataset_factory(dataset_class)

    test_data = dataset_cls(
        test_data_path, image_transform, mask_transform, is_file=True)
    test_dataloader = DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=4)

    model.eval()
    with torch.no_grad():
        epoch_metric = AvgMetricPool()
        for batch_idx, (x, y) in tqdm(enumerate(test_dataloader)):
            image = x.to(device)
            true_mask = y.to(device)

            pred_mask = model(image)

            if class_num > 2:
                pred_mask = torch.softmax(pred_mask, dim=1)
            else:
                pred_mask = torch.sigmoid(pred_mask)
#             loss = criterion(pred_mask, true_mask)
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
#             batch_metric["loss"] = loss.item()
            epoch_metric.add_batch(batch_metric)

            torch.cuda.empty_cache()
            del x, y, image, true_mask, pred_mask

        return epoch_metric.avg_metric


def get_args():
    parser = argparse.ArgumentParser(
        description='Test model on images and target masks')

    parser.add_argument('--net', '-m', type=str, default="UNet",
                        help='Model used to start a test process')
    parser.add_argument('--checkpoints', '-ck', type=str, default="UNet",
                        help='Trained model')
    parser.add_argument('--class_num', '-cn', type=int,
                        default=2, help='Number of classes')

    parser.add_argument('--dataset_cls', '-dcls', type=str,
                        default="LungSegDataset", help='The class used to load train_data')

    parser.add_argument("--test_data", '-td', type=str,
                        default="", help='Data used to test the trained model')

    parser.add_argument('--abs', '-a', type=bool, default=False,
                        help='Wether the test_data_path/checkpoints_path is abs path or not')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoints_path = args.checkpoints
    test_data_path = args.test_data
    if not args.abs:
        checkpoints_path = "./checkpoints/{}".format(checkpoints_path)
        test_data_path = "/zhangtong/dealed_dataset/{}".format(test_data_path)
    model = get_model(checkpoints_path, args.net, args.class_num)
    print("start test")
    metric = test(model, test_data_path, device, args.dataset_cls, args.class_num)
    print("test end")
    print(print_metric(metric))
# end main
