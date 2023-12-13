import json

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa


class LungSegDataset(Dataset):
    def __init__(self, file_path_or_data, transforms=None, mask_transforms=None, is_file=True):
        if is_file:
            with open(file_path_or_data, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = file_path_or_data
        # self.data = self.data[:10]
        self.transform = transforms
        self.mask_transform = mask_transforms

    def __getitem__(self, index):
        image = np.load(self.data[index]['image_path'])
        mask = np.load(self.data[index]['lung_mask'])
        if np.max(image) > 1.0:
            image = image / 255.0
        image = image[:, :, np.newaxis]
        mask = mask[:, :, np.newaxis]

        image = np.expand_dims(image, axis=0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.int32)
        seq2 = iaa.Sequential(
            [iaa.Resize({"height": 224, "width": 224},  interpolation='area')])
        image, mask = seq2(images=image, segmentation_maps=mask)
        image = image[:, :, :, 0]
        mask = mask[:, :, :, 0]
        # image = Image.fromarray(image).convert('L')
        # mask = Image.fromarray(mask).convert('L')

        # if self.transform:
        #     image = self.transform(image)
        # if self.mask_transform:
        #     mask = self.mask_transform(mask)
        return torch.Tensor(image), torch.Tensor(mask)

    def __len__(self):
        return len(self.data)
