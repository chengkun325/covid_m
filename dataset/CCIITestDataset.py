import json

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from utils.data_format import mask_to_onehot


class CCIITestDataset(Dataset):
    def __init__(self, file_path_or_data, transforms=None, mask_transforms=None, is_file=True):
        if is_file:
            with open(file_path_or_data, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = file_path_or_data
        self.transform = transforms
        self.mask_transform = mask_transforms

    def __getitem__(self, index):
        image = np.load(self.data[index]['image_path'])
        mask = np.load(self.data[index]['infection_mask'])
        org_image = np.load(self.data[index]['org_image_path'])
        org_mask = np.load(self.data[index]['org_infection_mask'])
        if np.max(image) > 1.0:
            image = image / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.uint8)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask_to_onehot(mask, palette=[[0], [1], [2], [3]])
        mask = np.swapaxes(mask, 0, -1)
        mask = mask[:, :, :, 0]
        return torch.Tensor(image), torch.Tensor(mask), torch.Tensor(org_image), torch.Tensor(org_mask), self.data[index]['location']

    def __len__(self):
        return len(self.data)
