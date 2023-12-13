import json
import itertools
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from torch.utils.data.sampler import Sampler


class Covid19SemiDataset(Dataset):
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
        if np.max(image) > 1.0:
            image = image / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.uint8)

        return torch.Tensor(image), torch.Tensor(mask)

    def __len__(self):
        return len(self.data)



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
