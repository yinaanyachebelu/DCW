
#########

# code to find mean and std for dataset derived from : https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html

import torchvision
from torchvision import transforms
import torch.utils.data
from torch.utils.data import DataLoader
import datasets.transforms as T
import datasets.samplers as samplers
from pathlib import Path

from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection

from util.misc import get_local_rank, get_local_size
import util.misc as utils
import os
import cv2


data_path = './datasets/Dataset_final/DATA_0_COCO_format/'


class WeedData(Dataset):

    def __init__(self,
                 data,
                 directory,
                 transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # import
        path = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def make_coco_transforms(image_set):

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor()
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    # Each key in dict below is tuple  : ( Path to images, Annotation file for those images  )
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'{mode}_test2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset


def batch_mean_and_sd(loader):

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        print(images.shape)
    #     b, c, h, w = images.shape
    #     nb_pixels = b * h * w
    #     sum_ = torch.sum(images, dim=[0, 2, 3])
    #     sum_of_square = torch.sum(images ** 2,
    #                               dim=[0, 2, 3])
    #     fst_moment = (cnt * fst_moment + sum_) / (
    #                   cnt + nb_pixels)
    #     snd_moment = (cnt * snd_moment + sum_of_square) / (
    #                         cnt + nb_pixels)
    #     cnt += nb_pixels

    # mean, std = fst_moment, torch.sqrt(
    #   snd_moment - fst_moment ** 2)
    #    return mean, std


if __name__ == '__main__':

    batch_size = 4

    dataset_train = build_dataset(image_set='train', args=args)
    sampler_train = samplers.DistributedSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=1,
                                   pin_memory=True)

    # mean, std = batch_mean_and_sd(data_loader_train)
    # print("mean and std: \n", mean, std)
    batch_mean_and_sd(data_loader_train)
