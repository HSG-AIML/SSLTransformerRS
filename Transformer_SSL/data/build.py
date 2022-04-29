# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import torch
import numpy as np
from PIL import ImageFilter, ImageOps
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2

# from timm.data.transforms import _pil_interp

from .cached_image_folder import CachedImageFolder
from .custom_image_folder import CustomImageFolder
from .samplers import SubsetRandomSampler


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True, config=config
    )
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = (
        config.AUG.MIXUP > 0
        or config.AUG.CUTMIX > 0.0
        or config.AUG.CUTMIX_MINMAX is not None
    )
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == "imagenet":
        prefix = "train" if is_train else "val"
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(
                config.DATA.DATA_PATH,
                ann_file,
                prefix,
                transform,
                cache_mode=config.DATA.CACHE_MODE if is_train else "part",
            )
        else:
            # ToDo: test custom_image_folder
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = CustomImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    if config.AUG.SSL_AUG:
        if config.AUG.SSL_AUG_TYPE == "byol":
            if config.train_mode == "sen12ms":
                # statistics form uniform sample of 10k observations from SEN12MS
                normalize = A.Normalize(
                    mean=[
                        0.7326,  # s1
                        0.3734,  # s1
                        80.2513,
                        67.1305,
                        61.9878,
                        61.7679,
                        73.5373,
                        105.9787,
                        121.4665,
                        118.3868,
                        132.6419,
                        42.9694,
                        1.3114,
                        110.6207,
                        74.3797,
                    ],
                    std=[
                        0.1634,  # s1
                        0.1526,  # s1
                        4.5654,
                        7.4498,
                        9.4785,
                        14.4985,
                        14.3098,
                        20.0204,
                        24.3366,
                        25.5085,
                        27.1181,
                        7.5455,
                        0.1892,
                        24.8511,
                        20.4592,
                    ],
                )

                transform_1 = A.Compose(
                    [
                        A.RandomResizedCrop(
                            config.DATA.IMG_SIZE,
                            config.DATA.IMG_SIZE,
                            scale=(config.AUG.SSL_AUG_CROP, 1.0),
                        ),
                        A.HorizontalFlip(),
                        A.GaussianBlur(p=1.0),
                        normalize,
                        ToTensorV2(),
                    ]
                )
                transform_2 = A.Compose(
                    [
                        A.RandomResizedCrop(
                            config.DATA.IMG_SIZE,
                            config.DATA.IMG_SIZE,
                            scale=(config.AUG.SSL_AUG_CROP, 1.0),
                        ),
                        A.HorizontalFlip(),
                        A.GaussianBlur(p=1.0),
                        normalize,
                        ToTensorV2(),
                    ]
                )
            else:
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

                transform_1 = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.0)
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                        ),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur()], p=1.0),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
                transform_2 = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.0)
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                        ),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur()], p=0.1),
                        transforms.RandomApply([ImageOps.solarize], p=0.2),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )

            transform = (transform_1, transform_2)
            return transform
        else:
            raise NotImplementedError

    if config.AUG.SSL_LINEAR_AUG:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(config.DATA.IMG_SIZE + 32),
                    transforms.CenterCrop(config.DATA.IMG_SIZE),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transform

    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER
            if config.AUG.COLOR_JITTER > 0
            else None,
            auto_augment=config.AUG.AUTO_AUGMENT
            if config.AUG.AUTO_AUGMENT != "none"
            else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4
            )
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(
                    size, interpolation=_pil_interp(config.DATA.INTERPOLATION)
                ),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION),
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultibandGrayscale(object):
    """Create 'Grayscale' version of multiband imagery
    where r==g==b==...==..."""

    def __call__(self, x):
        return np.stack([np.mean(x, axis=0)] * x.shape[0])
