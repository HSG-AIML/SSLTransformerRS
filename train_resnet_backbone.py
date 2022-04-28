import os

os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import random

import wandb
import numpy as np
import torch

from dfc_dataset import DFCDataset
from resnet_simclr import DoubleResNetSimCLR
from simclr_double_backbone import SimCLRDoubleBackbone

# os.environ['WANDB_MODE'] = 'offline'
wandb.login()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu:0")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu:0")


run = wandb.init(
    project="simclr-double-backbone",
    config={
        "epochs": 201,
        "learning_rate": 0.00003,
        "batch_size": 200,
        "seed": 42,
        "num_classes": 8,
        "dataloader_workers": 8,
        # "train_dir" : "/netscratch/lscheibenreif/grss-dfc-20",
        # "train_mode" : "test",
        "train_dir": "/ds2/remote_sensing/sen12ms",
        "train_mode": "sen12ms",
        "val_dir": "/netscratch/lscheibenreif/grss-dfc-20",
        "val_mode": "validation",
        "clip_sample_values": True,
        "transforms": None,
        "train_used_data_fraction": 1,
        "s1_input_channels": 2,
        "s2_input_channels": 13,
        "learning_rate_schedule": {
            100: 0.1
        },  # {50 : 0.1} at epoch `key` multiply lr by `value`
        "image_px_size": 128,
        "cover_all_parts_validation": True,  # take image_px_size crops s.t. one epoch covers every pixel of every scene
        "cover_all_parts_train": False,  # take image_px_size crops at random offsets
        "balanced_classes_train": False,  # take crops from observations from small classes more frequently
        "balanced_classes_validation": False,
        "target": "dfc_label",  # "lc_label",
        ###### simclr specific parameters #####
        "arch": "resnet50",
        "weight_decay": 1e-4,
        "fp16_precision": True,
        "out_dim": 128,
        "temperature": 0.07,
        "n_views": 2,  # only supported number
        "device": device,
        "disable_cuda": False,
        "log_every_n_steps": 1000,
        "use_logging": False,
    },
)

config = wandb.config
config["run_name"] = run.name

# Input sizes don't change
torch.backends.cudnn.benchmark = True

# Ensure deterministic behavior
# torch.backends.cudnn.deterministic = True
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

train_dataset = DFCDataset(
    config.train_dir,
    mode=config.train_mode,
    transforms=config.transforms,
    clip_sample_values=config.clip_sample_values,
    used_data_fraction=config.train_used_data_fraction,
    image_px_size=config.image_px_size,
    cover_all_parts=config.cover_all_parts_train,
    balanced_classes=config.balanced_classes_train,
)
val_dataset = DFCDataset(
    config.val_dir,
    mode=config.val_mode,
    transforms=config.transforms,
    clip_sample_values=config.clip_sample_values,
    image_px_size=config.image_px_size,
    cover_all_parts=config.cover_all_parts_validation,
    balanced_classes=config.balanced_classes_validation,
)

# train_dataset = DFCDataset(config.train_dir, mode=config.train_mode, transforms=config.transforms, clip_sample_values=config.clip_sample_values, used_data_fraction=config.train_used_data_fraction)
# val_dataset = DFCDataset(config.val_dir, mode=config.val_mode, transforms=config.transforms, clip_sample_values=config.clip_sample_values)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.dataloader_workers,
    drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.dataloader_workers,
)

model = DoubleResNetSimCLR(base_model=config.arch, out_dim=config.out_dim)
model.backbone1.conv1 = torch.nn.Conv2d(
    config.s1_input_channels,
    64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
)
model.backbone2.conv1 = torch.nn.Conv2d(
    config.s2_input_channels,
    64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
)

optimizer = torch.optim.Adam(
    model.parameters(), config.learning_rate, weight_decay=config.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
)

simclr = SimCLRDoubleBackbone(
    model=model, optimizer=optimizer, scheduler=scheduler, args=config
)

s = simclr.train(train_loader)
