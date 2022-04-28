import os

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import random
import sys
import json
import wandb
import numpy as np
import torch

from utils import dotdictify
from d_swin_utils import SwinTrainer
from dfc_dataset import DFCDataset

sys.path.insert(0, "./Transformer_SSL")
from Transformer_SSL.models import build_model
from Transformer_SSL.models.swin_transformer import DoubleSwinTransformer
from Transformer_SSL.optimizer import build_optimizer
from Transformer_SSL.lr_scheduler import build_scheduler

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

with open("configs/backbone_config.json", "r") as fp:
    config = json.load(fp)

run = wandb.init(config=config, project="d-swin-backbone")

config = wandb.config
config["run_name"] = run.name

config = dotdictify(config)

# Input sizes don't change
torch.backends.cudnn.benchmark = True

# Ensure deterministic behavior
# torch.backends.cudnn.deterministic = True
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

print(f"{config.batch_size=}")
print(f"{config.image_px_size=}")
print(f"{config.model_config.DATA.IMG_SIZE=}")

assert config.image_px_size == config.model_config.DATA.IMG_SIZE


train_dataset = DFCDataset(
    config.train_dir,
    mode=config.train_mode,
    transforms=config.transforms,
    clip_sample_values=config.clip_sample_values,
    used_data_fraction=config.used_data_fraction,
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
    drop_last=True,
)

s1_backbone = build_model(config.model_config)
config.model_config.MODEL.SWIN.IN_CHANS = 13
s2_backbone = build_model(config.model_config)
model = DoubleSwinTransformer(s1_backbone, s2_backbone)

optimizer = build_optimizer(config, model)
lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

trainer = SwinTrainer(
    model=model, optimizer=optimizer, scheduler=lr_scheduler, args=config
)

s = trainer.train(train_loader, val_loader)
