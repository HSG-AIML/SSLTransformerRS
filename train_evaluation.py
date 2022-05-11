import os

os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import json
import random
import argparse
from distutils.util import strtobool

import numpy as np
from tqdm import tqdm
import wandb
import torch
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F

from dfc_dataset import DFCDataset
from dfc_model import DualBaseline
from resnet_simclr import NormalSimCLRDownstream, DoubleResNetSimCLRDownstream
from metrics import ClasswiseMultilabelMetrics, ClasswiseAccuracy, PixelwiseMetrics
from utils import save_checkpoint_single_model, dotdictify
from validation_utils import validate_all
from Transformer_SSL.models import build_model
from Transformer_SSL.models.swin_transformer import (
    DoubleSwinTransformerSegmentation,
    DoubleSwinTransformerDownstream,
    DownstreamSharedDSwin,
)

model_name_map = {
    "resnet18": "baseline",
    "resnet50": "baseline",
    "DualBaseline": "dual-baseline",
    "SwinBaseline": "swin-baseline",
    "DualSwinBaseline": "dual-swin-baseline",
    "NormalSimCLRDownstream": "normal-simclr",
    "DoubleAlignmentDownstream": "alignment",
    "DoubleResNetSimCLRDownstream": "simclr",
    "DoubleSwinTransformerDownstream": "swin-t",
    "DoubleSwinTransformerSegmentation": "swin-t",
    "MobyDownstream": "moby",
    "DownstreamSharedDSwin": "shared-swin-t",
    "SharedDSwinBaseline": "shared-swin-t-baseline",
}
target_name_map = {
    "dfc_label": "single-classification",
    "dfc_multilabel_one_hot": "multi-classification",
    "dfc": "pixel-classification"
}
bool_args = [
    "clip_sample_values",
    "only_rgb",
    "rgb_plus_s1",
    "cover_all_parts_validation",
    "cover_all_parts_train",
    "balanced_classes_train",
    "balanced_classes_validation",
    "s1_normalization_fixed",
    "finetuning",
    "simclr_dataset",
]

parser = argparse.ArgumentParser(description="train_evaluation_script")

# optimization
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--epochs", default=201, type=int)
parser.add_argument("--learning_rate", default=0.00001, type=float)
parser.add_argument(
    "--classifier_lr", default=3e-6, type=float
)  # lr for the classification layer
parser.add_argument("--batch_size", default=100, type=int)
parser.add_argument("--adam_betas", default=(0.9, 0.999), type=tuple)
parser.add_argument("--weight_decay", default=0.001, type=float)
parser.add_argument("--learning_rate_schedule", default={100: 0.1}, type=dict)

# train set
parser.add_argument(
    "--train_dir", default="/netscratch/lscheibenreif/grss-dfc-20", type=str
)
parser.add_argument("--train_mode", default="validation", type=str)

# create a validation set as 80% of the training set:
parser.add_argument("--create_validation_set", default="False", type=str)

# test set
parser.add_argument(
    "--val_dir", default="/netscratch/lscheibenreif/grss-dfc-20", type=str
)
parser.add_argument("--val_mode", default="test", type=str)
parser.add_argument("--clip_sample_values", default="True", type=str)
parser.add_argument("--transforms", default=None)
parser.add_argument("--num_classes", default=8, type=int)
parser.add_argument("--only_rgb", default="False", type=str)
parser.add_argument("--rgb_plus_s1", default="False", type=str)
parser.add_argument("--dataloader_workers", default=8, type=int)
parser.add_argument("--s1_input_channels", default=2, type=int)
parser.add_argument("--s2_input_channels", default=13, type=int)
parser.add_argument("--train_used_data_fraction", default=1, type=float)
parser.add_argument("--image_px_size", default=224, type=int)
parser.add_argument("--cover_all_parts_validation", default="True", type=str)
parser.add_argument("--cover_all_parts_train", default="False", type=str)
parser.add_argument("--balanced_classes_train", default="True", type=str)
parser.add_argument("--balanced_classes_validation", default="False", type=str)
parser.add_argument("--s1_normalization_fixed", default="True", type=str)
parser.add_argument("--simclr_dataset", default="False", type=str)
parser.add_argument(
    "--out_dim", default=128, type=int
)  # as used in normal-simclr trained checkpoint

# model
parser.add_argument(
    "--model",
    default="resnet18",
    choices=[
        "resnet18",
        "resnet50",
        "SwinBaseline",
        "DualBaseline",
        "DualSwinBaseline",
        #      "DoubleAlignmentDownstream",
        "DoubleResNetSimCLRDownstream",
        "NormalSimCLRDownstream",
        "DoubleSwinTransformerDownstream",
        "DoubleSwinTransformerSegmentation",
        #      "MobyDownstream",
        "DownstreamSharedDSwin",
        "SharedDSwinBaseline",
    ],
    type=str,
)
parser.add_argument(
    "--base_model",
    default="resnet18",
    choices=["resnet18", "resnet50", "VGGEncoder"],
    type=str,
)
parser.add_argument(
    "--target",
    default="dfc_label",
    choices=["dfc_label", "dfc_multilabel_one_hot", "dfc"],
    type=str,
)
parser.add_argument("--finetuning", default="False", type=str)
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument("--embedding_size", default=256, type=int)
parser.add_argument("--wandb_project", default=None, type=str)

args = parser.parse_args()
model_name = model_name_map[args.model]
target_name = target_name_map[args.target]

if args.wandb_project is not None and args.wandb_project != "None":
    project = args.wandb_project
else:
    project = "-".join(["EV", model_name, target_name])

# set up wandb logging
wandb.login()
run = wandb.init(
    project=project,
    config={k: strtobool(v) if k in bool_args else v for k, v in vars(args).items()},
)
config = wandb.config

# remove this to enable different LR for backbone and head!! (for finetuning)
# config.classifier_lr = config.learning_rate

print("CONFIG:", config)

if config.model == "DoubleAlignmentDownstream":
    assert config.base_model == "VGGEncoder", "Wrong base_model for MMA model"

if config.model == "NormalSimCLRDownstream":
    assert (
        config.simclr_dataset
    ), "Set simclr_dataset to true for normal SimCLR training"

# Input sizes don't change
torch.backends.cudnn.benchmark = True

# Ensure deterministic behavior
# torch.backends.cudnn.deterministic = True
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu:0")


print(f"model_name {model_name}")

if model_name == "baseline" or model_name == "swin-baseline":
    input_channels = config.s1_input_channels + config.s2_input_channels

    if model_name == "baseline":
        model = eval(config.model)(pretrained=False, num_classes=config.num_classes)
        model.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    elif model_name == "swin-baseline":

        with open("configs/backbone_config.json", "r") as fp:
            swin_conf = dotdictify(json.load(fp))

        swin_conf.model_config.MODEL.NUM_CLASSES = config.num_classes
        assert config.image_px_size == swin_conf.model_config.DATA.IMG_SIZE
        swin_conf.model_config.MODEL.SWIN.IN_CHANS = input_channels

        model = build_model(swin_conf.model_config)

    else:
        raise ValueError(f"something wrong with model_name: {model_name}")

elif model_name == "dual-baseline":
    if config.base_model == "resnet50":
        model = eval(config.model)(
            config.base_model,
            config.s1_input_channels,
            config.s2_input_channels,
            feature_dim=2 * 2048,
        )
    else:
        # resnet18
        model = eval(config.model)(
            config.base_model, config.s1_input_channels, config.s2_input_channels
        )

elif model_name == "dual-swin-baseline":
    input_channels = config.s1_input_channels + config.s2_input_channels

    with open("configs/backbone_config.json", "r") as fp:
        swin_conf = dotdictify(json.load(fp))

    # swin_conf.model_config.MODEL.NUM_CLASSES = config.num_classes
    assert config.image_px_size == swin_conf.model_config.DATA.IMG_SIZE

    swin_conf.model_config.MODEL.SWIN.IN_CHANS = 2
    s1_backbone = build_model(swin_conf.model_config)
    swin_conf.model_config.MODEL.SWIN.IN_CHANS = 13
    s2_backbone = build_model(swin_conf.model_config)

    model = DoubleSwinTransformerDownstream(
        s1_backbone,
        s2_backbone,
        out_dim=config.num_classes,
        device=device,
        freeze_layers=False,
    )

elif model_name == "normal-simclr":
    checkpoint = torch.load(config.checkpoint, map_location=lambda device, loc: device)
    model = eval(config.model)(
        base_model=config.base_model,
        out_dim=config.out_dim,
        checkpoint=checkpoint,
        num_classes=config.num_classes,
    )

elif model_name == "alignment":
    model = eval(config.model)(config.base_model, device, config)

    # load trained weights
    checkpoint = torch.load(config.checkpoint, map_location=lambda device, loc: device)
    model.load_trained_state_dict(checkpoint)

elif model_name == "simclr":
    input_channels = config.s1_input_channels + config.s2_input_channels
    model = eval(config.model)(config.base_model, config.num_classes)
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

    # load trained weights
    checkpoint = torch.load(config.checkpoint, map_location=lambda device, loc: device)
    model.load_trained_state_dict(checkpoint["state_dict"])

elif model_name == "swin-t":
    input_channels = config.s1_input_channels + config.s2_input_channels

    with open("configs/backbone_config.json", "r") as fp:
        swin_conf = dotdictify(json.load(fp))

    assert config.image_px_size == swin_conf.model_config.DATA.IMG_SIZE

    s1_backbone = build_model(swin_conf.model_config)
    swin_conf.model_config.MODEL.SWIN.IN_CHANS = 13
    s2_backbone = build_model(swin_conf.model_config)
    checkpoint = torch.load(
        config.checkpoint
    )  # "checkpoints/d-swimdistinctive-armadillo-24-epoch150.pth")
    weights = checkpoint["state_dict"]
    s1_weights = {
        k[len("backbone1."):]: v for k, v in weights.items() if "backbone1" in k
    }
    s2_weights = {
        k[len("backbone2."):]: v for k, v in weights.items() if "backbone2" in k
    }
    s1_backbone.load_state_dict(s1_weights)
    s2_backbone.load_state_dict(s2_weights)

    if target_name == "pixel-classification":
        model = DoubleSwinTransformerSegmentation(
            s1_backbone, s2_backbone, out_dim=8, device=device
        )
    else:
        model = DoubleSwinTransformerDownstream(
            s1_backbone, s2_backbone, out_dim=8, device=device
        )

elif model_name == "shared-swin-t":
    with open("configs/shared_backbone_config.json", "r") as fp:
        swin_conf = dotdictify(json.load(fp))

    assert config.image_px_size == swin_conf.model_config.DATA.IMG_SIZE

    ssl_model = build_model(swin_conf.model_config)
    checkpoint = torch.load(config.checkpoint)
    weights = checkpoint["state_dict"]

    ssl_model.load_state_dict(weights)
    model = DownstreamSharedDSwin(ssl_model, config.num_classes)

elif model_name == "shared-swin-t-baseline":
    with open("configs/shared_backbone_config.json", "r") as fp:
        swin_conf = dotdictify(json.load(fp))

    assert config.image_px_size == swin_conf.model_config.DATA.IMG_SIZE

    ssl_model = build_model(swin_conf.model_config)
    model = DownstreamSharedDSwin(ssl_model, config.num_classes)

elif model_name == "moby":
    with open("configs/moby_config.json", "r") as fp:
        moby_conf = dotdictify(json.load(fp))

    weights = torch.load(config.checkpoint)
    # assert config.image_px_size == config.model_config.DATA.IMG_SIZE
    assert moby_conf.model_config.AMP_OPT_LEVEL == "O0"
    assert moby_conf.model_config.MODEL.TYPE == "moby"

    moby_conf.model_config.DATA.BATCH_SIZE = config.batch_size
    moby = build_model(moby_conf.model_config)
    moby.load_state_dict(weights)

    model = moby.encoder
    model.head = torch.nn.Linear(768, config.num_classes)

    # freeze layers up to head
    for name, param in model.named_parameters():
        if name not in ["head.weight", "head.bias"]:
            param.requires_grad = False

else:
    raise ValueError("Invalid model specified")

model = model.to(device)

# ignore label 255 (dataset class sets labels 3,8 (savanna, ice) for lr lc map to 255)
if target_name == "multi-classification":
    criterion = torch.nn.BCELoss(reduction="mean").to(device)
    sigmoid = torch.nn.Sigmoid().to(device)
elif target_name == "single-classification":
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="mean").to(device)
elif target_name == "pixel-classification":
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
else:
    raise ValueError("Invalid target specified")

if model_name in [
    "normal-simclr",
    "alignment",
    "simclr",
    "swin-t",
    "moby",
    "shared-swin-t",
    "shared-swin-t-baseline",
]:
    # all the SSL methods
    if config.finetuning:
        # train all parameters
        param_backbone = []
        param_head = []
        for p in model.parameters():
            if p.requires_grad:
                param_head.append(p)
            else:
                param_backbone.append(p)
            p.requires_grad = True
        # parameters = model.parameters()
        parameters = [
            {"params": param_backbone},  # train with default lr
            {
                "params": param_head,
                "lr": config.classifier_lr,
            },  # train with classifier lr
        ]
        print("Finetuning")
    else:
        # train only final linear layer for SSL methods
        print("Frozen backbone")
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert len(parameters) == 2  # fc.weight, fc.bias
else:
    parameters = model.parameters()

optimizer = torch.optim.Adam(
    parameters,
    lr=config.learning_rate,
    betas=config.adam_betas,
    weight_decay=config.weight_decay,
)

train_dataset = DFCDataset(
    config.train_dir,
    mode=config.train_mode,
    simclr_dataset=config.simclr_dataset,
    transforms=config.transforms,
    clip_sample_values=config.clip_sample_values,
    used_data_fraction=config.train_used_data_fraction,
    image_px_size=config.image_px_size,
    cover_all_parts=config.cover_all_parts_train,
    balanced_classes=config.balanced_classes_train,
    seed=config.seed,
)
# if config.create_validation_set:
#    # create subsampler from training set
val_dataset = DFCDataset(
    config.val_dir,
    mode=config.val_mode,
    simclr_dataset=config.simclr_dataset,
    transforms=config.transforms,
    clip_sample_values=config.clip_sample_values,
    image_px_size=config.image_px_size,
    cover_all_parts=config.cover_all_parts_validation,
    balanced_classes=config.balanced_classes_validation,
    seed=config.seed,
)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=config.dataloader_workers,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.dataloader_workers,
)

step = 0

for epoch in range(config.epochs):
    model.train()
    step += 1

    pbar = tqdm(train_loader)

    # track performance
    epoch_losses = torch.Tensor()
    if target_name == "single-classification":
        metrics = ClasswiseAccuracy(config.num_classes)
    elif target_name == "multi-classification":
        metrics = ClasswiseMultilabelMetrics(config.num_classes)
    elif target_name == "pixel-classification":
        metrics = PixelwiseMetrics(config.num_classes)

    if config.learning_rate_schedule.get(epoch) is not None:
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * config.learning_rate_schedule.get(epoch)

    for idx, sample in enumerate(pbar):

        if "x" in sample.keys():
            if torch.isnan(sample["x"]).any():
                # some s1 scenes are known to have NaNs...
                continue
        else:
            if torch.isnan(sample["s1"]).any() or torch.isnan(sample["s2"]).any():
                # some s1 scenes are known to have NaNs...
                continue

        if model_name == "baseline" or model_name == "swin-baseline":
            s1 = sample["s1"]
            s2 = sample["s2"]
            if config.s1_input_channels == 0:
                # no data fusion
                img = s2.to(device)
            elif config.s2_input_channels == 0:
                img = s1.to(device)
            else:
                # data fusion
                img = torch.cat([s1, s2], dim=1).to(device)

        elif model_name == "normal-simclr":
            x = sample["x"]
            img = x.to(device)

        elif model_name == "moby":
            img = torch.cat([sample["s1"], sample["s2"]], dim=1).to(device)

        elif model_name in [
            "dual-baseline",
            "dual-swin-baseline",
            "alignment",
            "simclr",
            "swin-t",
            "shared-swin-t",
            "shared-swin-t-baseline",
        ]:
            s1 = sample["s1"].to(device)
            s2 = sample["s2"].to(device)
            img = {"s1": s1, "s2": s2}

        if target_name == "single-classification":
            y = sample[config.target].long().to(device)
        elif target_name == "multi-classification":
            y = sample[config.target].to(device)
        elif target_name == "pixel-classification":
            y = sample[config.target].type(torch.LongTensor).to(device)

        y_hat = model(img)

        if target_name == "multi-classification":
            y_hat = sigmoid(y_hat)

        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if target_name == "multi-classification":
            pred = y_hat.round()
        elif target_name == "single-classification":
            _, pred = torch.max(y_hat, dim=1)
        elif target_name == "pixel-classification":
            probas = F.softmax(y_hat, dim=1)
            pred = torch.argmax(probas, axis=1)

        epoch_losses = torch.cat([epoch_losses, loss[None].detach().cpu()])
        metrics.add_batch(y, pred)

        pbar.set_description(f"Epoch:{epoch}, Loss:{epoch_losses[-100:].mean():.4}")

    mean_loss = epoch_losses.mean()

    if target_name == "single-classification":
        train_stats = {
            "train_loss": mean_loss.item(),
            "train_average_accuracy": metrics.get_average_accuracy(),
            "train_overall_accuracy": metrics.get_overall_accuracy(),
            **{
                "train_accuracy_" + k: v
                for k, v in metrics.get_classwise_accuracy().items()
            },
        }

    elif target_name == "multi-classification":
        train_stats = {
            "train_loss": mean_loss.item(),
            "train_average_f1": metrics.get_average_f1(),
            "train_overall_f1": metrics.get_overall_f1(),
            "train_average_recall": metrics.get_average_recall(),
            "train_overall_recall": metrics.get_overall_recall(),
            "train_average_precision": metrics.get_average_precision(),
            "train_overall_precision": metrics.get_overall_precision(),
            **{"train_f1_" + k: v for k, v in metrics.get_classwise_f1().items()},
        }

    elif target_name == "pixel-classification":
        train_stats = {
            "train_loss": mean_loss.item(),
            "train_average_accuracy": metrics.get_average_accuracy(),
            **{
                "train_accuracy_" + k: v
                for k, v in metrics.get_classwise_accuracy().items()
            },
        }
    wandb.log(train_stats, step=step)

    if epoch % 2 == 0:
        val_stats = validate_all(
            model, val_loader, criterion, device, config, model_name, target_name
        )
        print(f"Epoch:{epoch}", val_stats)
        wandb.log(val_stats, step=step)

    if epoch % 200 == 0:
        if epoch == 0:
            continue

        save_weights_path = (
            "checkpoints/" + "-".join([model_name, target_name, str(run.name), "epoch", str(epoch)]) + ".pth"
        )
        save_checkpoint_single_model(
            model, optimizer, val_stats, epoch, save_weights_path
        )
