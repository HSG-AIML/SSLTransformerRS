# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from functools import partial

# from math import remainder

# from timm.models import vit_deit_small_patch16_224
# from timm.models import vit_deit_small_patch16_224
from timm.models import vit_small_patch16_224 as vit_deit_small_patch16_224

from .swin_transformer import SwinTransformer, SharedDSwin
from .moby import MoBY
import torch

vit_models = dict(
    deit_small=vit_deit_small_patch16_224,
)


def build_model(config):
    model_type = config.MODEL.TYPE
    encoder_type = config.MODEL.MOBY.ENCODER

    if encoder_type == "swin":
        enc = partial(
            SwinTransformer,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
        )
    elif encoder_type.startswith("vit") or encoder_type.startswith("deit"):
        enc = vit_models[encoder_type]
    else:
        raise NotImplementedError(f"--> Unknown encoder_type: {encoder_type}")

    if model_type == "moby":
        encoder = enc(
            num_classes=0,
            drop_path_rate=config.MODEL.MOBY.ONLINE_DROP_PATH_RATE,
        )
        encoder_k = enc(
            num_classes=0,
            drop_path_rate=config.MODEL.MOBY.TARGET_DROP_PATH_RATE,
        )
        model = MoBY(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            contrast_momentum=config.MODEL.MOBY.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.MOBY.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.MOBY.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.MOBY.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.MOBY.PRED_NUM_LAYERS,
        )
    elif model_type == "linear":
        model = enc(
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == "d-swin":
        model = enc(
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == "shared-d-swin":
        """a d-swin trained swin transformer where blocks 2 and beyond are shared across views"""
        block1_s1 = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.DATA.S1_INPUT_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
        block1_s2 = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.DATA.S2_INPUT_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
        shared = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.DATA.S2_INPUT_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )

        s1_encoder = get_block1_encoder(block1_s1)
        s2_encoder = get_block1_encoder(block1_s2)
        shared_backbone = get_model_shared_backbone(shared)

        model = SharedDSwin(s1_encoder, s2_encoder, shared_backbone)

    else:
        raise NotImplementedError(f"--> Unknown model_type: {model_type}")

    return model


# utility functions to construct the shared double t-swin


def forward_block1(self, x):
    x = self.patch_embed(x)
    if self.ape:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)

    for layer in self.layers:
        x = layer(x)
    # x = self.layers[0](x)

    return x


def forward_shared_features(self, x):
    for layer in self.layers:
        x = layer(x)

    x = self.norm(x)  # B L C
    x = self.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    return x


def get_block1_encoder(t_swin_model):
    del t_swin_model.layers[-1]
    del t_swin_model.layers[-1]
    del t_swin_model.layers[-1]
    del t_swin_model.norm
    del t_swin_model.avgpool
    del t_swin_model.head

    t_swin_model.forward = partial(forward_block1, t_swin_model)

    return t_swin_model


def get_model_shared_backbone(t_swin_model):
    # del t_swin_model.patch_embed

    del t_swin_model.pos_drop
    del t_swin_model.layers[0]
    del t_swin_model.head

    t_swin_model.forward_features = partial(forward_shared_features, t_swin_model)

    return t_swin_model
