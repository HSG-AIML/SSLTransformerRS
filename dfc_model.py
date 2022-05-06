# https://github.com/rwindsor1/biobank-self-supervised-alignment/blob/main/src/models/VGGEncoders.py

import sys, os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet50


class DualBaseline(nn.Module):
    def __init__(
        self, model, s1_channels, s2_channels, num_classes=8, feature_dim=1024
    ):
        super(DualBaseline, self).__init__()
        self.model_s1 = eval(model)(pretrained=False, num_classes=num_classes)
        self.model_s1.conv1 = torch.nn.Conv2d(
            s1_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.model_s1.fc = torch.nn.Identity()

        self.model_s2 = eval(model)(pretrained=False, num_classes=num_classes)
        self.model_s2.conv1 = torch.nn.Conv2d(
            s2_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.model_s2.fc = torch.nn.Identity()

        self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        s1 = x["s1"]
        s2 = x["s2"]

        s1_emb = self.model_s1(s1)
        s2_emb = self.model_s2(s2)

        z = torch.cat([s1_emb, s2_emb], dim=1)
        z = self.fc(z)

        return z


class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvSequence, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DoubleAlignmentDownstream(nn.Module):
    """concatenate outputs from two backbones and add one linear layer"""

    def __init__(self, base_model, device, config):
        super(DoubleAlignmentDownstream, self).__init__()
        self.base_model = base_model
        self.backbone1 = eval(base_model)(
            embedding_size=config.embedding_size, input_modes=config.s1_input_channels
        ).to(device)
        self.backbone2 = eval(base_model)(
            embedding_size=config.embedding_size, input_modes=config.s2_input_channels
        ).to(device)

        # dim_mlp1 = self.backbone1.fc.in_features
        # dim_mlp2 = self.backbone2.fc.in_features

        # add final linear layer
        # note: this will produce differently sized vectors z depending on kernel_size, config.embedding_size and config.image_size
        # roughly: z.shape[1] = 2*config.embedding_size * (config.image_px_size/8/kernel_size)**2
        # where 2*config.embedding_size is due to the two inputs
        # and config.image_px_size/8 due to pooling of the VGG encoders

        kernel_size = 8
        # dim_fc = 2*config.embedding_size * (config.image_px_size/8/kernel_size)**2
        dim_fc = 2 * config.embedding_size
        print("FC dim: ", dim_fc)
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=kernel_size).to(device)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(int(dim_fc), config.num_classes, bias=True).to(device)

    def _get_basemodel(self, model_name):
        try:
            model = eval(self.base_model)
        except KeyError:
            raise InvalidBackboneError("Invalid backbone architecture")
        else:
            return model

    def forward(self, x):
        x1 = self.backbone1(x["s1"])
        x2 = self.backbone2(x["s2"])

        # in_map = torch.cat([x1, x2], dim=1)
        # out_map = self.avg_pool(in_map)
        # z = out_map.flatten(start_dim=1, end_dim=-1)

        x1 = self.avg_pool(x1).flatten(start_dim=1, end_dim=-1)
        x2 = self.avg_pool(x2).flatten(start_dim=1, end_dim=-1)

        z = torch.cat([x1, x2], dim=1)
        z = self.fc(z)

        return z

    def load_trained_state_dict(self, checkpoint):
        """load the pre-trained backbone weights"""

        # log = self.load_state_dict(weights, strict=True)
        log_s1 = self.backbone1.load_state_dict(
            checkpoint["s1_model_weights"], strict=True
        )
        log_s2 = self.backbone2.load_state_dict(
            checkpoint["s2_model_weights"], strict=True
        )

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False


class VGGEncoder(nn.Module):
    """Scan encoding model"""

    def __init__(self, embedding_size=512, input_modes=1):
        super(VGGEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.convs1 = ConvSequence(input_modes, 64, kernel_size=(3, 3), padding=1)
        self.convs2 = ConvSequence(64, 128, kernel_size=(3, 3), padding=1)
        self.convs3 = ConvSequence(128, 256, kernel_size=(3, 3), padding=1)
        self.convs4 = ConvSequence(256, embedding_size, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1, kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2, kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        return x4

    def forward_with_skips(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1, kernel_size=2)
        x2 = self.convs2(x2)
        x3 = F.max_pool2d(x2, kernel_size=2)
        x3 = self.convs3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x4 = self.convs4(x4)
        return x4, x3, x2, x1


class VGGEncoderShallow(nn.Module):
    """Scan encoding model"""

    def __init__(self, embedding_size=128, input_modes=1):
        super(VGGEncoderShallow, self).__init__()
        self.embedding_size = embedding_size
        self.convs1 = ConvSequence(input_modes, 64, kernel_size=(3, 3), padding=1)
        self.convs2 = ConvSequence(64, embedding_size, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x1 = self.convs1(x)
        x2 = F.max_pool2d(x1, kernel_size=2)
        x2 = self.convs2(x2)
        return x2


class Classifier(nn.Module):
    """Put a linear head on top of two VGGEncoders"""

    def __init__(self, s1_model, s2_model, embedding_size=128, num_classes=10):
        super(Classifier, self).__init__()
        self.s1_model = s1_model
        self.s2_model = s2_model

        self.fc = torch.nn.Linear(embedding_size * 2, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, s1, s2):
        s1_embedding = self.s1_model(s1)
        s2_embedding = self.s2_model(s2)

        embedding = torch.cat((s1_embedding, s2_embedding), dim=1)
        embedding = self.avgpool(embedding)
        embedding = torch.flatten(embedding, 1)

        output = self.fc(embedding)

        return output


class SegmentorVGG(nn.Module):
    """Small segmentation network on top of the VGGEncoders"""

    def __init__(self, s1_model, s2_model, embedding_size=128, num_classes=10):
        super(Segmentor, self).__init__()
        self.s1_model = s1_model
        self.s2_model = s2_model

        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = torch.nn.Conv2d(
            embedding_size * 2, embedding_size, kernel_size=3, padding=1
        )
        self.bn = torch.nn.BatchNorm2d(embedding_size)
        self.relu = torch.nn.ReLU(inplace=True)

        self.out_conv = torch.nn.Conv2d(embedding_size, num_classes, kernel_size=1)

    def forward(self, s1, s2):
        s1_embedding = self.s1_model(s1)
        s2_embedding = self.s2_model(s2)

        embedding = torch.cat((s1_embedding, s2_embedding), dim=1)

        embedding = self.up(embedding)
        embedding = self.conv(embedding)
        embedding = self.bn(embedding)
        embedding = self.relu(embedding)

        output = self.out_conv(embedding)

        return output


# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FeedbackUp(nn.Module):
    """Upscaling then double conv, return the upsampled to same size inputs too"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x), x1, x2


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class FeedbackUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FeedbackUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = FeedbackUp(1024, 512 // factor, bilinear)
        self.up2 = FeedbackUp(512, 256 // factor, bilinear)
        self.up3 = FeedbackUp(256, 128 // factor, bilinear)
        self.up4 = FeedbackUp(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x_in):
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x, dec1, enc1 = self.up1(x5, x4)
        x, dec2, enc2 = self.up2(x, x3)
        x, dec3, enc3 = self.up3(x, x2)
        x, dec4, enc4 = self.up4(x, x1)
        logits = self.outc(x)
        return {
            "logits": logits,
            # "dec0" : torch.max(logits, dim=1)[1], "enc0" : x_in,
            "dec1": dec1,
            "enc1": enc1,
            "dec2": dec2,
            "enc2": enc2,
            "dec3": dec3,
            "enc3": enc3,
            "dec4": dec4,
            "enc4": enc4,
        }


class UNetEncoder(nn.Module):
    """Encoder part of a UNet"""

    def __init__(self, n_channels, bilinear=True):
        super(UNetEncoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1

        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5


class DoubleUp(nn.Module):
    """Upscaling then double conv based on two UNet encoders"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, z, x2, y2):
        z = self.up(z)

        assert x2.shape == y2.shape

        # input is CHW
        diffY = x2.size()[2] - z.size()[2]
        diffX = x2.size()[3] - z.size()[3]

        z = F.pad(z, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        z = torch.cat([z, x2, y2], dim=1)
        return self.conv(z)


class DoubleUNet(nn.Module):
    """UNet with two separate encoders for different modalities
    maps are concatenated at every level"""

    def __init__(self, n_channels1, n_channels2, n_classes, bilinear=True):
        super(DoubleUNet, self).__init__()
        self.n_channels1 = n_channels1
        self.n_channels2 = n_channels2
        self.n_classes = n_classes
        self.bilinear = bilinear

        # unet1 = UNet(n_channels1, n_classes)
        # unet2 = UNet(n_channels2, n_classes)

        # self.encoder1 = torch.nn.Sequential(*[*list(unet1.inc.children()), *list(unet1.down1.children()), *list(unet1.down2.children()), *list(unet1.down3.children()), *list(unet1.down4.children())])
        # self.encoder2 = torch.nn.Sequential(*[*list(unet2.inc.children()), *list(unet2.down1.children()), *list(unet2.down2.children()), *list(unet2.down3.children()), *list(unet2.down4.children())])

        self.encoder1 = UNetEncoder(n_channels1, bilinear)
        self.encoder2 = UNetEncoder(n_channels2, bilinear)

        factor = 2 if bilinear else 1
        self.up1 = DoubleUp(2048, 1024 // factor, bilinear)
        self.up2 = DoubleUp(1024, 512 // factor, bilinear)
        self.up3 = DoubleUp(512, 256 // factor, bilinear)
        self.up4 = DoubleUp(256, 128, bilinear)
        self.outc = OutConv(128, n_classes)

    def forward(self, x, y):
        x1, x2, x3, x4, x5 = self.encoder1(x)
        y1, y2, y3, y4, y5 = self.encoder2(y)

        # merge representations at lowest layer
        z = torch.cat([x5, y5], dim=1)

        z = self.up1(z, x4, y4)
        z = self.up2(z, x3, y3)
        z = self.up3(z, x2, y2)
        z = self.up4(z, x1, y1)
        logits = self.outc(z)
        return logits
