import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU, ConvTranspose2dReLU
from ..base.model import Model


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        return self.block(x)


class ConvUpsampleBlock(ConvBlock):

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class ConvTransposeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            ConvTranspose2dReLU(out_channels, out_channels, kernel_size=4, padding=1,
                                stride=2, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = self.block(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True, type="upsample"):
        super().__init__()

        if type == "upsample":
            self.block = ConvUpsampleBlock(in_channels, out_channels, use_batchnorm)

        elif type == "transpose":
            self.block = ConvTransposeBlock(in_channels, out_channels, use_batchnorm)

        else:
            raise ValueError("Supported block types: `upsample`, `transpose`")

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            type="upsample",
    ):
        super().__init__()

        if type not in ("upsample", "transpose"):
            raise ValueError("Supported block types: `upsample`, `transpose`")

        if center:
            channels = encoder_channels[0]
            self.center = ConvBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels, type=type)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm, type=type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm, type=type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm, type=type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm, type=type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm, type=type)

        if type == "transpose":
            self.final_block = ConvBlock(out_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
        else:
            self.final_block = None

        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels, type="upsample"):

        if type == "upsample":
            channels = [
                encoder_channels[0] + encoder_channels[1],
                encoder_channels[2] + decoder_channels[0],
                encoder_channels[3] + decoder_channels[1],
                encoder_channels[4] + decoder_channels[2],
                0 + decoder_channels[3],
            ]
        else:
            channels = [
                encoder_channels[0],
                encoder_channels[1] + decoder_channels[0],
                encoder_channels[2] + decoder_channels[1],
                encoder_channels[3] + decoder_channels[2],
                encoder_channels[4] + decoder_channels[3],
            ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])

        if self.final_block:
            x = self.final_block(x)

        x = self.final_conv(x)

        return x
