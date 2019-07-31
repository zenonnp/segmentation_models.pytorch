import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU, ConvTranspose2dReLU
from ..base.model import Model


def pad_zeros(x, length):
    x = list(x)
    for _ in range(length - len(x)):
        x.append(0)
    return x


def pad_none(x, length):
    x = list(x)
    for _ in range(length - len(x)):
        x.append(None)
    return x


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        return self.block(x)


class ConvUpsampleX2Block(nn.Module):

    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvTransposeX2Block(nn.Module):

    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()

        self.conv1 = ConvTranspose2dReLU(
            in_channels,
            in_channels,
            kernel_size=4,
            padding=1,
            stride=2,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = self.conv1(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            block_type="upsample",
            num_blocks=5,
    ):
        super().__init__()

        self.num_blocks = num_blocks

        if block_type == "upsample":
            DecoderBlock = ConvUpsampleX2Block
        elif block_type == "transpose":
            DecoderBlock = ConvTransposeX2Block
        else:
            raise ValueError("Supported block types: `upsample`, `transpose`")

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = encoder_channels[:1] + decoder_channels[:-1]
        skip_channels = pad_zeros(encoder_channels[1:], length=num_blocks)
        out_channels = decoder_channels

        self.blocks = nn.ModuleList(
            [DecoderBlock(in_channels[i], out_channels[i], skip_channels[i], use_batchnorm)
             for i in range(num_blocks)]
        )

        self.final_conv = nn.Conv2d(out_channels[-1], final_channels, kernel_size=(1, 1))

        self.initialize()

    def forward(self, x, skips):

        if self.center:
            x = self.center(x)

        skips = pad_none(skips, length=self.num_blocks)  # make None padding ([0, 1, 2] -> [0, 1, 2, None, None])
        for i in range(self.num_blocks):
            x = self.blocks[i](x, skip=skips[i])

        x = self.final_conv(x)

        return x
