"""
28 x 28 --> 14 x 14 --> 7 x 7
3 downsample blocks and 3 upsample blocks
"""
import os
import sys
from collections import OrderedDict

import wandb
import torch
import torch.nn as nn

class UnetAutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, init_features):
        super(UnetAutoEncoder, self).__init__()

        features = init_features
        self.encoder1 = UnetAutoEncoder._block(in_channels, features, "encoder1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UnetAutoEncoder._block(features, features*2, "encoder2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UnetAutoEncoder._block(features*2, features*4, "encoder2")

        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = UnetAutoEncoder._block((features * 2) * 2, features * 2, "decoder2")
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = UnetAutoEncoder._block(features * 2, features, "decoder2")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))

        cat2 = torch.concat((enc2, self.upconv2(bottleneck)), dim=1)
        dec2 = self.decoder2(cat2)
        cat1 = torch.concat((enc1, self.upconv1(dec2)), dim=1)
        dec1 = self.decoder1(cat1)

        out = self.conv(dec1)

        return out
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([
            (
                name + "conv1",
                nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False)
            ),
            (
                name + "norm1",
                nn.BatchNorm2d(num_features=features)
            ),
            (
                name + "relu1",
                nn.ReLU(inplace=True)
            ),
            (
                name + "conv2",
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
            ),
            (
                name + "norm2",
                nn.BatchNorm2d(num_features=features)
            ),
            (
                name + "relu2",
                nn.ReLU(inplace=True)
            ),
        ]))

