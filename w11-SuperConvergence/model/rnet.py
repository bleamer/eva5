import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock

class CustomResModel(nn.Module):

    def __init__(self, block, resnet_block):
        super(CustomResModel, self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block_layers = nn.Sequential(
            block(64, 128, resnet_block=resnet_block),
            block(128, 256),
            block(256, 512, resnet_block=resnet_block)
        )

        self.pool = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.block_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, n_in_channel, n_out_channel, resnet_block=None):
        super(ResBlock, self).__init__()

        self.layer = self.new_layer(n_in_channel, n_out_channel)
        self.resnet_block = None
        if resnet_block:
            self.resnet_block = nn.Sequential(
                resnet_block(n_out_channel, n_out_channel)
            )

    def new_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer(x)
        if self.resnet_block:
            x = x + self.resnet_block(x)
        return x



def ResModel():
    return CustomResModel(ResBlock, BasicBlock)
