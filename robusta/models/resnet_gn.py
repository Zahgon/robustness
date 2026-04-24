import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ["ResNet", "resnet50", "resnet101", "resnet152"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    pass


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        gn_init(self.bn1)
        gn_init(self.bn2)
        gn_init(self.bn3, zero_init=True)

    def forward(self, x):
        pass


def conv2d_init(m):
    pass


def gn_init(m, zero_init=False):
    pass


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_init(m)
        gn_init(self.bn1)

    def _make_layer(self, block, planes, blocks, stride=1):
        pass

    def forward(self, x):
        pass


def resnet50(**kwargs):
    pass


def resnet101(**kwargs):
    pass


def resnet152(**kwargs):
    pass
