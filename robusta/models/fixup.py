"""
Reference:
https://github.com/hongyi-zhang/Fixup/blob/master/imagenet/models/fixup_resnet_imagenet.py

---

BSD 3-Clause License

Copyright (c) 2019,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "fixup_resnet18",
    "fixup_resnet34",
    "fixup_resnet50",
    "fixup_resnet101",
    "fixup_resnet152",
]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    pass


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    pass


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        pass


class FixupBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv1x1(inplanes, planes)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        pass


class FixupResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(
                    m.conv1.weight,
                    mean=0,
                    std=np.sqrt(
                        2
                        / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))
                    )
                    * self.num_layers ** (-0.5),
                )
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(
                        m.downsample.weight,
                        mean=0,
                        std=np.sqrt(
                            2
                            / (
                                m.downsample.weight.shape[0]
                                * np.prod(m.downsample.weight.shape[2:])
                            )
                        ),
                    )
            elif isinstance(m, FixupBottleneck):
                nn.init.normal_(
                    m.conv1.weight,
                    mean=0,
                    std=np.sqrt(
                        2
                        / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))
                    )
                    * self.num_layers ** (-0.25),
                )
                nn.init.normal_(
                    m.conv2.weight,
                    mean=0,
                    std=np.sqrt(
                        2
                        / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))
                    )
                    * self.num_layers ** (-0.25),
                )
                nn.init.constant_(m.conv3.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(
                        m.downsample.weight,
                        mean=0,
                        std=np.sqrt(
                            2
                            / (
                                m.downsample.weight.shape[0]
                                * np.prod(m.downsample.weight.shape[2:])
                            )
                        ),
                    )
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        pass

    def forward(self, x):
        pass


def fixup_resnet18(**kwargs):
    """Constructs a Fixup-ResNet-18 model."""
    pass


def fixup_resnet34(**kwargs):
    """Constructs a Fixup-ResNet-34 model."""
    pass


def fixup_resnet50(**kwargs):
    """Constructs a Fixup-ResNet-50 model."""
    pass


def fixup_resnet101(**kwargs):
    """Constructs a Fixup-ResNet-101 model."""
    pass


def fixup_resnet152(**kwargs):
    """Constructs a Fixup-ResNet-152 model."""
    pass
