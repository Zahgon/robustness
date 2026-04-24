# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        pass


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    pass


def conv1x1(cin, cout, stride=1, bias=False):
    pass


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    pass


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        pass

    def load_from(self, weights, prefix=""):
        pass


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3, 64 * wf, kernel_size=7, stride=2, padding=3, bias=False
                        ),
                    ),
                    ("pad", nn.ConstantPad2d(1, 0)),
                    ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
                    # The following is subtly not the same!
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=64 * wf, cout=256 * wf, cmid=64 * wf
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=256 * wf, cout=256 * wf, cmid=64 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=256 * wf,
                                            cout=512 * wf,
                                            cmid=128 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=512 * wf, cout=512 * wf, cmid=128 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=512 * wf,
                                            cout=1024 * wf,
                                            cmid=256 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=1024 * wf, cout=1024 * wf, cmid=256 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit01",
                                        PreActBottleneck(
                                            cin=1024 * wf,
                                            cout=2048 * wf,
                                            cmid=512 * wf,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:02d}",
                                        PreActBottleneck(
                                            cin=2048 * wf, cout=2048 * wf, cmid=512 * wf
                                        ),
                                    )
                                    for i in range(2, block_units[3] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )
        # pylint: enable=line-too-long

        self.zero_head = zero_head
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("gn", nn.GroupNorm(32, 2048 * wf)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("avg", nn.AdaptiveAvgPool2d(output_size=1)),
                    ("conv", nn.Conv2d(2048 * wf, head_size, kernel_size=1, bias=True)),
                ]
            )
        )

    def forward(self, x):
        pass

    def load_from(self, weights, prefix="resnet/"):
        pass


KNOWN_MODELS = OrderedDict(
    [
        ("BiT-M-R50x1", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        ("BiT-M-R50x3", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        ("BiT-M-R101x1", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        ("BiT-M-R101x3", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        ("BiT-M-R152x2", lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
        ("BiT-M-R152x4", lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
        ("BiT-S-R50x1", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
        ("BiT-S-R50x3", lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
        ("BiT-S-R101x1", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
        ("BiT-S-R101x3", lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
        ("BiT-S-R152x2", lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
        ("BiT-S-R152x4", lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ]
)
