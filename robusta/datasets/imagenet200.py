# Copyright 2020-2021 Evgenia Rusak, Steffen Schneider, George Pachitariu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

""" ImageNet-R

Reference: https://github.com/hendrycks/imagenet-r

Originally licensed under

MIT License

Copyright (c) 2020 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import shutil
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from robusta.datasets.base import RemappedImageNet


class ImageNet200(RemappedImageNet):
    """Subset of ImageNet with 200 classes

    Reference: https://github.com/hendrycks/imagenet-r
    """

    def get_class_mask(self):
        pass

    def create_symlinks_to_imagenet(self):
        pass

    def __init__(
        self, imagenet_directory, imagenet_200_directory="/tmp/in200", transform=None
    ):
        self.imagenet_1k_location = imagenet_directory
        self.imagenet_200_location = imagenet_200_directory
        self.create_symlinks_to_imagenet()
        super().__init__(self.imagenet_200_location, transform=transform)
