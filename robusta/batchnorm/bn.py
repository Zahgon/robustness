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

""" Batch norm variants
"""

import torch
from torch import nn
from torch.nn import functional as F


def adapt_ema(model: nn.Module):
    pass


def adapt_parts(model: nn.Module, adapt_mean: bool, adapt_var: bool):
    pass


def adapt_bayesian(model: nn.Module, prior: float):
    pass


class PartlyAdaptiveBN(nn.Module):
    @staticmethod
    def find_bns(parent, estimate_mean, estimate_var):
        pass

    @staticmethod
    def adapt_model(model, adapt_mean, adapt_var):
        pass

    def __init__(self, layer, estimate_mean=True, estimate_var=True):
        super().__init__()
        self.layer = layer

        self.estimate_mean = estimate_mean
        self.estimate_var = estimate_var

        self.register_buffer("source_mean", layer.running_mean.data)
        self.register_buffer("source_var", layer.running_var.data)

        self.register_buffer(
            "estimated_mean",
            torch.zeros(layer.running_mean.size(), device=layer.running_mean.device),
        )
        self.register_buffer(
            "estimated_var",
            torch.ones(layer.running_var.size(), device=layer.running_mean.device),
        )

    def reset(self):
        pass

    @property
    def running_mean(self):
        pass

    @property
    def running_var(self):
        pass

    def forward(self, input):
        # Estimate training set statistics
        pass


class EMABatchNorm(nn.Module):
    @staticmethod
    def reset_stats(module):
        pass

    @staticmethod
    def find_bns(parent):
        pass

    @staticmethod
    def adapt_model(model):
        pass

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # store statistics, but discard result
        pass


class BayesianBatchNorm(nn.Module):
    """Use the source statistics as a prior on the target statistics"""

    @staticmethod
    def find_bns(parent, prior):
        pass

    @staticmethod
    def adapt_model(model, prior):
        pass

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0)

        self.prior = prior

    def forward(self, input):
        pass
