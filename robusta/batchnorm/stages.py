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

""" Helper functions for stages ablations """

import torchvision
from torch import nn


def split_model(model):
    pass


def use_train_statistics(module):
    pass


def choose_one_adaptation(model, stage):
    """select exactly on stage for adaptation"""
    pass


def leave_one_out_adaptation(model, stage):
    """set all BN layers to train mode except for ones in the selected stage"""
    pass
