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
import logging
from collections import OrderedDict  # pylint: disable=g-importing-member
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear
from torchvision import models

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        return x


class BiResNet(nn.Module):
    def __init__(self, num_classes):
        super(BiResNet, self).__init__()
        self.num_classes = num_classes

        self.branch1 = models.resnet18(weights=None).cuda()
        self.branch1.fc = nn.Identity()  # Remove classification head

        self.branch2 = models.resnet18(weights=None).cuda()
        self.branch2.fc = nn.Identity()  # Remove classification head

        self.head = Linear(512 * 2, num_classes)
        self.cls_loss = CrossEntropyLoss()

    def forward(self, x1, x2, labels=None):
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        x = torch.cat((x1, x2), dim=-1)
        logits = self.head(x)

        if labels is not None:
            loss = self.cls_loss(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, logits
        else:
            return logits

    def load_from(self, weights):
        pass


class UniResNet(nn.Module):
    def __init__(self, num_classes):
        super(UniResNet, self).__init__()
        self.num_classes = num_classes

        self.model = models.resnet18(weights=None).cuda()
        self.model.fc = Linear(512, num_classes)
        self.cls_loss = CrossEntropyLoss()

    def forward(self, x, labels=None):
        logits = self.model(x)

        if labels is not None:
            loss = self.cls_loss(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, logits
        else:
            return logits

    def load_from(self, weights):
        pass


class UniVGGNet(nn.Module):
    def __init__(self, num_classes):
        super(UniVGGNet, self).__init__()
        self.num_classes = num_classes

        self.model = models.vgg11(weights=None).cuda()
        self.model.classifier[6] = Linear(4096, num_classes)
        self.cls_loss = CrossEntropyLoss()

    def forward(self, x, labels=None):
        logits = self.model(x)

        if labels is not None:
            loss = self.cls_loss(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, logits
        else:
            return logits

    def load_from(self, weights):
        pass
