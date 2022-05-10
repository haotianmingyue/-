#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/10
# @file 016自定义层.py

import torch
import torch.nn.functional as F
from torch import nn


# custom layer
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - x.mean()

# layer = CenteredLayer()
# print(layer(torch.FloatTensor([1, 2, 3, 4])))
#
# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
#
# y = net(torch.rand(4, 8))
# print(y.mean())

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, x):
        linear = torch.matmul(x, self.weight.data) + self.bias.data
        return F.relu(linear)

# dense = MyLinear(5, 3)
# print(dense.weight, dense.bias)



