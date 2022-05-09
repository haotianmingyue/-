#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/9
# @file 016神经网络基础.py
import torch
from torch import nn
from torch.nn import functional as F

# set up rand_seed to make the result same/identically in every running
# torch.manual_seed(1000)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))

x = torch.rand(2, 20)

# net = MLP()
# print(net(x))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # *args collect parameters
        for block in args:
            # dict
            self._modules[block] = block

        # print(self._modules)
    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# print(net(x))

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # don't participate in training
        self.rand_weight = torch.rand((20,20), requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self, x):
        x = self.linear(x)
        # mm: matrix multiplication
        x = F.relu(torch.mm(x, self.rand_weight) + 1)
        x = self.linear(x)

        while x.abs().sum() > 1:
            x /= 2
        return x.sum()
# net = FixedHiddenMLP()
# print(net(x))

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())

        self.linear = nn.Linear(32, 16)

    def forward(self, x):
        return self.linear(self.net(x))

net = NestMLP()
print(net(x))
