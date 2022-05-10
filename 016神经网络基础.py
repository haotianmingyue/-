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

# x = torch.rand(2, 20)

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
#
# net = NestMLP()
# print(net(x))


# parameter management

# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
x = torch.rand(size=(2, 4))
# print(net(x))

# for i in range(3):
#     print(net[i].state_dict())

# print(*[(name, param.shape) for name, param in net.named_parameters()])


# collect parameters from nesting block
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4),
                         nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net

# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
#
# print(rgnet)

# init
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
# net.apply(init_normal)

# do different initialization for every block
# net[0].apply(i1)
# net[1].applt(i2)

# share weight, parameter binding, the shared layer's parameter is same
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
# the second layer is equal to the forth layer
print(net[2].weight.data[0] == net[4].weight.data[0])

