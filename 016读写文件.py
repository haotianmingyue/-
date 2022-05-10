#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/10
# @file 016读写文件.py
import torch
from torch import nn
from torch.nn import functional as F

# x = torch.arange(4)
# # torch.save(x, 'x-file')
#
# # x2 = torch.load("x-file")
# # print(x2)
# y = torch.arange(4)
#
# torch.save([x, y],'x-file')
#
# # when the number of parameters is equal to the list's length, return every values in list not the total list
# # x2, y2 = torch.load('x-file')
# # print(x2, '\n', y2)
# # b = torch.load('x-file')
# # you also can use * to collect parameters
# # l = [1, 2, 3, 4]
# # a, b, *c = l
# # print(a, b, c)
#
# # print(b)

# load or save model paramters

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))
#
net = MLP()
x = torch.randn(size=(2, 20))
y = net(x)
#
# # save weight
torch.save(net.state_dict(), 'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

y_clone = clone(x)
print(y_clone == y)
