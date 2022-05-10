#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/10
# @file 019卷积层.py
import torch
from torch import nn
from d2l import torch as d2

# k : kernel matrix
def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i + h, j:j + w] * k).sum()
    return y


# x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
# k = torch.arange(4, dtype=torch.float32).reshape(2, 2)
# print(corr2d(x, k))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


x = torch.ones((6, 8))
x[:, 2:6] = 0
# # print(x)
k = torch.tensor([[1.0, -1.0]])
# class parameter transfer
y = corr2d(x, k)
# print(y)

conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
x = x.reshape((1, 1, 6, 8))
y = y.reshape((1, 1, 6, 7))

for i in range(10):
    y_hat = conv2d(x)
    l = (y_hat - y)**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch{i+1}, loss {l.sum():.3f}')
print(conv2d.weight.data)



