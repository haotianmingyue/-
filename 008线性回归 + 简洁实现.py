#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/5
# @file 008线性回归 + 简洁实现.py
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train ) # shuffle : Is it out of order

batch_size = 10
data_iter = load_array((features,labels),batch_size)

# nn : neural network
from torch import nn

# nn.Sequential is a order container
net = nn.Sequential(nn.Linear(2, 1))  # nn.Linear(2, 1),linear layer, 2 input dimensional , 1 output dimensional

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# square loss
loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for x, y in data_iter:
        l = loss(net(x), y)
        trainer.zero_grad()
        l.backward()
        trainer.step() #update gradient
    # print(labels.__class__)
    # break
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1},loss {l:f}') # :f keep six float

# print(next(iter(data_iter)))