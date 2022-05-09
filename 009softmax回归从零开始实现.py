#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/6
# @file 009softmax回归从零开始实现.py
import torch
from  IPython import  display
from d2l import  torch as d2l

batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# for x, y in train_iter:

num_inputs = 784
num_outputs = 10

w = torch.normal(0, 0.01, (num_inputs, num_outputs))
b = torch.zeros(num_outputs)

def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition

def net(x):
    return softmax(torch.matmul(x.reshape((-1, w.shape[0])), w) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


