#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/4
# @file 004数据操作.py
import torch
x = torch.arange(12)
print(x,x.shape,x.numel())

x = x.reshape(3,4)
print(x)