#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/17
# @file 025VGG.py
'''
VGG block 3*3 conv  padding = 1
2*2 maxpool s = 2
'''
import torch
from torch import nn
from d2l import torch as d2l


conv_arch = ((1, 64), (1, 128), (2, 256), (2,512), (2,512))
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []  # generate iter object
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
from common_class.train_ch6 import train_ch6
net = vgg(conv_arch=conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# x = torch.randn(size=(1, 1, 224, 224))
#
# for blk in net:
#     x = blk(x)
#     print(blk.__class__.__name__, 'output shape:\t', x.shape)
