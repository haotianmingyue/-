# 开发者 haotian
# 开发时间: 2022/5/31 14:13
from d2l import torch as d2l
import torch
from torch import nn


def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad()
    return new_params

new_params = get_params(params, d2l.torch.try_gpu(0))

def allreduce(data):
    for i in range(1, len(data)):
        # 逐元素加
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)


data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)

def train_batch(x, y, device_params, devices, lr):
    x_shards, y_shards = split_batch(x, y, devices)
    ls = [loss(lenet(
        x_shard, device_w), y_shard).sum() for  x_shard, y_shard, device_w in zip(
        x_shardsm y_shards, device_params)]

    for l in ls:
        l.backward()
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad
                       for c in range(len(devices))])
    for param in device_params:
        d2l.sgd(param, lr, x.shape[0])





