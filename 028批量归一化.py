# 开发者 haotian
# 开发时间: 2022/5/26 20:05
'''
following layer's train speed is faster than the forward one
'''
import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(x, gamma, beta, moving_mean, moving_var, eps, momentum):
    # using is_grad_enabled to determine the model is train or test
    if not torch.is_grad_enabled():
        # if the model is test, your must directly use the moving_mean and the moving_var
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # judge the model is fc or conv
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            # fc model
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
        else:
            # conv model
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim = True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        # test ??
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    y = gamma * x_hat + beta
    return y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
            # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        # 保存更新过的moving_mean和moving_var
        y, self.moving_mean, self.moving_var = batch_norm(
            x, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return y


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())