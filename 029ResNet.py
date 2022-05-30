# 开发者 haotian
# 开发时间: 2022/5/30 16:17
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=(3, 3),
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=(3, 3), padding=1)
        if use_1x1conv:
            # 若通道数增加了，匹配增加的通道，针对输入输出通道不一致的情况
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=(1, 1), stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        # element-wise add
        Y += x
        return F.relu(Y)


# blk = Residual(3, 6, use_1x1conv=True, strides=2)
# x = torch.rand(4, 3, 6, 6)
# y = blk(x)
# print(y.shape)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 主要是调整 输入和输出通道不匹配的问题
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# 这里 first_block=True 是因为 输入输出都是64，没必要第一个块用 1*1 扩展通道
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))

# x = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     x = layer(x)
#     print(layer.__class__.__name__, 'output shape:\t', x.shape)

lr, num_epochs, batch_size = 0.05, 10, 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
