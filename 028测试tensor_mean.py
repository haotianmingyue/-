# 开发者 haotian
# 开发时间: 2022/5/26 20:47
import torch


# x = torch.arange(64, dtype=torch.float32).resize(4, 4, 4)
# x_mean = x.mean(dim=(1, 2), keepdim=True)
# print(x_mean, '\n', x_mean.shape)
y = torch.arange(16, dtype=torch.float32).reshape((2, 2, 2, 2))
y_mean = y.mean(dim=(0, 2, 3), keepdim=True)
y_mean_1 = y.mean(dim=1, keepdim=True)
print(y, '\n', y_mean.shape, '\n', y_mean, '\n', y_mean_1)