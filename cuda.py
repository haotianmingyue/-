# 开发者 haotian
# 开发时间: 2022/5/23 18:13
import torch
import torch.nn as nn
print(torch.__version__)  #注意是双下划线
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

# print(nn.Parameter(torch.rand((4, 4, 4, 4))) )* nn.Parameter(torch.zeros((1, 4, 1, 1)), requires_grad=True)