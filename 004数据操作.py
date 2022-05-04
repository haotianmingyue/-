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

x_1 = torch.zeros((2,3,4))
print(x_1)
x_2 = torch.tensor([[1,2,3],[3,2,1]])
print(x_2)


x_3 = torch.tensor([1.0,2,3,4]) #float
x_4 = torch.tensor([2,2,2,2])
#按元素 做操作
print(x_3+x_4,x_3-x_4,x_3*x_4,x_3/x_4)

#指数运算
print(torch.exp(x_3))

a = torch.arange(12,dtype=torch.float32).reshape((3,4))
b = torch.tensor([[1,2,3,4],[2,3,4,5],[3,4,5,6]])

#在哪一维上合并
x_5 = torch.cat((a,b),dim = 0)
x_6 = torch.cat((a,b),dim = 1)
print(x_5,x_6)

#逻辑运算符构建二元张量
#按元素比较
print(a == b)

#求和
print(a.sum())

#enough shape is not same. we still can get element-wise product by broadcasting mechanism
a_1 = torch.arange(3).reshape((3,1))
a_2 = torch.arange(2).reshape((1,2))
#operation process   make a_1's dimension to (3,2) by copy itself and make a_2;dimension to (3,2
print(a_1+a_2)

#query element [-1] select the final element of first dimension
# [1:3] get the first and second element of first dimension
# but [-1] back the result is less a dimension and [1:3] is not

print(b[-1])
print(b[1:3])

# wirite b[1,2] = 9

# found similar matrix
a_3 =  torch.zeros_like(a_2)

print(a_3)

# convert to numpy

A = a_3.numpy()
B = torch.tensor(A)

print(type(A),type(B))

# convert tensor to scalar
c = torch.tensor([3.5])
print(c,c.item(),float(c),int(c))
