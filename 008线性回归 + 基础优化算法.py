#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/4
# @file 008线性回归 + 基础优化算法.py
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w))) # mean, std , size()
    print(x.shape,w.shape)
    y = torch.matmul(x, w) + b
    print(y.shape)
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

# for x, y  in data_iter(batch_size, features, labels):
#     print(x, '\n', y)
#     break

# requires_grad
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(x, w, b):
    return torch.matmul(x, w) + b

#loss
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad(): # in this section, every element do not autograd
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() # gradient cleaning. every element needn't the last gradient need this

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size,features,labels):
        l = loss(net(x, w, b),y)
        l.sum().backward()
        sgd([w,b], lr, batch_size) # update gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels) # features are total
        print(f'epocc {epoch + 1}, loss {float(train_l.sum()/1000)}')


