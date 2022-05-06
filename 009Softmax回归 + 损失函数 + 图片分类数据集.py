#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/6
# @file 009Softmax回归 + 损失函数 + 图片分类数据集.py

import torch
import torchvision   # computer vision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# Converts a PIL Image or numpy.ndarray (H x W x C) in the range
# [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
trans = transforms.ToTensor()

# train = True ,download the train data, train = False , download the test date
mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

# print(len(mnist_test),len(mnist_train))
# notice that the dimension of tensor image is c*w*h
# print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):
    text_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    # back the text_labels from one_hot code
    # label are the set of 1,2,3,4,5,6,7,8,9 not the one-hot
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    # num_rows num_cols , the row and col of subplot
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()

    # axes.show()
    return axes


# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# print(y)
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

batch_size = 256

def get_dataloader_workers():
    # 4 process to read data
    return 6

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

# timer = d2l.Timer()
# for x, y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec')
def load_data_fashion_mnist(batch_size, resize=None):
    # a list of operation order, first there is a toTensor function
    trans = [transforms.ToTensor()]
    if resize:
        # if resize != None, indicate that it needs resize,so add a resize operation to the list
        trans.insert(0, transforms.Resize(resize))
    # make a operation set to the data
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))




