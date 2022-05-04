#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/5/4
# @file 004数据预处理.py
import os

# found a dataset and save it in a csv file
import torch

os.makedirs(os.path.join('..','data'),exist_ok=True)
# .. back to upper level directory
data_file = os.path.join('..','data','house_tiny.csv')
print(data_file)
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')   # NA is a missing value
    f.write('2,NA,106000\n')
    f.write('3,NA,178100\n')
    f.write('NA,NA,140000\n')
import pandas as pd
data = pd.read_csv(data_file)
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

inputs = inputs.fillna(inputs.mean())

inputs = pd.get_dummies(inputs,dummy_na=True)

print(inputs)
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)

print(x, y)
