# -*-coding:utf-8 -*-

"""
# File       : lossfunction_test.py
# Time       ：2023/2/14 11:36
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch

input = torch.randn(3, 3)
print(input)

sm = torch.nn.Softmax(dim=1)
print(sm(input))
log_value = torch.log(sm(input))
print(log_value)

print(-(log_value[0][0] + log_value[1][2] + log_value[2][1]) / 3)

loss = torch.nn.NLLLoss()
target = torch.tensor([0, 2, 1])

print(loss(log_value, target))

loss = torch.nn.CrossEntropyLoss()
print(loss(input, target))
