# -*-coding:utf-8 -*-

"""
# File       : size_test.py
# Time       ：2023/2/15 9:20
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch

x = torch.randn(size=(10, 20, 768))
print(x[:, 0].shape)
