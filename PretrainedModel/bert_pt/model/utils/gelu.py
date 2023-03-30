# -*-coding:utf-8 -*-

"""
# File       : gelu.py
# Time       ：2023/2/13 20:29
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
