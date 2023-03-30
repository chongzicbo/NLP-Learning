# -*-coding:utf-8 -*-

"""
# File       : layer_norm.py
# Time       ：2023/2/24 16:43
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 均值
        var = x.var(-1, unbiased=False, keepdim=True)  # 方差
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta  # gamma 和beta可训练
        return out


if __name__ == '__main__':
    x = torch.randn(size=(3, 4))
    print(x)
    print(torch.mean(x, -1))
