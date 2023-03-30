# -*-coding:utf-8 -*-

"""
# File       : layer_norm.py
# Time       ：2023/2/13 20:40
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        # x[batch_size,seq_len,embedding_size]
        mean = x.mean(-1, keepdim=True)  # x[batch_size,seq_len,1]
        std = x.std(-1, keepdim=True)  # x[batch_size,seq_len,1]
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 ## x[batch_size,seq_len,1]
