# -*-coding:utf-8 -*-

"""
# File       : position.py
# Time       ：2023/2/13 16:28
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2 表示从0开始，每隔2个
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


if __name__ == '__main__':
    d_model = 5
    max_len = 10

    positionEmbedding = PositionalEmbedding(d_model, max_len)
