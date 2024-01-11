# -*-coding:utf-8 -*-

"""
# File       : sublayer.py
# Time       ：2023/2/13 20:44
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from .layer_norm import LayerNorm
import torch.nn as nn


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x:[batch_size,seq_len,emb_size]残差层：输入x经过norm，结果传入指定层，再接dropout。dropout的结果与输入x相加
        return x + self.dropout(sublayer(self.norm(x)))
