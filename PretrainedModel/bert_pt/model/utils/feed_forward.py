# -*-coding:utf-8 -*-

"""
# File       : feed_forward.py
# Time       ：2023/2/13 20:28
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch.nn as nn
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


