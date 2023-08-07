# -*-coding:utf-8 -*-

"""
# File       : single.py
# Time       ：2023/2/13 20:49
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

import torch.nn as nn
import torch.nn.functional as F
import torch, math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # (q * k^T)/d^(1/2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e-9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
