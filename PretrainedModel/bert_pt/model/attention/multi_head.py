# -*-coding:utf-8 -*-

"""
# File       : multi_head.py
# Time       ：2023/2/13 20:59
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch.nn as nn

from .single import Attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h  # 每个head的维度
        self.h = h  # head数量
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # q,k,v:[batch_size,seq_len,embedding_size]
        batch_size = query.size(0)
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]  # 对q,k,v进行线性变换,分割成多个头[batch_size,d_model]=>[batch_size,h,-1,d_k]

        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )  # attn:[batch_size,heads,seq_len,seq_len]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)
