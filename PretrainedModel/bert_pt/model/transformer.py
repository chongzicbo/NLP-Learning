# -*-coding:utf-8 -*-

"""
# File       : transformer.py
# Time       ：2023/2/13 21:20
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

import torch.nn as nn
from .attention.multi_head import MultiHeadedAttention
from .utils.sublayer import SublayerConnection
from .utils.feed_forward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """

        :param hidden:
        :param attn_heads:
        :param feed_forward_hidden:
        :param dropout:
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)

        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # x[batch_size,seq_len,embedding_size]
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask))  # 输入经过attention层后，再进行残差计算
        x = self.output_sublayer(x, self.feed_forward)  # 对残差后的结果进过前馈层后，进行残差
        return self.dropout(x)  # 残差结果再进行dropout
