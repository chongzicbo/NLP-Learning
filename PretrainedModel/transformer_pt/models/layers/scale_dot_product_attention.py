# -*-coding:utf-8 -*-

"""
# File       : scale_dot_product_attention.py
# Time       ：2023/2/24 16:55
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import math
import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2.apply masking(opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3 softmax
        score = self.softmax(score)

        # 4.multiply with Value
        v = score @ v
        return v, score

if __name__ == '__main__':
    a=torch.randint()