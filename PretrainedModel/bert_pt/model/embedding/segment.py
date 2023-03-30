# -*-coding:utf-8 -*-

"""
# File       : segment.py
# Time       ：2023/2/13 16:24
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


