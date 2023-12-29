# -*-coding:utf-8 -*-

"""
# File       : token.py
# Time       ：2023/2/13 16:19
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
