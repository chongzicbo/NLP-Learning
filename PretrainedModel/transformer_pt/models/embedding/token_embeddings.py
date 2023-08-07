# -*-coding:utf-8 -*-

"""
# File       : token_embeddings.py
# Time       ：2023/3/3 15:31
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
