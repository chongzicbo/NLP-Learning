# -*-coding:utf-8 -*-

"""
# File       : bert.py
# Time       ：2023/2/14 9:39
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch.nn as nn
from .transformer import TransformerBlock
from .embedding.bert import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """

        :param vocab_size:
        :param hidden:
        :param n_layers:
        :param attn_heads:
        :param dropout:
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])  # 12层transformer

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # x=[batch_size,seq_len] => [batch_size,1,seq_len] =>[batch_size,seq_len,seq_len] => [batch_size, 1, seq_len, seq_len]
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_info)  # [batch_size,seq_len,embedding_size]
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)  # [batch_size,seq_len,embedding_size]

        return x
