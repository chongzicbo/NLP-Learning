# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: model.py
@time: 2022/6/21 21:26
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SentenceTransformer(nn.Layer):

    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"] * 3, 2)

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):
        query_token_embedding, _ = self.ptm(query_input_ids,
                                            query_token_type_ids,
                                            query_position_ids,
                                            query_attention_mask)
        query_token_embedding = self.dropout(query_token_embedding)
        query_attention_mask = paddle.unsqueeze(
            (query_input_ids != self.ptm.pad_token_id).astype(
                self.ptm.pooler.dense.weight.dtype),
            axis=2)
        # Set token embeddings to 0 for padding tokens
        query_token_embedding = query_token_embedding * query_attention_mask
        query_sum_embedding = paddle.sum(query_token_embedding, axis=1)
        query_sum_mask = paddle.sum(query_attention_mask, axis=1)
        query_mean = query_sum_embedding / query_sum_mask

        title_token_embedding, _ = self.ptm(title_input_ids,
                                            title_token_type_ids,
                                            title_position_ids,
                                            title_attention_mask)
        title_token_embedding = self.dropout(title_token_embedding)
        title_attention_mask = paddle.unsqueeze(
            (title_input_ids != self.ptm.pad_token_id).astype(
                self.ptm.pooler.dense.weight.dtype),
            axis=2)
        # Set token embeddings to 0 for padding tokens
        title_token_embedding = title_token_embedding * title_attention_mask
        title_sum_embedding = paddle.sum(title_token_embedding, axis=1)
        title_sum_mask = paddle.sum(title_attention_mask, axis=1)
        title_mean = title_sum_embedding / title_sum_mask

        sub = paddle.abs(paddle.subtract(query_mean, title_mean))
        projection = paddle.concat([query_mean, title_mean, sub], axis=-1)

        logits = self.classifier(projection)

        return logits
