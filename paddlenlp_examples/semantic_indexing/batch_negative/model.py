# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: model.py
@time: 2022/6/25 17:01
"""

import sys

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..base_model import SemanticIndexBase


class SemanticIndexBatchNeg(SemanticIndexBase):

    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 margin=0.3,
                 scale=30,
                 output_emb_size=None):
        super().__init__(pretrained_model, dropout, output_emb_size)

        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.sacle = scale

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):
        query_cls_embedding = self.get_pooled_embedding(query_input_ids,
                                                        query_token_type_ids,
                                                        query_position_ids,
                                                        query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(title_input_ids,
                                                        title_token_type_ids,
                                                        title_position_ids,
                                                        title_attention_mask)

        cosine_sim = paddle.matmul(query_cls_embedding,
                                   title_cls_embedding,
                                   transpose_y=True)

        # substract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(shape=[query_cls_embedding.shape[0]],
                                  fill_value=self.margin,
                                  dtype=paddle.get_default_dtype())

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.sacle

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64')
        labels = paddle.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss
