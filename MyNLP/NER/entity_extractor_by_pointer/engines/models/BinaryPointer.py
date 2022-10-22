#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: BinaryPointer.py
@time: 2022/9/19 15:53
"""
from abc import ABC

import torch
from torch import nn
from transformers import BertModel


class BinaryPointer(nn.Module, ABC):
    def __init__(self, num_labels):
        super(BinaryPointer, self).__init__()
        self.num_labels = num_labels
        hidden_size = 768
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.fc = nn.Linear(hidden_size, 2 * num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_hidden_states = self.bert_model(input_ids, attention_mask=attention_mask)[0]
        layer_hidden = self.layer_norm(bert_hidden_states)  # bert后接一个LayerNorm层
        fc_results = self.fc(layer_hidden)  # 接全连接层
        batch_size = fc_results.size(0)
        logits = fc_results.view(batch_size, -1, self.num_labels, 2)  # [batch_size,seq_len,num_labels,2]
        probs = torch.sigmoid(logits)  # 预测结果进行sigmoid,损失函数BCEWithLogitsLoss内部也会对预测结果进行sigmoid计算，相当于计算了两次sigmoid
        return logits, probs
