#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: convert_bert-base-chinese.py.py
@time: 2022/8/7 17:04
"""

# 由于key和框架的key没有完全对齐，主要里面用的都是Laynorm.gamma和Laynorm.beta来保存权重和偏置

import torch

state_dict = torch.load(
    '/mnt/e/working/huada_bgi/data/pretrained_model/huggingface/bert-base-chinese/pytorch_model.bin')
state_dict_new = {}
for k, v in state_dict.items():
    if 'LayerNorm.gamma' in k:
        k = k.replace('LayerNorm.gamma', 'LayerNorm.weight')
        state_dict_new[k] = v
    elif 'LayerNorm.beta' in k:
        k = k.replace('LayerNorm.beta', 'LayerNorm.bias')
        state_dict_new[k] = v
    else:
        state_dict_new[k] = v
torch.save(state_dict_new,
           '/mnt/e/working/huada_bgi/data/pretrained_model/huggingface/bert-base-chinese/bert4torch_pytorch_model.bin')
