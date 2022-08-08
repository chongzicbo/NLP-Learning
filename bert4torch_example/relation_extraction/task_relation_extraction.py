#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: task_relation_extraction.py
@time: 2022/8/7 17:35
"""
# 三元组抽取任务，基于“半指针-半标注”结构
# 文章介绍：https://kexue.fm/archives/7161
# 数据集：http://ai.baidu.com/broad/download?dataset=sked

import json
import numpy as np
from bert4torch.layers import LayerNorm
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
# from bert4torch.optimizers import ExponentialMovingAverage
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn

maxlen = 128
batch_size = 48
root_model_path = "/mnt/e/working/huada_bgi/data/pretrained_model/huggingface/bert-base-chinese"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + "/bert4torch_pytorch_model.bin"

predicate2id, id2predicate = {}, {}

