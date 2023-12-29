#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: basic_masked_language_model.py
@time: 2022/8/6 19:39
"""
import os

os.environ.setdefault("TF_KERAS", "1")
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
import tensorflow as tf

# keras默认是执行图模式，debug困难；使用eager模式方便debug
tf.config.run_functions_eagerly(True)

bert_dir = (
    "/mnt/e/working/huada_bgi/data/pretrained_model/bert/chinese_L-12_H-768_A-12/"
)
config_path = os.path.join(bert_dir, "bert_config.json")
checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
dict_path = os.path.join(bert_dir, "vocab.txt")

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)  # 建立模型，加载权重

token_ids, segment_ids = tokenizer.encode("科学技术是第一生产力")

# mask掉“技术”
token_ids[3] = token_ids[4] = tokenizer._token_mask_id
token_ids, segment_ids = to_array([token_ids], [segment_ids])

# 用mlm模型预测被mask掉的部分
probas = model.predict([token_ids, segment_ids])[0]
print(tokenizer.decode(probas[3:5].argmax(axis=1)))  # 结果正是“技术”
