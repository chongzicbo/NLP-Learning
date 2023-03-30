# -*-coding:utf-8 -*-

"""
# File       : prompt_text_classification.py
# Time       ：2023/3/8 11:18
# Author     ：chengbo
# version    ：python 3.8
# Description：https://zhuanlan.zhihu.com/p/462378735
"""

import os

import numpy as np

os.environ.setdefault("TF_KERAS", "1")
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding

print(tf.test.is_gpu_available())

model_dir = '/home/bocheng/data/pretrain_model/chinese/chinese_wwm_L-12_H-768_A-12'
config_path = os.path.join(model_dir, 'bert_config.json')
checkpoint_path = os.path.join(model_dir, 'bert_model.ckpt')
dict_path = os.path.join(model_dir, 'vocab.txt')
maxlen = 50

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)

tokenizer = Tokenizer(dict_path, do_lower_case=True)

prefix = u'接下来报导一则xx新闻。'
mask_idxs = [8, 9]
labels = [
    u'文化', u'娱乐', u'体育', u'财经', u'房产', u'汽车', u'教育', u'科技', u'军事', u'旅游', u'国际',
    u'证券', u'农业', u'电竞', u'民生'
]

##将xx这两个字符 用[mask]的id 代替
text = prefix + "长沙驿路高歌路虎极光汽车音响改装升级雷贝琴——纯净圆润"

token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
# print(token_ids)
token_ids[8] = tokenizer._token_mask_id
token_ids[9] = tokenizer._token_mask_id
token_ids = sequence_padding([token_ids])
segment_ids = sequence_padding([segment_ids])

# print(token_ids, token_ids.shape)
# print(tokenizer.encode(labels[0]))
label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])
# print('labe_ids:', label_ids, label_ids.shape)
y_pred = model.predict([token_ids, segment_ids])[:, mask_idxs] #(1, 2, 21128)
# print('y_pred', y_pred, y_pred.shape)
# print('label_ids[:, 0]', label_ids[:, 0])
mask1 = y_pred[:, 0, label_ids[:, 0] ] # (1, 15) label_ids[:, 0] 把label中的两个字 带入到 模型在 mask位置输出中
mask2 = y_pred[:, 1, label_ids[:, 1]]
# print('mask1:', mask1.shape)
y_pred = mask1 * mask2
# print(y_pred, y_pred.shape)
y_pred = y_pred.argmax(axis=1)
# print(labels[y_pred[0]])