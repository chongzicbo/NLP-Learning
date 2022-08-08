# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: basic_make_uncased_model_cased.py
@time: 2022/6/29 15:12
"""
from __future__ import print_function

import os

os.environ.setdefault("TF_KERAS", "1")
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import to_array
import numpy as np

# bert配置
bert_dir = "E:\\working\\huada_bgi\\data\\pretrained_model\\bert\\chinese_roberta_wwm_ext_L-12_H-768_A-12\\"
config_path = os.path.join(bert_dir, "bert_config.json")
checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
dict_path = os.path.join(bert_dir, "vocab.txt")

token_dict = load_vocab(dict_path)
new_token_dict = token_dict.copy()
compound_tokens = []

for t, i in sorted(token_dict.items(), key=lambda s: s[1]):
    tokens = []
    if t.isalpha():
        tokens.extend([t[:1].upper() + t[1:], t.upper()])
    elif t[:2] == "##" and t[2:].isalpha():
        tokens.append(t.upper())
    for token in tokens:
        if token not in new_token_dict:
            compound_tokens.append([i])
            new_token_dict[token] = len(new_token_dict)

tokenizer = Tokenizer(new_token_dict, do_lower_case=False)

model = build_transformer_model(config_path, checkpoint_path, compound_tokens=compound_tokens)

text = u"Welcome to BEIJING."
tokens = tokenizer.tokenize(text)

print(tokens)

token_ids, segment_ids = tokenizer.encode(text)
token_ids, segment_ids = to_array([token_ids], [segment_ids])
print(model.predict([token_ids, segment_ids]))
