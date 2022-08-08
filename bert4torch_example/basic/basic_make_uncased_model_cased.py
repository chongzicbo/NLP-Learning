#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: basic_make_uncased_model_cased.py
@time: 2022/8/7 17:18
"""

# 通过简单修改词表，使得不区分大小写的模型有区分大小写的能力
# 基本思路：将英文单词大写化后添加到词表中，并修改模型Embedding层

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
import torch

root_model_path = "/mnt/e/working/huada_bgi/data/pretrained_model/huggingface/bert-base-chinese"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + "/bert4torch_pytorch_model.bin"

token_dict = load_vocab(vocab_path)
new_token_dict = token_dict.copy()
compound_tokens = []

for t, i in sorted(token_dict.items(), key=lambda s: s[1]):
    # 这里主要考虑两种情况：1、首字母大写；2、整个单词大写。
    # Python2下，新增了5594个token；Python3下，新增了5596个token。
    tokens = []
    if t.isalpha():
        tokens.extend([t[:1].upper() + t[1:], t.upper()])
    elif t[:2] == '##' and t[2:].isalpha():
        tokens.append(t.upper())
    for token in tokens:
        if token not in new_token_dict:
            compound_tokens.append([i])
            new_token_dict[token] = len(new_token_dict)

tokenizer = Tokenizer(new_token_dict, do_lower_case=False)
model = build_transformer_model(config_path, checkpoint_path, compound_tokens=compound_tokens)

text = u"Welcome to BEIJING."

print(text)

token_ids, segment_ids = tokenizer.encode(text)
token_ids, segment_ids = torch.tensor([token_ids]), torch.tensor([segment_ids])
model.eval()
with torch.no_grad():
    print(model([token_ids, segment_ids])[0])
