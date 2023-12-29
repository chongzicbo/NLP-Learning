#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: basic_extract_features.py
@time: 2022/8/5 16:32
"""
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer

root_model_path = (
    "/mnt/e/working/huada_bgi/data/pretrained_model/huggingface/bert-base-chinese"
)
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + "/pytorch_model.bin"

tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path)

token_ids, segment_ids = tokenizer.encode("语言模型")
token_ids, segment_ids = torch.tensor([token_ids]), torch.tensor([segment_ids])

model.eval()
with torch.no_grad():
    print(model([token_ids, segment_ids])[0])
