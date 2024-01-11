# -*-coding:utf-8 -*-

"""
# File       : blenderbot_test.py
# Time       ：2023/2/22 17:04
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

from transformers import BlenderbotTokenizer

model_path = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_path)

print(tokenizer.num_special_tokens_to_add())
print(tokenizer.all_special_tokens)
print(tokenizer.additional_special_tokens)
